import os
import re
import io
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor
import asyncio

# ---------- Instalação automática do PyAV (se necessário) ----------def encode_webm_opus(pcm_bytes, sample_rate, channels=1, bitrate=64000):
    output = io.BytesIO()
    container = av.open(output, mode='w', format='webm')
    stream = container.add_stream('opus', rate=sample_rate, channels=channels)
    stream.bit_rate = bitrate

    audio_frame = av.AudioFrame.from_ndarray(
        np.frombuffer(pcm_bytes, dtype=np.int16).reshape(-1, channels),
        format='s16',
        layout='mono' if channels == 1 else 'stereo'
    )
    audio_frame.sample_rate = sample_rate

    for packet in stream.encode(audio_frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    return output.getvalue()
try:
    import av
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "av"])
    import av

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from piper import PiperVoice, SynthesisConfig
from pydub import AudioSegment

# ---------- Configuração de logs ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(processName)s | %(name)s | %(message)s",
)
logger = logging.getLogger("piper-api")

# ---------- Forçar CPU ----------
ort.set_default_logger_severity(3)

# ---------- Diretórios ----------
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"

VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# ---------- Workers ----------
TTS_WORKERS = int(os.getenv("TTS_WORKERS", 10))
MIX_WORKERS = int(os.getenv("MIX_WORKERS", 4))
logger.info(f"Workers: TTS={TTS_WORKERS}, Mix={MIX_WORKERS}")

# ---------- Inicializador dos workers TTS ----------
def _init_tts_worker():
    ort.set_default_logger_severity(3)
    mod = sys.modules['__main__']
    mod._worker_voice_cache = {}

# ---------- VoicePool ----------
class VoicePool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = 2):
        import queue
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=False)
            self.pool.put(voice)

    def get(self, timeout=2.0):
        return self.pool.get(timeout=timeout)

    def put(self, voice):
        self.pool.put(voice)

# ---------- Registro de vozes ----------
VOICE_PATHS: Dict[str, Tuple[str, str]] = {}
voices_metadata: Dict[str, dict] = {}

def register_voice(voice_name, model_path, config_path, meta):
    VOICE_PATHS[voice_name] = (model_path, config_path)
    voices_metadata[voice_name] = meta

def load_all_voices():
    for item in VOICES_DIR.iterdir():
        if item.is_dir():
            voice_name = item.name
            onnx_files = list(item.glob("*.onnx"))
            if not onnx_files:
                continue
            model_path = str(onnx_files[0])
            base_name = onnx_files[0].stem
            json_path = item / f"{base_name}.onnx.json"
            if not json_path.exists():
                json_candidates = list(item.glob("*.json"))
                if not json_candidates:
                    continue
                json_path = json_candidates[0]
            config_path = str(json_path)
            genero = "Desconhecido"
            meta_path = item / f"{voice_name}.json"
            if meta_path.exists():
                try:
                    import json
                    with open(meta_path) as f:
                        meta = json.load(f)
                        genero = meta.get("genero", "Desconhecido")
                except:
                    pass
            register_voice(voice_name, model_path, config_path, {"genero": genero})
            logger.info(f"✅ Voz registrada: {voice_name} ({genero})")
    for onnx_file in VOICES_DIR.glob("*.onnx"):
        voice_name = onnx_file.stem
        if voice_name in VOICE_PATHS:
            continue
        json_file = onnx_file.with_suffix(".onnx.json")
        if json_file.exists():
            register_voice(voice_name, str(onnx_file), str(json_file), {"genero": "Personalizada"})
            logger.info(f"✅ Voz personalizada registrada: {voice_name}")

load_all_voices()
logger.info(f"Total de vozes disponíveis: {len(VOICE_PATHS)}")

# ---------- Cache de efeitos e ambiente (processo principal) ----------
effect_cache: Dict[Tuple[str, str], AudioSegment] = {}
ambient_cache: Dict[Tuple[str, float], AudioSegment] = {}

def load_effect(voice_name, effect_file):
    cache_key = (voice_name, effect_file)
    if cache_key in effect_cache:
        return effect_cache[cache_key]
    voice_dir = VOICES_DIR / voice_name
    effect_path = voice_dir / effect_file
    if not effect_path.exists():
        effect_path = EFFECTS_DIR / effect_file
    if not effect_path.exists():
        raise FileNotFoundError(f"Efeito '{effect_file}' não encontrado")
    seg = AudioSegment.from_wav(str(effect_path))
    effect_cache[cache_key] = seg
    return seg

def load_ambient(ambient_file, volume_db):
    cache_key = (ambient_file, volume_db)
    if cache_key in ambient_cache:
        return ambient_cache[cache_key]
    ambient_path = AMBIENT_DIR / f"{ambient_file}.wav"
    if not ambient_path.exists():
        raise FileNotFoundError(f"Ambiente '{ambient_file}.wav' não encontrado")
    seg = AudioSegment.from_wav(str(ambient_path))
    seg = seg + volume_db
    ambient_cache[cache_key] = seg
    return seg

# ---------- Funções dos workers ----------

def get_voice_pool(voice_name):
    mod = sys.modules['__main__']
    cache = getattr(mod, '_worker_voice_cache', None)
    if cache is None:
        cache = {}
        mod._worker_voice_cache = cache
    if voice_name not in cache:
        model_path, config_path = VOICE_PATHS[voice_name]
        pool = VoicePool(model_path, config_path)
        cache[voice_name] = pool
    return cache[voice_name]

def synthesize_text(voice_name, text, speed, noise_scale, noise_w_scale):
    pool = get_voice_pool(voice_name)
    voice = pool.get()
    try:
        config = SynthesisConfig(
            length_scale=speed,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            volume=1.0
        )
        chunk_generator = voice.synthesize(text, syn_config=config)
        audio_bytes = b''.join(chunk.audio_int16_bytes for chunk in chunk_generator)
        sample_rate = voice.config.sample_rate
        return sample_rate, audio_bytes
    finally:
        pool.put(voice)
def encode_webm_opus(pcm_bytes, sample_rate, channels=1, bitrate=64000):
    output = io.BytesIO()
    container = av.open(output, mode='w', format='webm')
    stream = container.add_stream('opus', rate=sample_rate, channels=channels)
    stream.bit_rate = bitrate

    audio_frame = av.AudioFrame.from_ndarray(
        np.frombuffer(pcm_bytes, dtype=np.int16).reshape(-1, channels),
        format='s16',
        layout='mono' if channels == 1 else 'stereo'
    )
    audio_frame.sample_rate = sample_rate

    for packet in stream.encode(audio_frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    return output.getvalue()

def mix_and_export_task(segments_data, ambient_cfg, target_rate=22050):
    # Monta segmentos padronizados
    audio_segments = []
    for data in segments_data:
        if 'pcm_bytes' in data:
            seg = AudioSegment(
                data=data['pcm_bytes'],
                sample_width=2,
                frame_rate=data['sample_rate'],
                channels=1
            )
        elif 'effect' in data:
            voice_dir = VOICES_DIR / data['voice']
            effect_path = voice_dir / data['effect']
            if not effect_path.exists():
                effect_path = EFFECTS_DIR / data['effect']
            seg = AudioSegment.from_wav(str(effect_path))
        else:
            continue
        seg = seg.set_channels(1).set_sample_width(2).set_frame_rate(target_rate)
        audio_segments.append(seg)

    if not audio_segments:
        raise ValueError("Nenhum segmento para mixagem")

    combined = AudioSegment.empty()
    for seg in audio_segments:
        combined += seg

    # Normalização
    target_dBFS = -20.0
    if combined.dBFS != target_dBFS:
        combined = combined.apply_gain(target_dBFS - combined.dBFS)

    # Ambiente com cache local ao processo
    if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
        cache_key = (ambient_cfg['file'], ambient_cfg.get('volume_db', -15))
        if not hasattr(mix_and_export_task, '_ambient_cache'):
            mix_and_export_task._ambient_cache = {}
        if cache_key not in mix_and_export_task._ambient_cache:
            ambient_path = AMBIENT_DIR / f"{ambient_cfg['file']}.wav"
            ambient = AudioSegment.from_wav(str(ambient_path))
            ambient = ambient + ambient_cfg.get('volume_db', -15)
            ambient = ambient.set_channels(1).set_sample_width(2).set_frame_rate(target_rate)
            mix_and_export_task._ambient_cache[cache_key] = ambient
        ambient = mix_and_export_task._ambient_cache[cache_key]

        if len(ambient) < len(combined):
            ambient = ambient * ((len(combined) // len(ambient)) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)

    # Codificação WebM/Opus via PyAV
    pcm_bytes = combined.raw_data
    webm_bytes = encode_webm_opus(pcm_bytes, target_rate)
    return webm_bytes

# ---------- Pools de processos ----------
tts_pool = ProcessPoolExecutor(max_workers=TTS_WORKERS, initializer=_init_tts_worker)
mix_pool = ProcessPoolExecutor(max_workers=MIX_WORKERS)

# ---------- Modelos ----------
class AmbientConfig(BaseModel):
    enabled: bool = False
    file: Optional[str] = None
    volume_db: float = Field(default=-15.0, ge=-60.0, le=12.0)

class SpeakerMapping(BaseModel):
    role: str
    voice: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    noise_w_scale: Optional[float] = Field(default=None, ge=0.0, le=2.0)

class TTSRequest(BaseModel):
    voice: Optional[str] = None
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=1.5)
    noise_w_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# ---------- FastAPI ----------
app = FastAPI(title="Piper TTS API (Multiprocessing)")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    t_total_start = time.perf_counter()

    # --- Validação e mapeamento de speakers ---
    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice:
            raise HTTPException(400, "Campo 'voice' é obrigatório")
        if req.voice not in VOICE_PATHS:
            raise HTTPException(404, f"Voz '{req.voice}' não encontrada")
        speaker_map = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        current_role = None
    else:
        speaker_map = {}
        for spk in req.speakers:
            noise_s = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
            noise_w = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
            speaker_map[spk.role] = (spk.voice, spk.speed, noise_s, noise_w)
        for role, (v, _, _, _) in speaker_map.items():
            if v not in VOICE_PATHS:
                raise HTTPException(404, f"Voz '{v}' (speaker '{role}') não encontrada")
        current_role = None

    # --- Divisão e planejamento ---
    parts = re.split(r'(\[.*?\])', req.text)
    parts = [p.strip() for p in parts if p.strip()]

    tts_tasks = []
    segment_data = [None] * len(parts)
    loop = asyncio.get_running_loop()

    for idx, part in enumerate(parts):
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
            continue

        if part in req.effects:
            effect_file = req.effects[part]
            voice_for_eff = speaker_map[current_role][0] if is_dialog and current_role else req.voice
            segment_data[idx] = {'effect': effect_file, 'voice': voice_for_eff}
            continue

        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Speaker não definido. Use [papel] antes do texto.")
            voice_name, speed, noise_s, noise_w = speaker_map[current_role]
        else:
            voice_name = req.voice
            speed = req.speed
            noise_s = req.noise_scale
            noise_w = req.noise_w_scale

        fut = loop.run_in_executor(tts_pool, synthesize_text,
                                   voice_name, part, speed, noise_s, noise_w)
        tts_tasks.append((fut, idx))

    # --- Síntese em paralelo ---
    t_synth_start = time.perf_counter()
    if tts_tasks:
        futures, indices = zip(*tts_tasks)
        results = await asyncio.gather(*futures)
        for (sr, pcm), idx in zip(results, indices):
            segment_data[idx] = {'pcm_bytes': pcm, 'sample_rate': sr}
        t_synth_end = time.perf_counter()
    else:
        t_synth_end = time.perf_counter()

    # --- Preparar payload para mixagem ---
    mix_payload = [d for d in segment_data if d is not None]

    # Serialização compatível Pydantic v1/v2
    try:
        ambient_dict = req.ambient.model_dump()
    except AttributeError:
        ambient_dict = req.ambient.dict()

    # --- Mixagem e exportação ---
    t_mix_start = time.perf_counter()
    try:
        mixed_bytes = await loop.run_in_executor(mix_pool, mix_and_export_task,
                                                 mix_payload, ambient_dict, 22050)
    except Exception as e:
        logger.error(f"❌ Falha na mixagem/exportação: {e}")
        raise HTTPException(500, f"Erro na mixagem: {str(e)}")
    t_mix_end = time.perf_counter()

    # --- Cálculo dos tempos e duração do áudio ---
    synth_duration = t_synth_end - t_synth_start
    mix_duration = t_mix_end - t_mix_start
    total_time = time.perf_counter() - t_total_start
    audio_est = sum(len(d.get('pcm_bytes', b'')) / 2 / 22050 for d in mix_payload if 'pcm_bytes' in d)

    # Única linha de log com todos os tempos
    logger.info(
        f"✅ Concluída | total={total_time:.3f}s | synth={synth_duration:.3f}s | "
        f"mix={mix_duration:.3f}s | audio={audio_est:.1f}s"
    )

    return Response(content=mixed_bytes, media_type="audio/webm")

# ---------- Health ----------
@app.get("/started")
async def started():
    return Response(status_code=200, content="started")

@app.get("/ready")
async def ready():
    if VOICE_PATHS:
        return Response(status_code=200, content="ready")
    return Response(status_code=503, content="loading models")

@app.get("/live")
async def live():
    return Response(status_code=200, content="alive")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "voices": list(VOICE_PATHS.keys()),
        "workers": {"tts": TTS_WORKERS, "mix": MIX_WORKERS}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
