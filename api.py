import os
import re
import io
import sys
import time
import json
import logging
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor
import asyncio

# ---------- Instalação automática do Piper (se necessário) ----------
try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-tts"])
    from piper import PiperVoice, SynthesisConfig

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment

# ---------- Configuração de logs ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(processName)s | %(name)s | %(message)s",
)
logger = logging.getLogger("piper-api")

# ---------- Forçar CPU (sem CUDA) ----------
ort.set_default_logger_severity(3)

# ---------- Diretórios ----------
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"

VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# ---------- Contador global para atribuição de núcleos ----------
# Cada processo filho incrementa este contador com lock para pegar o próximo núcleo.
_cpu_counter = mp.Value('i', 0)
_cpu_lock = mp.Lock()

# ---------- Workers (ajuste conforme seus núcleos) ----------
# Para 16 vCPUs, sugiro 14 para TTS, 1 para mixagem, 1 para o FastAPI/OS.
# Se quiser usar TODOS os 16, ajuste TTS_WORKERS=15 e MIX_WORKERS=1 (soma=16)
TTS_WORKERS = int(os.getenv("TTS_WORKERS", 14))
MIX_WORKERS = int(os.getenv("MIX_WORKERS", 1))
logger.info(f"Workers: TTS={TTS_WORKERS} processos, Mix={MIX_WORKERS} processos")

# ---------- Inicializador dos workers TTS (executado em cada processo) ----------
def _init_tts_worker():
    # 1. Limita o ONNX Runtime e o OpenMP a 1 thread por processo
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ORT_NUM_THREADS"] = "1"
    ort.set_default_logger_severity(3)

    # 2. Define afinidade de CPU (core isolation) – apenas Linux
    with _cpu_lock:
        cpu_id = _cpu_counter.value
        _cpu_counter.value += 1

    total_cpus = os.cpu_count()
    if cpu_id >= total_cpus:
        cpu_id = cpu_id % total_cpus  # round-robin se houver mais workers que núcleos

    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"TTS Worker fixado ao núcleo {cpu_id}")
    except AttributeError:
        logger.warning("sched_setaffinity não suportado (não é Linux). Ignorando.")
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade no TTS: {e}")

    # 3. Cache de vozes por processo
    mod = sys.modules['__main__']
    mod._worker_voice_cache = {}

# ---------- Inicializador dos workers de mixagem ----------
def _init_mix_worker():
    # Limita threads de bibliotecas que eventualmente usem OpenMP
    os.environ["OMP_NUM_THREADS"] = "1"

    with _cpu_lock:
        cpu_id = _cpu_counter.value
        _cpu_counter.value += 1

    total_cpus = os.cpu_count()
    if cpu_id >= total_cpus:
        cpu_id = cpu_id % total_cpus

    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"Mix Worker fixado ao núcleo {cpu_id}")
    except AttributeError:
        logger.warning("sched_setaffinity não suportado (não é Linux). Ignorando.")
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade na mixagem: {e}")

# ---------- VoicePool (pool de instâncias PiperVoice) ----------
class VoicePool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = 2):
        import queue
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=False)
            self.pool.put(voice)

    def get(self, timeout: float = 2.0):
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
                    with open(meta_path) as f:
                        meta = json.load(f)
                        genero = meta.get("genero", "Desconhecido")
                except Exception:
                    pass
            register_voice(voice_name, model_path, config_path, {"genero": genero})
            logger.info(f"Voz registrada: {voice_name} ({genero})")
    for onnx_file in VOICES_DIR.glob("*.onnx"):
        voice_name = onnx_file.stem
        if voice_name in VOICE_PATHS:
            continue
        json_file = onnx_file.with_suffix(".onnx.json")
        if json_file.exists():
            register_voice(voice_name, str(onnx_file), str(json_file), {"genero": "Personalizada"})
            logger.info(f"Voz personalizada registrada: {voice_name}")

load_all_voices()
logger.info(f"Total de vozes disponíveis: {len(VOICE_PATHS)}")

# ---------- Caches de efeitos e ambiente (processo principal) ----------
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

# ---------- Funções executadas nos workers ----------

def get_voice_pool(voice_name):
    """Retorna um pool de vozes para o processo atual (cache local)."""
    mod = sys.modules['__main__']
    cache = getattr(mod, '_worker_voice_cache', None)
    if cache is None:
        cache = {}
        mod._worker_voice_cache = cache
    if voice_name not in cache:
        model_path, config_path = VOICE_PATHS[voice_name]
        pool = VoicePool(model_path, config_path, pool_size=1)  # pool_size=1 para manter simplicidade
        cache[voice_name] = pool
    return cache[voice_name]

def synthesize_text(voice_name, text, speed, noise_scale, noise_w_scale):
    """Sintetiza um trecho de texto e retorna (sample_rate, pcm_bytes)."""
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

def mix_and_export_task(segments_data, ambient_cfg, target_rate=22050):
    """
    Recebe segmentos (PCM ou efeitos), mixa, aplica ambiente (se configurado),
    normaliza e exporta para WAV (PCM 16-bit mono).
    Retorna bytes do arquivo WAV.
    """
    # Monta segmentos padronizados (mono, 16-bit, target_rate)
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

    # Normalização para -20 dBFS
    target_dBFS = -20.0
    if combined.dBFS != target_dBFS:
        combined = combined.apply_gain(target_dBFS - combined.dBFS)

    # Ambiente com cache local ao processo (usando atributo estático)
    if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
        cache_key = (ambient_cfg['file'], ambient_cfg.get('volume_db', -15))
        # Cache local no processo (não compartilhado entre processos)
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

    # Exporta para WAV em memória
    output = io.BytesIO()
    combined.export(output, format="wav")
    wav_bytes = output.getvalue()
    return wav_bytes

# ---------- Pools de processos com isolamento de núcleos ----------
tts_pool = ProcessPoolExecutor(
    max_workers=TTS_WORKERS,
    initializer=_init_tts_worker
)
mix_pool = ProcessPoolExecutor(
    max_workers=MIX_WORKERS,
    initializer=_init_mix_worker
)

# ---------- Modelos Pydantic ----------
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
app = FastAPI(title="Piper TTS API (Core Isolation + WAV)")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    t_total_start = time.perf_counter()

    # --- Validação e mapeamento de speakers ---
    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice:
            raise HTTPException(400, "Campo 'voice' é obrigatório quando não há speakers.")
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

    # --- Divisão do texto em partes (diálogo, texto, efeitos) ---
    parts = re.split(r'(\[.*?\])', req.text)
    parts = [p.strip() for p in parts if p.strip()]

    tts_tasks = []
    segment_data = [None] * len(parts)
    loop = asyncio.get_running_loop()

    for idx, part in enumerate(parts):
        # Detecta marcador de papel (ex: [joao])
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
            continue

        # Verifica se é um efeito sonoro (ex: "tosse", "porta")
        if part in req.effects:
            effect_file = req.effects[part]
            voice_for_eff = speaker_map[current_role][0] if is_dialog and current_role else req.voice
            segment_data[idx] = {'effect': effect_file, 'voice': voice_for_eff}
            continue

        # Síntese TTS
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

    # --- Executa todas as sínteses em paralelo ---
    t_synth_start = time.perf_counter()
    if tts_tasks:
        futures, indices = zip(*tts_tasks)
        results = await asyncio.gather(*futures)
        for (sr, pcm), idx in zip(results, indices):
            segment_data[idx] = {'pcm_bytes': pcm, 'sample_rate': sr}
    t_synth_end = time.perf_counter()

    # --- Prepara payload para mixagem ---
    mix_payload = [d for d in segment_data if d is not None]

    # Serialização compatível Pydantic v1/v2
    try:
        ambient_dict = req.ambient.model_dump()
    except AttributeError:
        ambient_dict = req.ambient.dict()

    # --- Mixagem e exportação WAV ---
    t_mix_start = time.perf_counter()
    try:
        wav_bytes = await loop.run_in_executor(mix_pool, mix_and_export_task,
                                               mix_payload, ambient_dict, 22050)
    except Exception as e:
        logger.error(f"Falha na mixagem/exportação: {e}")
        raise HTTPException(500, f"Erro na mixagem: {str(e)}")
    t_mix_end = time.perf_counter()

    # --- Log de desempenho ---
    synth_duration = t_synth_end - t_synth_start
    mix_duration = t_mix_end - t_mix_start
    total_time = time.perf_counter() - t_total_start
    audio_est = sum(len(d.get('pcm_bytes', b'')) / 2 / 22050 for d in mix_payload if 'pcm_bytes' in d)

    logger.info(
        f"Concluída | total={total_time:.3f}s | synth={synth_duration:.3f}s | "
        f"mix={mix_duration:.3f}s | audio={audio_est:.1f}s"
    )

    return Response(content=wav_bytes, media_type="audio/wav")

# ---------- Endpoints de saúde ----------
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

# ---------- Ponto de entrada ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
