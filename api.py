import os
import re
import io
import sys
import time
import json
import logging
import subprocess
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor
import asyncio
from collections import defaultdict

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

# ---------- Contador global para afinidade de núcleos ----------
_cpu_counter = mp.Value('i', 0)
_cpu_lock = mp.Lock()

# ---------- Workers (configuração otimizada) ----------
TTS_WORKERS = int(os.getenv("TTS_WORKERS", 8))
MIX_WORKERS = int(os.getenv("MIX_WORKERS", 5))
logger.info(f"api nova! Workers: TTS={TTS_WORKERS} processos, Mix={MIX_WORKERS} processos")

# ---------- Inicializador dos workers TTS ----------
def _init_tts_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ORT_NUM_THREADS"] = "1"
    ort.set_default_logger_severity(3)

    with _cpu_lock:
        cpu_id = _cpu_counter.value
        _cpu_counter.value += 1

    total_cpus = os.cpu_count()
    if cpu_id >= total_cpus:
        cpu_id = cpu_id % total_cpus

    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"TTS Worker fixado ao núcleo {cpu_id}")
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade no TTS: {e}")

    mod = sys.modules['__main__']
    mod._worker_voice_cache = {}

# ---------- Inicializador dos workers de mixagem ----------
def _init_mix_worker():
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
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade na mixagem: {e}")

# ---------- VoicePool ----------
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

# ---------- Caches para efeitos e ambiente (processo principal) ----------
effect_cache: Dict[Tuple[str, str], bytes] = {}
ambient_cache: Dict[Tuple[str, float], bytes] = {}

def load_effect_bytes(voice_name: str, effect_file: str) -> bytes:
    cache_key = (voice_name, effect_file)
    if cache_key in effect_cache:
        return effect_cache[cache_key]

    voice_dir = VOICES_DIR / voice_name
    effect_path = voice_dir / effect_file
    if not effect_path.exists():
        effect_path = EFFECTS_DIR / effect_file
    if not effect_path.exists():
        raise FileNotFoundError(f"Efeito '{effect_file}' não encontrado")

    with open(effect_path, 'rb') as f:
        data = f.read()
    effect_cache[cache_key] = data
    return data

def load_ambient_bytes(ambient_file: str, volume_db: float) -> bytes:
    cache_key = (ambient_file, volume_db)
    if cache_key in ambient_cache:
        return ambient_cache[cache_key]

    ambient_path = AMBIENT_DIR / f"{ambient_file}.wav"
    if not ambient_path.exists():
        raise FileNotFoundError(f"Ambiente '{ambient_file}.wav' não encontrado")

    with open(ambient_path, 'rb') as f:
        data = f.read()
    ambient_cache[cache_key] = data
    return data

# ---------- Funções executadas nos workers ----------
def get_voice_pool(voice_name):
    mod = sys.modules['__main__']
    cache = getattr(mod, '_worker_voice_cache', None)
    if cache is None:
        cache = {}
        mod._worker_voice_cache = cache
    if voice_name not in cache:
        model_path, config_path = VOICE_PATHS[voice_name]
        pool = VoicePool(model_path, config_path, pool_size=1)
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

def mix_and_export_task(segments_data, ambient_cfg, target_rate=22050):
    """
    Mixa múltiplos segmentos usando FFmpeg (único comando) e retorna bytes WAV.
    """
    t0 = time.perf_counter()
    temp_files = []
    ffmpeg_cmd = ["ffmpeg", "-y"]

    try:
        # 1. Prepara arquivos temporários para cada segmento
        for data in segments_data:
            if 'pcm_bytes' in data:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(data['pcm_bytes'])
                    temp_files.append(f.name)
            elif 'effect' in data:
                # Efeito já está em cache no processo principal, mas aqui precisamos do arquivo.
                # Como estamos no worker, não temos o cache. Carregamos do disco.
                voice_dir = VOICES_DIR / data['voice']
                effect_path = voice_dir / data['effect']
                if not effect_path.exists():
                    effect_path = EFFECTS_DIR / data['effect']
                if not effect_path.exists():
                    raise FileNotFoundError(f"Efeito '{data['effect']}' não encontrado")
                # Copia para temp para padronizar
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    with open(effect_path, 'rb') as src:
                        f.write(src.read())
                    temp_files.append(f.name)
            else:
                continue

        # 2. Adiciona ambiente, se habilitado
        if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
            ambient_path = AMBIENT_DIR / f"{ambient_cfg['file']}.wav"
            if ambient_path.exists():
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    with open(ambient_path, 'rb') as src:
                        f.write(src.read())
                    temp_files.append(f.name)

        if not temp_files:
            raise ValueError("Nenhum arquivo para mixar")

        # 3. Monta comando FFmpeg
        for f in temp_files:
            ffmpeg_cmd.extend(["-i", f])

        filter_complex = f"amix=inputs={len(temp_files)}:duration=longest"
        ffmpeg_cmd.extend([
            "-filter_complex", filter_complex,
            "-ar", str(target_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            "pipe:1"
        ])

        # 4. Executa
        result = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        wav_bytes = result.stdout

        t_total = time.perf_counter() - t0
        logger.debug(f"Mixagem FFmpeg concluída em {t_total:.3f}s | {len(temp_files)} arquivos")
        return wav_bytes

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg erro: {e.stderr.decode()}")
        raise RuntimeError("Falha na mixagem com FFmpeg")
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass

# ---------- Pools de processos ----------
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
app = FastAPI(title="Piper TTS API (Otimizada + FFmpeg)")

# ---------- Estatísticas acumuladas para relatório ----------
stats = defaultdict(list)
stats_lock = asyncio.Lock()

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    global stats
    t_total_start = time.perf_counter()

    # --- 1. Parse e validação ---
    t_parse_start = time.perf_counter()
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

    t_parse = time.perf_counter() - t_parse_start

    # --- 2. Divisão do texto ---
    t_split_start = time.perf_counter()
    parts = re.split(r'(\[.*?\])', req.text)
    parts = [p.strip() for p in parts if p.strip()]
    t_split = time.perf_counter() - t_split_start

    # --- 3. Preparação das tarefas TTS e mix ---
    t_prep_start = time.perf_counter()
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

    t_prep = time.perf_counter() - t_prep_start

    # --- 4. Execução das sínteses em paralelo ---
    t_synth_total_start = time.perf_counter()
    synth_times = []
    if tts_tasks:
        futures, indices = zip(*tts_tasks)
        results = await asyncio.gather(*futures)
        for (sr, pcm), idx in zip(results, indices):
            segment_data[idx] = {'pcm_bytes': pcm, 'sample_rate': sr}
    t_synth_total = time.perf_counter() - t_synth_total_start

    # --- 5. Preparação do payload para mixagem ---
    t_mix_prep_start = time.perf_counter()
    mix_payload = [d for d in segment_data if d is not None]
    try:
        ambient_dict = req.ambient.model_dump()
    except AttributeError:
        ambient_dict = req.ambient.dict()
    t_mix_prep = time.perf_counter() - t_mix_prep_start

    # --- 6. Mixagem e exportação ---
    t_mix_start = time.perf_counter()
    try:
        wav_bytes = await loop.run_in_executor(mix_pool, mix_and_export_task,
                                               mix_payload, ambient_dict, 22050)
    except Exception as e:
        logger.error(f"Falha na mixagem/exportação: {e}")
        raise HTTPException(500, f"Erro na mixagem: {str(e)}")
    t_mix = time.perf_counter() - t_mix_start

    # --- 7. Tempos finais ---
    t_total = time.perf_counter() - t_total_start

    # --- Log detalhado da requisição ---
    logger.info(
        f"⏱️ Tempos | total={t_total:.3f}s | parse={t_parse:.3f}s | split={t_split:.3f}s | "
        f"prep={t_prep:.3f}s | synth_total={t_synth_total:.3f}s | mix_prep={t_mix_prep:.3f}s | mix={t_mix:.3f}s"
    )

    # --- Acumula estatísticas ---
    async with stats_lock:
        stats['total'].append(t_total)
        stats['parse'].append(t_parse)
        stats['split'].append(t_split)
        stats['prep'].append(t_prep)
        stats['synth_total'].append(t_synth_total)
        stats['mix_prep'].append(t_mix_prep)
        stats['mix'].append(t_mix)

    return Response(content=wav_bytes, media_type="audio/wav")

# ---------- Endpoint para relatório de estatísticas ----------
@app.get("/stats")
async def get_stats():
    """Retorna um relatório com médias, máximos, mínimos de cada etapa."""
    async with stats_lock:
        if not stats['total']:
            return {"message": "Nenhuma requisição processada ainda."}

        report = {}
        for key, values in stats.items():
            report[key] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p95": sorted(values)[int(0.95 * len(values))] if len(values) > 1 else values[0],
            }
        return report

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
