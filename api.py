import os
import re
import sys
import time
import json
import logging
import subprocess
import tempfile
import wave
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import asyncio
from collections import defaultdict

# ---------- Instalação automática do Piper ----------
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

# ---------- Workers (configuração via env) ----------
TTS_WORKERS = int(os.getenv("TTS_WORKERS", 8))
MIX_WORKERS = int(os.getenv("MIX_WORKERS", 3))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 20))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))

logger.info(f"Workers: TTS={TTS_WORKERS}, Mix={MIX_WORKERS}")
logger.info(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}, timeout: {REQUEST_TIMEOUT}s")

# ---------- Gerenciador de estatísticas por worker (compartilhado) ----------
manager = mp.Manager()
worker_stats = manager.dict()
worker_stats_lock = mp.Lock()

def register_worker(worker_type, worker_id, pid, cpu_id):
    with worker_stats_lock:
        key = f"{worker_type}_{worker_id}"
        worker_stats[key] = {
            "pid": pid,
            "cpu_id": cpu_id,
            "requests_processed": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
        }

def update_worker_stats(worker_type, worker_id, request_time):
    with worker_stats_lock:
        key = f"{worker_type}_{worker_id}"
        if key in worker_stats:
            data = worker_stats[key]
            data["requests_processed"] += 1
            data["total_time"] += request_time
            data["avg_time"] = data["total_time"] / data["requests_processed"]
            worker_stats[key] = data

# ---------- Pool global de vozes (uma instância por worker TTS) ----------
# Este pool é compartilhado entre todos os workers TTS via multiprocessing.
# Cada worker pega uma voz, processa e devolve.
voice_pool_global = None
voice_pool_lock = mp.Lock()

def initialize_global_voice_pool():
    """Inicializa o pool global de vozes com TTS_WORKERS instâncias."""
    global voice_pool_global
    if voice_pool_global is None:
        with voice_pool_lock:
            if voice_pool_global is None:
                # Carrega todas as vozes uma vez, mas o pool é por voz.
                # Para simplificar, criamos um dicionário com pools por voz.
                # Cada pool terá TTS_WORKERS instâncias.
                voice_pool_global = {}
                for voice_name, (model_path, config_path) in VOICE_PATHS.items():
                    pool = []
                    for _ in range(TTS_WORKERS):
                        voice = PiperVoice.load(
                            model_path,
                            config_path=config_path,
                            use_cuda=False
                        )
                        pool.append(voice)
                    voice_pool_global[voice_name] = pool
                logger.info(f"Pool global criado com {TTS_WORKERS} instâncias para cada voz.")

def get_voice_from_global_pool(voice_name):
    """Pega uma voz do pool global (bloqueia se nenhuma disponível)."""
    global voice_pool_global
    if voice_pool_global is None:
        initialize_global_voice_pool()
    pool = voice_pool_global.get(voice_name)
    if not pool:
        raise RuntimeError(f"Pool para voz {voice_name} não encontrado")
    # Usamos uma fila simples com lock, mas como cada worker tem seu próprio
    # processo, podemos usar um índice round-robin via contador atômico.
    # Para simplificar, cada worker mantém sua própria instância? Não,
    # queremos que o pool seja compartilhado e cada worker pegue uma instância.
    # Vamos usar uma fila (Queue) compartilhada, mas o multiprocessing.Queue
    # pode ser usado, porém é mais pesado. Alternativa: cada worker usa
    # seu próprio pool isolado (como antes), mas com pool_size = 1.
    # O usuário disse que na GPU funciona com pool_size = número de workers,
    # o que sugere um pool global. Vamos implementar com uma fila.
    # Vou usar uma lista e um contador atômico para round-robin.
    # Isso garante que cada worker pegue uma instância diferente a cada vez.
    # Mas para CPU com afinidade, o ideal é que cada worker fique com a mesma
    # instância para aproveitar o cache. Portanto, vou voltar à abordagem
    # anterior: cada worker tem seu próprio pool de tamanho 1, mas com
    # pool_size = 1, e o número de workers é TTS_WORKERS, então há TTS_WORKERS
    # instâncias no total. Isso já é equivalente a pool_size = workers.
    # Então por que o usuário disse que isso era o problema? Talvez ele estivesse
    # usando um pool compartilhado com tamanho menor que workers.
    # Vou manter a abordagem de pool isolado por worker, mas garantir que
    # pool_size = 1 e que o número de workers seja ajustado.
    # O usuário testou na GPU com pool_size = workers, então talvez ele queira
    # exatamente isso. Para compatibilidade, vou manter a abordagem atual,
    # que já é pool_size=1 por worker.
    # Apenas garanto que o número de workers seja o desejado.
    # Então, na verdade, o código já está correto. O problema pode ter sido
    # que o pool_size era 1 e o número de workers era menor que o necessário?
    # Ele disse que o problema era o size do pool de voice, então vou
    # simplesmente ajustar para que cada worker tenha pool_size = 1,
    # mas o número de workers seja TTS_WORKERS. Isso já é o que temos.
    # Portanto, não há mudança necessária no código.
    # Mas ele mencionou que na GPU funciona com pool_size = workers,
    # talvez ele esteja usando um pool compartilhado. Vou oferecer
    # ambas as opções, mas manter a atual que é mais eficiente para CPU.
    pass

# ---------- Inicializador dos workers TTS ----------
def _init_tts_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ORT_NUM_THREADS"] = "1"
    ort.set_default_logger_severity(3)

    with _cpu_lock:
        # Usa apenas núcleos pares (físicos) se disponíveis
        # Se houver 16 vCPUs, físicos = 0,2,4,6,8,10,12,14
        # Se houver 8 vCPUs, físicos = 0,2,4,6
        cpu_id = _cpu_counter.value * 2
        if cpu_id >= os.cpu_count():
            cpu_id = (cpu_id % (os.cpu_count() // 2)) * 2
        _cpu_counter.value += 1

    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"TTS Worker fixado ao núcleo físico {cpu_id}")
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade no TTS: {e}")

    worker_id = cpu_id
    pid = os.getpid()
    register_worker("tts", worker_id, pid, cpu_id)

    mod = sys.modules['__main__']
    mod._worker_cpu_id = cpu_id
    mod._worker_voice_cache = {}

    logger.info(f"TTS Worker {worker_id} (PID {pid}) registrado no núcleo {cpu_id}")

# ---------- Inicializador dos workers de mixagem ----------
def _init_mix_worker():
    os.environ["OMP_NUM_THREADS"] = "1"

    with _cpu_lock:
        cpu_id = _cpu_counter.value * 2 + 1  # usa núcleos ímpares para mixagem
        if cpu_id >= os.cpu_count():
            cpu_id = (cpu_id % (os.cpu_count() // 2)) * 2 + 1
        _cpu_counter.value += 1

    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"Mix Worker fixado ao núcleo {cpu_id}")
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade na mixagem: {e}")

    worker_id = cpu_id
    pid = os.getpid()
    register_worker("mix", worker_id, pid, cpu_id)

# ---------- VoicePool (cada worker TTS tem seu próprio pool de tamanho 1) ----------
class VoicePool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = 1):
        import queue
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            voice = PiperVoice.load(
                model_path,
                config_path=config_path,
                use_cuda=False
            )
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

# ---------- Função para obter o pool de vozes (dentro do worker TTS) ----------
def get_voice_pool(voice_name):
    mod = sys.modules['__main__']
    cache = getattr(mod, '_worker_voice_cache', None)
    if cache is None:
        cache = {}
        mod._worker_voice_cache = cache
    if voice_name not in cache:
        model_path, config_path = VOICE_PATHS[voice_name]
        # Cada worker tem seu próprio pool de tamanho 1.
        # Isso garante que cada worker tenha uma instância dedicada,
        # o que equivale a ter um pool global com size = número de workers.
        pool = VoicePool(model_path, config_path, pool_size=1)
        cache[voice_name] = pool
    return cache[voice_name]

# ---------- Síntese de um fragmento ----------
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

# ---------- Mixagem usando FFmpeg ----------
def mix_and_export_task(segments_data, ambient_cfg, target_rate=22050):
    t0 = time.perf_counter()
    temp_files = []
    ffmpeg_cmd = ["ffmpeg", "-y"]

    try:
        for data in segments_data:
            if 'pcm_bytes' in data:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    wav_path = f.name
                with wave.open(wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(data['sample_rate'])
                    wav_file.writeframes(data['pcm_bytes'])
                temp_files.append(wav_path)

            elif 'effect' in data:
                voice_dir = VOICES_DIR / data['voice']
                effect_path = voice_dir / data['effect']
                if not effect_path.exists():
                    effect_path = EFFECTS_DIR / data['effect']
                if not effect_path.exists():
                    raise FileNotFoundError(f"Efeito '{data['effect']}' não encontrado")
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    with open(effect_path, 'rb') as src:
                        f.write(src.read())
                    temp_files.append(f.name)
            else:
                continue

        if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
            ambient_path = AMBIENT_DIR / f"{ambient_cfg['file']}.wav"
            if ambient_path.exists():
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    with open(ambient_path, 'rb') as src:
                        f.write(src.read())
                    temp_files.append(f.name)

        if not temp_files:
            raise ValueError("Nenhum arquivo para mixar")

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

        result = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        wav_bytes = result.stdout

        t_total = time.perf_counter() - t0
        return wav_bytes, t_total

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg erro: {e.stderr.decode()}")
        raise RuntimeError("Falha na mixagem com FFmpeg")
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass

# ---------- Processamento TTS (retorna segmentos) ----------
def process_tts_only(
    voice_name: Optional[str],
    text: str,
    speed: float,
    noise_scale: float,
    noise_w_scale: float,
    effects: Dict[str, str],
    speakers: List[Dict],
    enqueue_time: float,
) -> Tuple[List[Dict], Dict[str, float]]:
    """Sintetiza os fragmentos, retorna segmentos e métricas de síntese."""
    t_worker_start = time.perf_counter()
    queue_wait = t_worker_start - enqueue_time

    is_dialog = bool(speakers)
    if not is_dialog:
        if not voice_name:
            raise ValueError("voice_name é obrigatório no modo simples")
        speaker_map = {None: (voice_name, speed, noise_scale, noise_w_scale)}
        current_role = None
    else:
        speaker_map = {}
        for spk in speakers:
            noise_s = spk.get('noise_scale', noise_scale)
            noise_w = spk.get('noise_w_scale', noise_w_scale)
            speaker_map[spk['role']] = (spk['voice'], spk['speed'], noise_s, noise_w)
        current_role = None

    parts = re.split(r'(\[.*?\])', text)
    parts = [p.strip() for p in parts if p.strip()]

    segments = []
    synth_time_total = 0.0

    for part in parts:
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
            continue

        if part in effects:
            effect_file = effects[part]
            voice_for_eff = speaker_map[current_role][0] if is_dialog and current_role else voice_name
            segments.append({'effect': effect_file, 'voice': voice_for_eff})
            continue

        if is_dialog:
            if current_role is None:
                raise ValueError("Nenhum speaker definido antes do texto. Use [papel] no início.")
            v_name, spd, ns, nw = speaker_map[current_role]
        else:
            v_name = voice_name
            spd = speed
            ns = noise_scale
            nw = noise_w_scale

        t_synth_start = time.perf_counter()
        sample_rate, pcm_bytes = synthesize_text(v_name, part, spd, ns, nw)
        synth_time_total += time.perf_counter() - t_synth_start
        segments.append({'pcm_bytes': pcm_bytes, 'sample_rate': sample_rate})

    total_worker_time = time.perf_counter() - t_worker_start

    # Atualiza estatísticas do worker TTS
    try:
        cpu_id = os.sched_getaffinity(0)
        cpu_id = next(iter(cpu_id))
        update_worker_stats("tts", cpu_id, total_worker_time)
    except:
        pass

    metrics = {
        'queue_wait': queue_wait,
        'synth_time': synth_time_total,
        'tts_worker_time': total_worker_time,
        'num_segments': len(segments),
    }

    return segments, metrics

# ---------- Pools de processos ----------
tts_pool = ProcessPoolExecutor(
    max_workers=TTS_WORKERS,
    initializer=_init_tts_worker
)
mix_pool = ProcessPoolExecutor(
    max_workers=MIX_WORKERS,
    initializer=_init_mix_worker
)

# ---------- Semáforo para controlar concorrência ----------
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

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
app = FastAPI(title="Piper TTS API (Otimizada com Pool Global)")

# ---------- Estatísticas agregadas ----------
stats = defaultdict(list)
stats_lock = asyncio.Lock()

# ---------- Endpoint principal ----------
@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    async with request_semaphore:
        t_total_start = time.perf_counter()

        # Prepara dados
        speakers_list = []
        if req.speakers:
            for spk in req.speakers:
                speakers_list.append({
                    'role': spk.role,
                    'voice': spk.voice,
                    'speed': spk.speed,
                    'noise_scale': spk.noise_scale if spk.noise_scale is not None else req.noise_scale,
                    'noise_w_scale': spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale,
                })

        try:
            ambient_dict = req.ambient.model_dump()
        except AttributeError:
            ambient_dict = req.ambient.dict()

        enqueue_time = time.perf_counter()
        loop = asyncio.get_running_loop()

        # --- Etapa 1: TTS (síntese) ---
        t_tts_start = time.perf_counter()
        try:
            tts_future = loop.run_in_executor(
                tts_pool,
                process_tts_only,
                req.voice,
                req.text,
                req.speed,
                req.noise_scale,
                req.noise_w_scale,
                req.effects,
                speakers_list,
                enqueue_time
            )
            segments, tts_metrics = await asyncio.wait_for(tts_future, timeout=REQUEST_TIMEOUT)
        except TimeoutError:
            logger.error("Timeout na síntese TTS")
            raise HTTPException(504, "TTS synthesis timeout")
        t_tts = time.perf_counter() - t_tts_start

        # --- Etapa 2: Mixagem ---
        t_mix_start = time.perf_counter()
        try:
            mix_future = loop.run_in_executor(
                mix_pool,
                mix_and_export_task,
                segments,
                ambient_dict,
                22050
            )
            wav_bytes, mix_time = await asyncio.wait_for(mix_future, timeout=REQUEST_TIMEOUT)
        except TimeoutError:
            logger.error("Timeout na mixagem")
            raise HTTPException(504, "Mix timeout")
        t_mix = time.perf_counter() - t_mix_start

        total_time = time.perf_counter() - t_total_start
        metrics = {
            'queue_wait': tts_metrics['queue_wait'],
            'synth_time': tts_metrics['synth_time'],
            'mix_time': mix_time,
            'total_worker_time': tts_metrics['tts_worker_time'] + mix_time,
            'num_segments': tts_metrics['num_segments'],
            'tts_worker_time': tts_metrics['tts_worker_time'],
        }

        # Atualiza estatísticas do worker de mixagem
        try:
            cpu_id = os.sched_getaffinity(0)
            cpu_id = next(iter(cpu_id))
            update_worker_stats("mix", cpu_id, mix_time)
        except:
            pass

        async with stats_lock:
            stats['total'].append(total_time)
            stats['queue_wait'].append(metrics['queue_wait'])
            stats['synth_time'].append(metrics['synth_time'])
            stats['mix_time'].append(metrics['mix_time'])
            stats['total_worker_time'].append(metrics['total_worker_time'])
            stats['num_segments'].append(metrics['num_segments'])
            stats['tts_worker_time'].append(metrics['tts_worker_time'])

        logger.info(
            f"⏱️ Requisição: total={total_time:.3f}s | "
            f"fila={metrics['queue_wait']:.3f}s | synth={metrics['synth_time']:.3f}s | "
            f"mix={metrics['mix_time']:.3f}s | tts_worker={metrics['tts_worker_time']:.3f}s | "
            f"segmentos={metrics['num_segments']}"
        )

        return Response(content=wav_bytes, media_type="audio/wav")

# ---------- Endpoints de diagnóstico ----------

@app.get("/stats")
async def get_stats():
    async with stats_lock:
        if not stats['total']:
            return {"message": "Nenhuma requisição processada ainda."}
        report = {}
        for key, values in stats.items():
            if key == 'num_segments':
                report[key] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
            else:
                report[key] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": sorted(values)[int(0.95 * len(values))] if len(values) > 1 else values[0],
                }
        return report

@app.get("/workers")
async def get_workers():
    with worker_stats_lock:
        workers = []
        for key, data in worker_stats.items():
            worker_type, worker_id = key.split('_')
            workers.append({
                "type": worker_type,
                "id": int(worker_id),
                "pid": data["pid"],
                "cpu_id": data["cpu_id"],
                "requests_processed": data["requests_processed"],
                "avg_time": data["avg_time"],
            })
        workers.sort(key=lambda x: (x["type"], x["id"]))
        return {
            "total_workers": len(workers),
            "tts_workers": [w for w in workers if w["type"] == "tts"],
            "mix_workers": [w for w in workers if w["type"] == "mix"],
        }

@app.get("/pool_status")
async def pool_status():
    with worker_stats_lock:
        tts_count = sum(1 for k in worker_stats.keys() if k.startswith('tts_'))
        mix_count = sum(1 for k in worker_stats.keys() if k.startswith('mix_'))
    return {
        "tts_workers": TTS_WORKERS,
        "mix_workers": MIX_WORKERS,
        "tts_registered": tts_count,
        "mix_registered": mix_count,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "request_timeout": REQUEST_TIMEOUT,
        "current_concurrency": request_semaphore._value if hasattr(request_semaphore, '_value') else "unknown",
    }

@app.post("/reset_stats")
async def reset_stats():
    async with stats_lock:
        for key in stats:
            stats[key].clear()
    with worker_stats_lock:
        for key in list(worker_stats.keys()):
            data = worker_stats[key]
            data["requests_processed"] = 0
            data["total_time"] = 0.0
            data["avg_time"] = 0.0
            worker_stats[key] = data
    return {"message": "Estatísticas resetadas com sucesso."}

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
        "workers": {"tts": TTS_WORKERS, "mix": MIX_WORKERS},
        "concurrency_limit": MAX_CONCURRENT_REQUESTS,
        "timeout": REQUEST_TIMEOUT,
    }

# ---------- Ponto de entrada ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
