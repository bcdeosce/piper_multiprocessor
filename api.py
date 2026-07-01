import os
import sys
import re
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

# ---------- CRÍTICO: Define variáveis de ambiente ANTES de importar onnxruntime ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_NUM_THREADS"] = "1"

# ---------- MONKEY PATCH DO ONNX RUNTIME ----------
import onnxruntime as ort
_original_ort_session = ort.InferenceSession

def _patched_ort_session(model_path, sess_options=None, providers=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return _original_ort_session(model_path, sess_options, providers=providers, **kwargs)

ort.InferenceSession = _patched_ort_session

# ---------- Instalação automática de dependências ----------
try:
    import psutil
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

try:
    import aiohttp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-tts"])
    from piper import PiperVoice, SynthesisConfig

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

# ---------- Configuração de logs com buffer em memória ----------
class MemoryHandler(logging.Handler):
    def __init__(self, capacity=100):
        super().__init__()
        self.capacity = capacity
        self.buffer = []

    def emit(self, record):
        self.buffer.append(self.format(record))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

memory_handler = MemoryHandler(capacity=100)
memory_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        memory_handler
    ]
)
logger = logging.getLogger("piper-api")
logger.setLevel(logging.INFO)
memory_handler.setLevel(logging.WARNING)

ort.set_default_logger_severity(3)

BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
BENCH_DIR = BASE_DIR / "bench"

VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)
BENCH_DIR.mkdir(exist_ok=True)

# ---------- Detecção de núcleos físicos ----------
def get_physical_cores_from_proc() -> List[int]:
    cores = {}
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu = None
            core = None
            for line in f:
                if line.startswith('processor'):
                    cpu = int(line.split(':')[1].strip())
                elif line.startswith('core id'):
                    core = int(line.split(':')[1].strip())
                    if core is not None and cpu is not None:
                        cores.setdefault(core, []).append(cpu)
        if cores:
            return sorted([min(cpus) for cpus in cores.values()])
    except Exception as e:
        logger.warning(f"Erro ao ler /proc/cpuinfo: {e}")
    return []

def get_physical_cores_from_sys() -> List[int]:
    physical = []
    try:
        for cpu in range(os.cpu_count()):
            path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            try:
                with open(path, 'r') as f:
                    siblings = f.read().strip().split(',')
                    if int(siblings[0]) == cpu:
                        physical.append(cpu)
            except FileNotFoundError:
                continue
        return sorted(set(physical))
    except Exception as e:
        logger.warning(f"Erro ao ler sys: {e}")
    return []

def get_physical_cores() -> List[int]:
    env_cores = os.getenv("PHYSICAL_CORES")
    if env_cores:
        try:
            cores = [int(c.strip()) for c in env_cores.split(',') if c.strip()]
            logger.info(f"Usando PHYSICAL_CORES do env: {cores}")
            return cores
        except:
            pass
    cores = get_physical_cores_from_proc()
    if cores:
        logger.info(f"Detectados via /proc/cpuinfo: {cores}")
        return cores
    cores = get_physical_cores_from_sys()
    if cores:
        logger.info(f"Detectados via sys: {cores}")
        return cores
    logger.warning("Não foi possível detectar núcleos físicos, usando todos")
    return list(range(os.cpu_count()))

ALL_CORES = list(range(os.cpu_count()))
PHYSICAL_CORES = get_physical_cores()
HYPER_THREAD_CORES = [c for c in ALL_CORES if c not in PHYSICAL_CORES]

logger.info(f"Núcleos totais: {ALL_CORES}")
logger.info(f"Núcleos físicos: {PHYSICAL_CORES}")
logger.info(f"Núcleos Hyper-Thread: {HYPER_THREAD_CORES}")

# ---------- Contador e locks ----------
_cpu_counter = mp.Value('i', 0)
_cpu_lock = mp.Lock()

# ---------- Workers ----------
TTS_WORKERS = int(os.getenv("TTS_WORKERS", 14))
MIX_WORKERS = int(os.getenv("MIX_WORKERS", 6))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 1000))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))

if TTS_WORKERS > len(PHYSICAL_CORES):
    logger.warning(
        f"TTS_WORKERS ({TTS_WORKERS}) excede núcleos físicos ({len(PHYSICAL_CORES)}). "
        f"Limitando para {len(PHYSICAL_CORES)}."
    )
    TTS_WORKERS = len(PHYSICAL_CORES)

logger.info(f"Workers: TTS={TTS_WORKERS} (físicos), Mix={MIX_WORKERS}")

# ---------- GERENCIADOR COMPARTILHADO ----------
manager = mp.Manager()

# Estatísticas dos workers
worker_stats = manager.dict()
worker_stats_lock = mp.Lock()

# Estatísticas agregadas para /bench e /stats
bench_stats = {
    "total": manager.list(),
    "tts_wall": manager.list(),
    "mix": manager.list(),
    "queue_wait": manager.list(),
    # Métricas detalhadas da mixagem
    "mix_temp_files_creation": manager.list(),
    "mix_concat_time": manager.list(),
    "mix_ambient_loop_time": manager.list(),
    "mix_ambient_trim_time": manager.list(),
    "mix_amix_time": manager.list(),
    "mix_cleanup_time": manager.list(),
    "mix_total_mix_time": manager.list(),
    "num_segments": manager.list(),
    "synth_time": manager.list(),
}
bench_stats_lock = mp.Lock()

def register_worker(worker_type, worker_id, pid, cpu_id, is_physical):
    with worker_stats_lock:
        key = f"{worker_type}_{worker_id}"
        worker_stats[key] = {
            "pid": pid,
            "cpu_id": cpu_id,
            "is_physical": is_physical,
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

def compute_stats(values: list) -> dict:
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "p95": 0, "count": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "count": len(values),
    }

# ---------- Inicializador TTS ----------
def _init_tts_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ORT_NUM_THREADS"] = "1"
    ort.set_default_logger_severity(3)

    with _cpu_lock:
        idx = _cpu_counter.value
        _cpu_counter.value += 1
        cpu_id = PHYSICAL_CORES[idx % len(PHYSICAL_CORES)]

    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"TTS Worker fixado ao núcleo físico {cpu_id}")
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade no TTS: {e}")

    worker_id = cpu_id
    pid = os.getpid()
    register_worker("tts", worker_id, pid, cpu_id, is_physical=True)

    mod = sys.modules['__main__']
    mod._worker_cpu_id = cpu_id
    mod._worker_voice_cache = {}

    logger.info(f"Worker {pid} iniciando pré-carregamento de {len(VOICE_PATHS)} vozes...")
    load_start = time.perf_counter()
    for voice_name in VOICE_PATHS.keys():
        try:
            pool = get_voice_pool(voice_name)
        except Exception as e:
            logger.error(f"  Falha ao carregar voz '{voice_name}': {e}")
    load_time = time.perf_counter() - load_start
    logger.info(f"Worker {pid} pré-carregou {len(VOICE_PATHS)} vozes em {load_time:.2f}s")
    logger.info(f"TTS Worker {worker_id} (PID {pid}) registrado e pronto")

# ---------- Inicializador Mix ----------
def _init_mix_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    ort.set_default_logger_severity(3)

    with _cpu_lock:
        idx = _cpu_counter.value
        _cpu_counter.value += 1

        used_physical = set()
        for data in worker_stats.values():
            if data.get("is_physical", False):
                used_physical.add(data["cpu_id"])
        physical_available = [c for c in PHYSICAL_CORES if c not in used_physical]

        if physical_available:
            cpu_id = physical_available[0]
            is_physical = True
        else:
            used_all = set(data["cpu_id"] for data in worker_stats.values())
            ht_available = [c for c in HYPER_THREAD_CORES if c not in used_all]
            if ht_available:
                cpu_id = ht_available[0]
                is_physical = False
            else:
                cpu_id = idx % len(ALL_CORES)
                is_physical = cpu_id in PHYSICAL_CORES

    try:
        os.sched_setaffinity(0, {cpu_id})
        logger.info(f"Mix Worker fixado ao núcleo {cpu_id} ({'físico' if is_physical else 'Hyper-Thread'})")
    except Exception as e:
        logger.warning(f"Falha ao definir afinidade na mixagem: {e}")

    worker_id = cpu_id
    pid = os.getpid()
    register_worker("mix", worker_id, pid, cpu_id, is_physical)

# ---------- VoicePool ----------
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
        logger.info(f"Worker {os.getpid()} carregou voz {voice_name}")
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

# ---------- MIXAGEM COM MÉTRICAS DETALHADAS ----------
def mix_and_export_task(segments_data, ambient_cfg, target_rate=22050):
    metrics = {}
    t0_total = time.perf_counter()
    
    logger.info("=" * 60)
    logger.info("🔊 INÍCIO DA MIXAGEM/CONCATENAÇÃO")
    logger.info(f"📦 Recebidos {len(segments_data)} segmentos")

    logger.info("📋 Ordem dos segmentos:")
    for i, data in enumerate(segments_data):
        if 'pcm_bytes' in data:
            size_kb = len(data['pcm_bytes']) / 1024
            logger.info(f"  [{i}] VOZ   | sample_rate={data['sample_rate']} | tamanho={size_kb:.1f}KB")
        elif 'effect' in data:
            logger.info(f"  [{i}] EFEITO | arquivo='{data['effect']}' | voz='{data['voice']}'")
        else:
            logger.info(f"  [{i}] DESCONHECIDO | {data}")

    temp_files = []
    try:
        # --- 1. CRIAÇÃO DOS ARQUIVOS TEMPORÁRIOS ---
        t0 = time.perf_counter()
        logger.info("📁 Criando arquivos temporários...")
        for idx, data in enumerate(segments_data):
            if 'pcm_bytes' in data:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    wav_path = f.name
                with wave.open(wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(data['sample_rate'])
                    wav_file.writeframes(data['pcm_bytes'])
                temp_files.append(wav_path)
                logger.info(f"  [{idx}] WAV criado: {wav_path} (PCM, {data['sample_rate']}Hz, {len(data['pcm_bytes'])/1024:.1f}KB)")

            elif 'effect' in data:
                voice_dir = VOICES_DIR / data['voice']
                effect_path = voice_dir / data['effect']
                if not effect_path.exists():
                    effect_path = EFFECTS_DIR / data['effect']
                if not effect_path.exists():
                    logger.error(f"  [{idx}] Efeito '{data['effect']}' não encontrado")
                    continue
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    with open(effect_path, 'rb') as src:
                        f.write(src.read())
                    temp_files.append(f.name)
                logger.info(f"  [{idx}] WAV copiado: {f.name} (efeito '{data['effect']}', {effect_path.stat().st_size/1024:.1f}KB)")
            else:
                logger.warning(f"  [{idx}] Ignorado: dados desconhecidos")
        metrics['temp_files_creation'] = time.perf_counter() - t0

        # --- ADICIONAR AMBIENTE ---
        ambient_volume_db = ambient_cfg.get('volume_db', -15.0)
        ambient_added = False
        ambient_original_duration = 0.0

        if ambient_cfg.get('enabled') and ambient_cfg.get('file'):
            ambient_path = AMBIENT_DIR / f"{ambient_cfg['file']}.wav"
            if ambient_path.exists():
                with wave.open(str(ambient_path), 'rb') as wf:
                    ambient_frames = wf.getnframes()
                    ambient_rate = wf.getframerate()
                    ambient_original_duration = ambient_frames / ambient_rate
                logger.info(f"🌧️ Ambiente '{ambient_cfg['file']}.wav' duração original: {ambient_original_duration:.2f}s, volume: {ambient_volume_db}dB")

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    with open(ambient_path, 'rb') as src:
                        f.write(src.read())
                    temp_files.append(f.name)
                    ambient_added = True
                logger.info(f"🌧️ Ambiente original copiado: {f.name}")
            else:
                logger.warning(f"🌧️ Arquivo de ambiente '{ambient_cfg['file']}.wav' não encontrado")

        if not temp_files:
            raise ValueError("Nenhum arquivo para processar")

        # --- CONCATENAÇÃO DOS SEGMENTOS (voz + efeitos) ---
        if ambient_added:
            ambient_file = temp_files[-1]
            voice_files = temp_files[:-1]
        else:
            voice_files = temp_files
            ambient_file = None

        logger.info(f"📊 Total de arquivos de voz/efeitos: {len(voice_files)}")

        # Concatena voz + efeitos
        filter_concat = f"concat=n={len(voice_files)}:v=0:a=1"
        concat_cmd = ["ffmpeg", "-y"]
        for f in voice_files:
            concat_cmd.extend(["-i", f])
        concat_cmd.extend([
            "-filter_complex", filter_concat,
            "-ar", str(target_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            "pipe:1"
        ])

        logger.info(f"🔗 Comando de concatenação (voz + efeitos):")
        logger.info(f"  {' '.join(concat_cmd)}")

        t0 = time.perf_counter()
        logger.info("⏳ Executando concatenação dos segmentos...")
        result_concat = subprocess.run(concat_cmd, capture_output=True, check=True, timeout=10)
        metrics['concat_time'] = time.perf_counter() - t0
        main_audio_bytes = result_concat.stdout
        main_duration = len(main_audio_bytes) / (target_rate * 2)
        logger.info(f"✅ Concatenação concluída em {metrics['concat_time']:.3f}s | duração={main_duration:.2f}s | tamanho={len(main_audio_bytes)/1024:.1f}KB")

        # --- SE HOUVER AMBIENTE, FAZ O LOOP E MIXA ---
        if ambient_added and ambient_file:
            if ambient_original_duration > 0:
                repeat_times = int(main_duration // ambient_original_duration) + 1
            else:
                repeat_times = 1

            logger.info(f"🔄 Ambiente original: {ambient_original_duration:.2f}s, cobrir {main_duration:.2f}s, repetir {repeat_times}x")

            loop_cmd = ["ffmpeg", "-y", "-stream_loop", str(repeat_times), "-i", ambient_file,
                       "-c", "copy", "-f", "wav", "pipe:1"]
            logger.info(f"🔄 Loop ambiente: {' '.join(loop_cmd)}")
            t0 = time.perf_counter()
            result_loop = subprocess.run(loop_cmd, capture_output=True, check=True, timeout=10)
            metrics['ambient_loop_time'] = time.perf_counter() - t0
            looped_ambient_bytes = result_loop.stdout
            looped_duration = len(looped_ambient_bytes) / (target_rate * 2)
            logger.info(f"✅ Loop em {metrics['ambient_loop_time']:.3f}s | duração={looped_duration:.2f}s")

            if looped_duration > main_duration:
                logger.info(f"✂️ Cortando ambiente de {looped_duration:.2f}s para {main_duration:.2f}s")
                trim_cmd = [
                    "ffmpeg", "-y",
                    "-i", "pipe:0",
                    "-filter_complex", f"atrim=0:{main_duration}",
                    "-ar", str(target_rate),
                    "-ac", "1",
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    "pipe:1"
                ]
                t0 = time.perf_counter()
                result_trim = subprocess.run(trim_cmd, input=looped_ambient_bytes, capture_output=True, check=True, timeout=10)
                metrics['ambient_trim_time'] = time.perf_counter() - t0
                looped_ambient_bytes = result_trim.stdout
                looped_duration = len(looped_ambient_bytes) / (target_rate * 2)
                logger.info(f"✅ Corte em {metrics['ambient_trim_time']:.3f}s | agora {looped_duration:.2f}s")
            else:
                metrics['ambient_trim_time'] = 0.0

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_amb_loop:
                f_amb_loop.write(looped_ambient_bytes)
                looped_ambient_file = f_amb_loop.name

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_main:
                f_main.write(main_audio_bytes)
                main_file = f_main.name

            volume_filter = f"volume={ambient_volume_db}dB"
            mix_cmd = [
                "ffmpeg", "-y",
                "-i", main_file,
                "-i", looped_ambient_file,
                "-filter_complex", f"[1]{volume_filter}[amb];[0][amb]amix=inputs=2:duration=shortest:normalize=0",
                "-ar", str(target_rate),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                "-f", "wav",
                "pipe:1"
            ]

            logger.info(f"🔀 Mixagem: {' '.join(mix_cmd)}")
            t0 = time.perf_counter()
            result_mix = subprocess.run(mix_cmd, capture_output=True, check=True, timeout=10)
            metrics['amix_time'] = time.perf_counter() - t0
            wav_bytes = result_mix.stdout
            logger.info(f"✅ Mixagem em {metrics['amix_time']:.3f}s | tamanho={len(wav_bytes)/1024:.1f}KB")

            t0 = time.perf_counter()
            for f in [main_file, looped_ambient_file]:
                try: os.unlink(f)
                except: pass
            metrics['cleanup_time'] = time.perf_counter() - t0

        else:
            wav_bytes = main_audio_bytes
            metrics['ambient_loop_time'] = 0.0
            metrics['ambient_trim_time'] = 0.0
            metrics['amix_time'] = 0.0
            metrics['cleanup_time'] = 0.0

        metrics['total_mix_time'] = time.perf_counter() - t0_total

        # Atualiza estatísticas do worker de mixagem
        try:
            cpu_id = os.sched_getaffinity(0)
            cpu_id = next(iter(cpu_id))
            update_worker_stats("mix", cpu_id, metrics['total_mix_time'])
        except Exception as e:
            logger.warning(f"Mix stats: {e}")

        logger.info(f"🎯 MIXAGEM FINALIZADA em {metrics['total_mix_time']:.3f}s | tamanho final {len(wav_bytes)/1024:.1f}KB")
        logger.info("=" * 60)

        return wav_bytes, metrics

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg erro: {e.stderr.decode()}")
        logger.error(f"   Comando: {' '.join(e.cmd)}")
        raise RuntimeError("Falha na mixagem com FFmpeg")
    finally:
        for f in temp_files:
            try: os.unlink(f)
            except: pass

# ---------- Processamento TTS ----------
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
    t_worker_start = time.perf_counter()
    queue_wait = t_worker_start - enqueue_time

    is_dialog = bool(speakers)
    if not is_dialog:
        if not voice_name:
            raise ValueError("voice_name obrigatório no modo simples")
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

    logger.info(f"📋 Partes: {parts}")

    segments = []
    synth_time_total = 0.0

    for part in parts:
        logger.debug(f"🔍 '{part}'")

        # 1. EFEITO (ANTES DE TAG)
        if part in effects:
            effect_file = effects[part]
            voice_for_eff = speaker_map[current_role][0] if is_dialog and current_role else voice_name
            segments.append({'effect': effect_file, 'voice': voice_for_eff})
            logger.info(f"🎬 Efeito: '{part}' -> '{effect_file}'")
            continue

        # 2. TAG DE SPEAKER
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
                logger.info(f"🔄 Speaker: {current_role}")
            continue

        # 3. SÍNTESE
        if is_dialog:
            if current_role is None:
                raise ValueError("Nenhum speaker definido antes do texto. Use [papel] no início.")
            v_name, spd, ns, nw = speaker_map[current_role]
        else:
            v_name = voice_name
            spd = speed
            ns = noise_scale
            nw = noise_w_scale

        logger.info(f"🗣️ Sintetizando (voz={v_name}, speed={spd}): '{part[:50]}...'")
        t_synth_start = time.perf_counter()
        sample_rate, pcm_bytes = synthesize_text(v_name, part, spd, ns, nw)
        synth_time_total += time.perf_counter() - t_synth_start
        segments.append({'pcm_bytes': pcm_bytes, 'sample_rate': sample_rate})
        logger.info(f"✅ Áudio: sample_rate={sample_rate}, tamanho={len(pcm_bytes)/1024:.1f}KB")

    total_worker_time = time.perf_counter() - t_worker_start

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

# ---------- WARM-UP ----------
def warmup_workers(tts_pool):
    logger.info("🔥 Iniciando warm-up dos workers TTS...")
    warmup_start = time.perf_counter()

    dummy_voice = list(VOICE_PATHS.keys())[0] if VOICE_PATHS else None
    if not dummy_voice:
        logger.warning("Nenhuma voz disponível para warm-up.")
        return

    futures = []
    for _ in range(TTS_WORKERS):
        future = tts_pool.submit(
            process_tts_only,
            dummy_voice,
            "Warm-up",
            1.0,
            0.667,
            0.8,
            {},
            [],
            time.perf_counter()
        )
        futures.append(future)

    for future in futures:
        try:
            future.result(timeout=60)
        except Exception as e:
            logger.warning(f"Warm-up falhou: {e}")

    warmup_time = time.perf_counter() - warmup_start
    logger.info(f"✅ Warm-up concluído em {warmup_time:.2f}s")

# ---------- Pools ----------
tts_pool = ProcessPoolExecutor(
    max_workers=TTS_WORKERS,
    initializer=_init_tts_worker
)
mix_pool = ProcessPoolExecutor(
    max_workers=MIX_WORKERS,
    initializer=_init_mix_worker
)

request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) if MAX_CONCURRENT_REQUESTS > 0 else None

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
app = FastAPI(title="Piper TTS API (CPU)")

warmup_complete = False
warmup_lock = asyncio.Lock()

# ---------- Evento de startup ----------
@app.on_event("startup")
async def startup():
    global warmup_complete
    logger.info("🚀 Iniciando servidor...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, warmup_workers, tts_pool)
    async with warmup_lock:
        warmup_complete = True
    logger.info("✅ Servidor pronto.")

# ---------- Endpoint principal ----------
@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    async with warmup_lock:
        if not warmup_complete:
            raise HTTPException(503, "Carregando modelos. Tente novamente.")

    if request_semaphore:
        await request_semaphore.acquire()
    try:
        t_total_start = time.perf_counter()

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

        mix_future = loop.run_in_executor(
            mix_pool,
            mix_and_export_task,
            segments,
            ambient_dict,
            22050
        )
        wav_bytes, mix_metrics = await asyncio.wait_for(mix_future, timeout=REQUEST_TIMEOUT)

        total_time = time.perf_counter() - t_total_start

        # Atualiza estatísticas para /bench e /stats
        with bench_stats_lock:
            bench_stats["total"].append(total_time)
            bench_stats["tts_wall"].append(tts_metrics['tts_worker_time'])
            bench_stats["mix"].append(mix_metrics['total_mix_time'])
            bench_stats["queue_wait"].append(tts_metrics['queue_wait'])
            bench_stats["synth_time"].append(tts_metrics['synth_time'])
            bench_stats["num_segments"].append(tts_metrics['num_segments'])
            bench_stats["mix_temp_files_creation"].append(mix_metrics['temp_files_creation'])
            bench_stats["mix_concat_time"].append(mix_metrics['concat_time'])
            bench_stats["mix_ambient_loop_time"].append(mix_metrics['ambient_loop_time'])
            bench_stats["mix_ambient_trim_time"].append(mix_metrics['ambient_trim_time'])
            bench_stats["mix_amix_time"].append(mix_metrics['amix_time'])
            bench_stats["mix_cleanup_time"].append(mix_metrics['cleanup_time'])
            bench_stats["mix_total_mix_time"].append(mix_metrics['total_mix_time'])

        logger.info(
            f"⏱️ Requisição: total={total_time:.3f}s | "
            f"fila={tts_metrics['queue_wait']:.3f}s | synth={tts_metrics['synth_time']:.3f}s | "
            f"mix={mix_metrics['total_mix_time']:.3f}s | segmentos={tts_metrics['num_segments']}"
        )

        return Response(content=wav_bytes, media_type="audio/wav")
    finally:
        if request_semaphore:
            request_semaphore.release()

# ================= ENDPOINTS DE DIAGNÓSTICO =================

# 1. STATS (com todas as métricas detalhadas)
@app.get("/stats")
async def get_stats():
    with bench_stats_lock:
        if not bench_stats["total"]:
            return Response(
                content=json.dumps({"message": "Nenhuma requisição ainda."}, indent=2),
                media_type="application/json"
            )
        report = {
            "total": compute_stats(list(bench_stats["total"])),
            "tts_wall": compute_stats(list(bench_stats["tts_wall"])),
            "mix": compute_stats(list(bench_stats["mix"])),
            "queue_wait": compute_stats(list(bench_stats["queue_wait"])),
            "synth_time": compute_stats(list(bench_stats["synth_time"])),
            "num_segments": compute_stats(list(bench_stats["num_segments"])),
            "mix_temp_files_creation": compute_stats(list(bench_stats["mix_temp_files_creation"])),
            "mix_concat_time": compute_stats(list(bench_stats["mix_concat_time"])),
            "mix_ambient_loop_time": compute_stats(list(bench_stats["mix_ambient_loop_time"])),
            "mix_ambient_trim_time": compute_stats(list(bench_stats["mix_ambient_trim_time"])),
            "mix_amix_time": compute_stats(list(bench_stats["mix_amix_time"])),
            "mix_cleanup_time": compute_stats(list(bench_stats["mix_cleanup_time"])),
            "mix_total_mix_time": compute_stats(list(bench_stats["mix_total_mix_time"])),
        }
    return Response(
        content=json.dumps(report, indent=2),
        media_type="application/json"
    )

# 2. BENCH
@app.get("/bench")
async def get_bench():
    with bench_stats_lock:
        total = compute_stats(list(bench_stats["total"]))
        tts = compute_stats(list(bench_stats["tts_wall"]))
        mix = compute_stats(list(bench_stats["mix"]))
        queue = compute_stats(list(bench_stats["queue_wait"]))
    return Response(
        content=json.dumps({
            "benchmark_results": {
                "total": total,
                "tts_wall": tts,
                "mix": mix,
                "queue_wait": queue
            },
            "configuration": {
                "voices": list(VOICE_PATHS.keys()),
                "tts_workers": TTS_WORKERS,
                "mix_workers": MIX_WORKERS,
                "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
                "request_timeout": REQUEST_TIMEOUT,
                "physical_cores": PHYSICAL_CORES,
                "hyper_thread_cores": HYPER_THREAD_CORES,
            }
        }, indent=2),
        media_type="application/json"
    )

# 3. WORKERS
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
                "is_physical": data.get("is_physical", True),
                "requests_processed": data["requests_processed"],
                "avg_time": data["avg_time"],
            })
        workers.sort(key=lambda x: (x["type"], x["id"]))
    return Response(
        content=json.dumps({
            "tts_workers": TTS_WORKERS,
            "mix_workers": MIX_WORKERS,
            "active_tts_workers": len([w for w in workers if w["type"] == "tts"]),
            "active_mix_workers": len([w for w in workers if w["type"] == "mix"]),
            "per_worker": workers,
            "physical_cores": PHYSICAL_CORES,
            "hyper_thread_cores": HYPER_THREAD_CORES,
        }, indent=2),
        media_type="application/json"
    )

# 4. POOL_STATUS (RESTAURADO)
@app.get("/pool_status")
async def pool_status():
    with worker_stats_lock:
        tts_count = sum(1 for k in worker_stats.keys() if k.startswith('tts_'))
        mix_count = sum(1 for k in worker_stats.keys() if k.startswith('mix_'))
    with bench_stats_lock:
        total_requests = len(bench_stats["total"])
    return Response(
        content=json.dumps({
            "tts_workers": TTS_WORKERS,
            "mix_workers": MIX_WORKERS,
            "tts_registered": tts_count,
            "mix_registered": mix_count,
            "physical_cores": PHYSICAL_CORES,
            "hyper_thread_cores": HYPER_THREAD_CORES,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "request_timeout": REQUEST_TIMEOUT,
            "current_concurrency": request_semaphore._value if request_semaphore and hasattr(request_semaphore, '_value') else "disabled",
            "total_requests_processed": total_requests,
        }, indent=2),
        media_type="application/json"
    )

# 5. DIAGNOSE_CORES (RESTAURADO)
@app.get("/diagnose_cores")
async def diagnose_cores():
    try:
        cpuinfo = []
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
        current = {}
        for line in lines:
            if line.strip() == '':
                if current:
                    cpuinfo.append(current)
                    current = {}
                continue
            key, val = line.split(':', 1)
            current[key.strip()] = val.strip()
        if current:
            cpuinfo.append(current)

        cores_map = {}
        for cpu in cpuinfo:
            if 'processor' in cpu and 'core id' in cpu:
                proc = int(cpu['processor'])
                core = int(cpu['core id'])
                cores_map.setdefault(core, []).append(proc)

        siblings = {}
        for cpu in ALL_CORES:
            path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            try:
                with open(path, 'r') as f:
                    siblings[cpu] = f.read().strip()
            except:
                siblings[cpu] = str(cpu)

        workers_info = []
        for key, data in worker_stats.items():
            worker_type, worker_id = key.split('_')
            pid = data["pid"]
            cpu_id = data["cpu_id"]
            is_physical = data.get("is_physical", False)

            try:
                proc = psutil.Process(pid)
                affinity = proc.cpu_affinity()
                num_threads = len(proc.threads())
                has_multiple_threads = num_threads > 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                affinity = "N/A"
                num_threads = "N/A"
                has_multiple_threads = "N/A"

            workers_info.append({
                "type": worker_type,
                "id": int(worker_id),
                "pid": pid,
                "assigned_cpu": cpu_id,
                "is_physical": is_physical,
                "real_affinity": affinity,
                "num_threads": num_threads,
                "has_multiple_threads": has_multiple_threads,
                "requests_processed": data["requests_processed"],
                "avg_time": data["avg_time"],
            })

        physical_from_proc = sorted([min(cpus) for cpus in cores_map.values()]) if cores_map else []
        physical_from_sys = []
        for cpu in ALL_CORES:
            path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            try:
                with open(path, 'r') as f:
                    sibs = f.read().strip().split(',')
                    if int(sibs[0]) == cpu:
                        physical_from_sys.append(cpu)
            except:
                pass
        physical_from_sys = sorted(set(physical_from_sys))

        return Response(
            content=json.dumps({
                "total_cores": os.cpu_count(),
                "cores_map": cores_map,
                "siblings": siblings,
                "physical_cores_from_proc": physical_from_proc,
                "physical_cores_from_sys": physical_from_sys,
                "current_physical_cores": PHYSICAL_CORES,
                "current_hyper_thread_cores": HYPER_THREAD_CORES,
                "workers": workers_info,
                "workers_with_multiple_threads": [w for w in workers_info if w.get("has_multiple_threads") is True],
                "analysis": {
                    "threading_issue": any(w.get("has_multiple_threads") is True for w in workers_info),
                    "affinity_mismatch": any(
                        w.get("real_affinity") != "N/A" and w.get("assigned_cpu") not in w.get("real_affinity", [])
                        for w in workers_info
                    )
                }
            }, indent=2),
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Erro no diagnose: {e}")
        return Response(
            content=json.dumps({"error": str(e)}, indent=2),
            status_code=500,
            media_type="application/json"
        )

# 6. RESOURCES
@app.get("/resources")
async def get_resources():
    cpu_util = -1.0
    try:
        cpu_util = psutil.cpu_percent(interval=0.1)
    except:
        pass
    return Response(
        content=json.dumps({
            "cpu_utilization_percent": cpu_util,
            "cpu_cores_available": os.cpu_count(),
            "physical_cores": PHYSICAL_CORES,
            "hyper_thread_cores": HYPER_THREAD_CORES,
            "tts_workers": TTS_WORKERS,
            "mix_workers": MIX_WORKERS,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "request_timeout": REQUEST_TIMEOUT,
        }, indent=2),
        media_type="application/json"
    )

# 7. RESET_STATS
@app.post("/reset_stats")
async def reset_stats():
    with bench_stats_lock:
        for key in bench_stats:
            bench_stats[key] = manager.list()
    with worker_stats_lock:
        for key in list(worker_stats.keys()):
            data = worker_stats[key]
            data["requests_processed"] = 0
            data["total_time"] = 0.0
            data["avg_time"] = 0.0
            worker_stats[key] = data
    return Response(
        content=json.dumps({"message": "Estatísticas resetadas."}, indent=2),
        media_type="application/json"
    )

# 8. LOGS
@app.get("/logs")
async def get_logs():
    return Response(
        content=json.dumps({"logs": memory_handler.buffer}, indent=2),
        media_type="application/json"
    )

# 9. CARGA
@app.get("/carga")
async def get_carga():
    if not BENCH_DIR.exists():
        return Response(
            content=json.dumps({"message": "Nenhum teste de carga ainda."}, indent=2),
            media_type="application/json"
        )
    files = sorted(BENCH_DIR.glob("carga_results_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return Response(
            content=json.dumps({"message": "Nenhum arquivo de carga."}, indent=2),
            media_type="application/json"
        )
    with open(files[0], "r") as f:
        data = json.load(f)
    return Response(
        content=json.dumps({
            "file": files[0].name,
            "data": data,
            "available_files": [f.name for f in files]
        }, indent=2),
        media_type="application/json"
    )

# 10. CARGA_FILES
@app.get("/carga_files")
async def list_carga_files():
    if not BENCH_DIR.exists():
        return Response(content=json.dumps({"files": []}, indent=2), media_type="application/json")
    files = sorted(BENCH_DIR.glob("carga_results_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return Response(
        content=json.dumps({"files": [f.name for f in files]}, indent=2),
        media_type="application/json"
    )

# 11. CARGA/{FILE_NAME}
@app.get("/carga/{file_name}")
async def get_carga_file(file_name: str):
    file_path = BENCH_DIR / file_name
    if not file_path.exists():
        raise HTTPException(404, "Arquivo não encontrado")
    with open(file_path, "r") as f:
        data = json.load(f)
    return Response(content=json.dumps(data, indent=2), media_type="application/json")

# 12. RUN_LOAD_TEST (com suporte a speakers e instalação automática do aiohttp)
@app.post("/run_load_test")
async def run_load_test(
    ramp_max: int = 51,
    step: int = 5,
    duration: int = 30,
    timeout: int = 30,
    voice: Optional[str] = None,
    speakers: Optional[List[Dict[str, Any]]] = None,
    ambient_file: str = "ubs",
    ambient_volume: float = -5.0,
    effects: Optional[Dict[str, str]] = None,
    dialogs: Optional[List[str]] = None,
):
    """
    Executa teste de carga local.
    
    Parâmetros:
    - voice: string (modo simples, uma voz)
    - speakers: lista de dicts (modo diálogo) ex: [{"role":"paciente","voice":"crianca","speed":1.0}, {"role":"acompanhante","voice":"faber","speed":0.95}]
    - Se ambos forem fornecidos, speakers tem prioridade.
    """
    import aiohttp, statistics, random
    from contextlib import asynccontextmanager

    use_speakers = speakers is not None and len(speakers) > 0

    if not use_speakers and voice is None:
        voice = list(VOICE_PATHS.keys())[0] if VOICE_PATHS else "crianca"

    if effects is None:
        effects = {
            "[tosse]": "tosse.wav",
            "[suspiro]": "suspiro.wav",
            "[inspiracao]": "inspiracao.wav"
        }

    if dialogs is None:
        dialogs = [
            "[paciente] Estou com dor de cabeça forte. [tosse] Ele está assim há três dias, doutor.",
            "[paciente] Tenho tido muita tosse, [tosse]... [tosse].. [acompanhante] E febre desde ontem.",
            "[paciente] Sinto falta de ar [inspiracao] ao caminhar. [acompanhante] Ele já tem histórico de asma.",
            "[paciente] Estou muito cansada [inspiracao] e com falta de ar. [acompanhante] Ela parou de fumar há um mês.",
            "[paciente] A febre começou ontem à noite, depois que ele caiu. [suspiro] Dói muito aqui!",
            "[paciente] O remédio não está fazendo efeito. [acompanhante] Ele está tomando dipirona, mas não melhora.",
            "[paciente] Precisamos de uma receita para antibiótico. [suspiro] Só dar o cu não está ajudando.",
            "[paciente] Meu peito dói quando respiro fundo [inspiracao]. E ele chupou um pau de 25 centímetros?",
            "[paciente] Quando posso voltar ao trabalho? [acompanhante] Precisa de atestado por mais três dias.",
            "[paciente] Ele está com os exames alterados. [suspiro] Vou precisar de cirurgia?"
        ]

    BENCH_DIR.mkdir(exist_ok=True)
    fname = BENCH_DIR / f"carga_results_local_{int(time.time())}.json"
    results = []

    logger.info(f"🚀 Iniciando teste de carga: ramp_max={ramp_max}, step={step}, duration={duration}s")
    if use_speakers:
        logger.info(f"   Modo diálogo com speakers: {speakers}")
    else:
        logger.info(f"   Modo simples com voice: {voice}")

    async with aiohttp.ClientSession() as sess:
        for concurrency in range(1, ramp_max + 1, step):
            sem = asyncio.Semaphore(concurrency)
            start = time.perf_counter()
            succ = 0
            fail = 0
            lats = []
            total_requests = 0

            async def worker():
                nonlocal succ, fail, lats, total_requests
                while time.perf_counter() - start < duration:
                    async with sem:
                        d = random.choice(dialogs)
                        payload = {
                            "text": d,
                            "effects": effects,
                            "ambient": {"enabled": True, "file": ambient_file, "volume_db": ambient_volume}
                        }
                        if use_speakers:
                            payload["speakers"] = speakers
                        else:
                            payload["voice"] = voice

                        t0 = time.perf_counter()
                        try:
                            async with sess.post("http://localhost:8000/synthesize", json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                                if resp.status == 200:
                                    succ += 1
                                    lats.append(time.perf_counter() - t0)
                                else:
                                    fail += 1
                                total_requests += 1
                        except Exception:
                            fail += 1
                            total_requests += 1
                        await asyncio.sleep(0)

            tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
            await asyncio.sleep(duration)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            avg = statistics.mean(lats) if lats else 0.0
            p95 = sorted(lats)[int(0.95 * len(lats))] if lats else 0.0
            thr = succ / duration
            err = fail / total_requests if total_requests else 1.0

            point = {
                "concurrency": concurrency,
                "throughput": thr,
                "avg_latency": avg,
                "p95_latency": p95,
                "error_rate": err,
                "total_requests": total_requests,
                "success_count": succ,
                "failure_count": fail,
            }
            results.append(point)

            logger.info(f"  Concorrência {concurrency}: throughput={thr:.2f} req/s | latência={avg:.3f}s | p95={p95:.3f}s | erros={err*100:.1f}%")

            with open(fname, "w") as f:
                json.dump(results, f, indent=2)

    logger.info(f"✅ Teste concluído. Resultados em {fname}")
    return Response(
        content=json.dumps({
            "message": "Teste de carga concluído.",
            "file": fname.name,
            "params": {
                "ramp_max": ramp_max,
                "step": step,
                "duration": duration,
                "timeout": timeout,
                "mode": "dialog" if use_speakers else "simple",
                "speakers": speakers if use_speakers else None,
                "voice": voice if not use_speakers else None,
                "ambient_file": ambient_file,
                "ambient_volume": ambient_volume,
                "effects": effects,
            },
            "data": results,
        }, indent=2),
        media_type="application/json"
    )

# ---------- Endpoints de saúde ----------
@app.get("/started")
async def started():
    return Response(status_code=200, content="started")

@app.get("/ready")
async def ready():
    async with warmup_lock:
        if warmup_complete:
            return Response(status_code=200, content="ready")
        return Response(status_code=503, content="loading models")

@app.get("/live")
async def live():
    return Response(status_code=200, content="alive")

@app.get("/health")
async def health():
    async with warmup_lock:
        status = "ok" if warmup_complete else "loading"
    return Response(
        content=json.dumps({
            "status": status,
            "voices": list(VOICE_PATHS.keys()),
            "workers": {"tts": TTS_WORKERS, "mix": MIX_WORKERS},
            "physical_cores": PHYSICAL_CORES,
            "hyper_thread_cores": HYPER_THREAD_CORES,
            "concurrency_limit": MAX_CONCURRENT_REQUESTS,
            "timeout": REQUEST_TIMEOUT,
            "warmup_complete": warmup_complete,
        }, indent=2),
        media_type="application/json"
    )

# ---------- Ponto de entrada ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
