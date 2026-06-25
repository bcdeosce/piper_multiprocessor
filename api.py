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
from concurrent.futures import ProcessPoolExecutor
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

# ---------- Workers (configuração otimizada) ----------
TTS_WORKERS = int(os.getenv("TTS_WORKERS", 8))
MIX_WORKERS = int(os.getenv("MIX_WORKERS", 5))
logger.info(f"Workers: TTS={TTS_WORKERS} processos, Mix={MIX_WORKERS} processos")

# ---------- Inicializador dos workers TTS ----------
def _init_tts_worker():
    # Força o ONNX Runtime e OpenMP a usarem apenas 1 thread
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ORT_NUM_THREADS"] = "1"
    ort.set_default_logger_severity(3)

    # Define afinidade do processo (thread principal)
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

    # Armazena o cpu_id no módulo global para uso posterior
    mod = sys.modules['__main__']
    mod._worker_cpu_id = cpu_id
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

# ---------- VoicePool com controle fino do ONNX Runtime ----------
class VoicePool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = 1, cpu_id: int = 0):
        import queue
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            # Cria as opções da sessão ONNX
            sess_options = ort.SessionOptions()
            
            # Força o uso de APENAS 1 thread por worker (essencial para o isolamento)
            sess_options.intra_op_num_threads = 1
            
            # Define a afinidade das threads internas para o núcleo específico do worker
            # O formato é uma string com os IDs dos cores: '0;1;2' para os cores 0, 1 e 2
            sess_options.add_session_config_entry(
                'session.intra_op_thread_affinities', 
                str(cpu_id)  # Ex: '5' para fixar no core 5
            )
            
            # Cria a sessão do ONNX Runtime
            session = ort.InferenceSession(
                model_path, 
                sess_options, 
                providers=['CPUExecutionProvider']
            )
            
            # Carrega a voz passando a sessão já criada
            voice = PiperVoice.load(
                model_path, 
                config_path=config_path, 
                session=session, 
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
def get_voice_pool(voice_name, cpu_id):
    mod = sys.modules['__main__']
    cache = getattr(mod, '_worker_voice_cache', None)
    if cache is None:
        cache = {}
        mod._worker_voice_cache = cache
    if voice_name not in cache:
        model_path, config_path = VOICE_PATHS[voice_name]
        pool = VoicePool(model_path, config_path, pool_size=1, cpu_id=cpu_id)
        cache[voice_name] = pool
    return cache[voice_name]

# ---------- Síntese de um fragmento (texto) ----------
def synthesize_text(voice_name, text, speed, noise_scale, noise_w_scale, cpu_id):
    pool = get_voice_pool(voice_name, cpu_id)
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

# ---------- Mixagem usando FFmpeg (corrigida) ----------
def mix_and_export_task(segments_data, ambient_cfg, target_rate=22050):
    t0 = time.perf_counter()
    temp_files = []
    ffmpeg_cmd = ["ffmpeg", "-y"]

    try:
        for data in segments_data:
            if 'pcm_bytes' in data:
                # Cria um arquivo WAV com cabeçalho
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

        # Adicionar ambiente
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

# ---------- Processamento de uma requisição inteira (dentro do worker TTS) ----------
def process_entire_request(
    voice_name: Optional[str],
    text: str,
    speed: float,
    noise_scale: float,
    noise_w_scale: float,
    effects: Dict[str, str],
    speakers: List[Dict],
    ambient_cfg: Dict,
    enqueue_time: float,  # timestamp quando a tarefa foi enviada
    cpu_id: int,          # ID do núcleo onde este worker está fixado
) -> Tuple[bytes, Dict[str, float]]:
    """
    Retorna (wav_bytes, metrics) onde metrics contém:
        - queue_wait: tempo de espera na fila (desde o enqueue até o início)
        - synth_time: soma do tempo de síntese de todos os fragmentos
        - mix_time: tempo gasto na mixagem (FFmpeg)
        - total_worker_time: tempo total dentro do worker (incluindo overhead)
        - num_segments: número de segmentos processados
    """
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
        sample_rate, pcm_bytes = synthesize_text(v_name, part, spd, ns, nw, cpu_id)
        synth_time_total += time.perf_counter() - t_synth_start
        segments.append({'pcm_bytes': pcm_bytes, 'sample_rate': sample_rate})

    # Mixagem
    wav_bytes, mix_time = mix_and_export_task(segments, ambient_cfg, target_rate=22050)

    total_worker_time = time.perf_counter() - t_worker_start

    metrics = {
        'queue_wait': queue_wait,
        'synth_time': synth_time_total,
        'mix_time': mix_time,
        'total_worker_time': total_worker_time,
        'num_segments': len(segments),
    }

    return wav_bytes, metrics

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
app = FastAPI(title="Piper TTS API (Requisição Inteira por Worker)")

# ---------- Estatísticas (agora com múltiplas métricas) ----------
stats = defaultdict(list)
stats_lock = asyncio.Lock()

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    t_total_start = time.perf_counter()

    # Converter speakers para lista de dicionários
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

    # Captura o timestamp antes de enviar para a fila
    enqueue_time = time.perf_counter()

    loop = asyncio.get_running_loop()
    try:
        # Obtém o cpu_id do worker que vai executar (precisamos passar para o worker)
        # Como não temos como saber qual worker será, vamos passar None e o worker
        # vai pegar o cpu_id do seu próprio contexto (via módulo global)
        wav_bytes, metrics = await loop.run_in_executor(
            tts_pool,
            process_entire_request,
            req.voice,
            req.text,
            req.speed,
            req.noise_scale,
            req.noise_w_scale,
            req.effects,
            speakers_list,
            ambient_dict,
            enqueue_time,
            None  # cpu_id será obtido dentro do worker via módulo global
        )
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        raise HTTPException(500, f"Falha no processamento: {str(e)}")

    t_total = time.perf_counter() - t_total_start

    # Acumula todas as métricas
    async with stats_lock:
        stats['total'].append(t_total)
        stats['queue_wait'].append(metrics['queue_wait'])
        stats['synth_time'].append(metrics['synth_time'])
        stats['mix_time'].append(metrics['mix_time'])
        stats['total_worker_time'].append(metrics['total_worker_time'])
        stats['num_segments'].append(metrics['num_segments'])

    logger.info(
        f"⏱️ Requisição concluída em {t_total:.3f}s | "
        f"fila={metrics['queue_wait']:.3f}s | synth={metrics['synth_time']:.3f}s | "
        f"mix={metrics['mix_time']:.3f}s | worker={metrics['total_worker_time']:.3f}s | "
        f"segmentos={metrics['num_segments']}"
    )
    return Response(content=wav_bytes, media_type="audio/wav")

# ---------- Endpoint de estatísticas (AGORA COM MÉTRICAS DETALHADAS) ----------
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
