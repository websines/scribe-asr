"""All tunables in one place."""

from pydantic import BaseModel


class GPUConfig(BaseModel):
    device: int = 0
    vram_limit_gb: float = 8.0
    vram_total_gb: float = 24.0   # RTX 3090


class ASRConfig(BaseModel):
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    lookahead_frames: int = 6     # 0→80ms, 1→160ms, 6→560ms, 13→1120ms
    left_context: int = 70
    encoder_step_ms: int = 80     # FastConformer fixed
    sample_rate: int = 16000

    @property
    def chunk_ms(self) -> int:
        return (self.lookahead_frames + 1) * self.encoder_step_ms

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)


class VADConfig(BaseModel):
    threshold: float = 0.5
    min_silence_ms: int = 300     # silence before we consider evaluating turn end
    speech_pad_ms: int = 30
    window_samples: int = 512     # 32ms at 16kHz — Silero's required chunk size


class TurnDetectorConfig(BaseModel):
    model_repo: str = "pipecat-ai/smart-turn-v3"
    model_file: str = "smart-turn-v3.2-cpu.onnx"
    threshold: float = 0.5       # probability above this = turn complete
    audio_window_sec: float = 8.0  # last N seconds fed to model
    cpu_threads: int = 1


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 50


class Settings(BaseModel):
    gpu: GPUConfig = GPUConfig()
    asr: ASRConfig = ASRConfig()
    vad: VADConfig = VADConfig()
    turn: TurnDetectorConfig = TurnDetectorConfig()
    server: ServerConfig = ServerConfig()


settings = Settings()


# --- Code Term Replacements ---
CODE_TERMS: dict[str, str] = {
    "o of n": "O(n)",
    "o of n squared": "O(n²)",
    "o of n log n": "O(n log n)",
    "o of one": "O(1)",
    "o of log n": "O(log n)",
    "big o": "Big-O",
    "b f s": "BFS",
    "d f s": "DFS",
    "d p": "DP",
    "leet code": "LeetCode",
    "get hub": "GitHub",
    "git hub": "GitHub",
    "cube control": "kubectl",
    "cube c t l": "kubectl",
    "pie torch": "PyTorch",
    "tensor flow": "TensorFlow",
    "num pie": "NumPy",
    "jay son": "JSON",
    "j son": "JSON",
    "next js": "Next.js",
    "node js": "Node.js",
    "post gres": "Postgres",
    "my sequel": "MySQL",
    "no sequel": "NoSQL",
    "mongo d b": "MongoDB",
    "docket": "Docker",
    "end point": "endpoint",
    "a p i": "API",
    "rest a p i": "REST API",
    "graph q l": "GraphQL",
    "web socket": "WebSocket",
    "j w t": "JWT",
    "o auth": "OAuth",
    "c i c d": "CI/CD",
    "hash map": "hashmap",
    "hash set": "hashset",
    "two pointer": "two-pointer",
    "breadth first": "breadth-first",
    "depth first": "depth-first",
}
