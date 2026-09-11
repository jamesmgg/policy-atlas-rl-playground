import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    port: int
    checkpoint_dir: Path
    checkpoint_every_n: int
    max_episodes: int
    use_gpu: bool
    seed: int = 42
    eval_episodes: int = 10
    cpu_threads: int = 1


def load_settings() -> Settings:
    return Settings(
        port=int(os.environ.get("PORT", "8901")),
        checkpoint_dir=Path(os.environ.get("CHECKPOINT_DIR", "/app/data/checkpoints")),
        checkpoint_every_n=int(os.environ.get("CHECKPOINT_EVERY_N", "50")),
        max_episodes=int(os.environ.get("MAX_EPISODES", "5000")),
        use_gpu=os.environ.get("USE_GPU", "0") == "1",
        seed=int(os.environ.get("SEED", "42")),
        eval_episodes=max(1, int(os.environ.get("EVAL_EPISODES", "10"))),
        cpu_threads=max(1, min(64, int(os.environ.get("TORCH_NUM_THREADS", "1")))),
    )


settings = load_settings()
