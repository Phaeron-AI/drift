# DRIFT - Folder Structure


```
drift/
│
├── docker-compose.yml             # Orchestrates Redis, MinIO, and the Node Gateway locally
├── Makefile                       # Helper commands (e.g., `make dev`, `make train`)
├── .gitignore                     # Global gitignore (ignores node_modules, checkpoints, venvs)
│
└── apps/
    │
    ├── web/                       # Tier 1: Client Layer (React / TypeScript)
    │   ├── package.json
    │   ├── tailwind.config.js
    │   └── src/
    │       ├── components/        # Canvas interactive UI, upload buttons, video player
    │       ├── store/             # Zustand state (holds trajectory vectors before sending)
    │       └── lib/               # API clients communicating with the Node Gateway
    │
    ├── gateway/                   # Tier 2: Orchestration Layer (Node.js / Fastify)
    │   ├── package.json
    │   └── src/
    │       ├── routes/            # HTTP endpoints (e.g., POST /generate-video)
    │       ├── queue/             # BullMQ producer (pushes jobs to Redis)
    │       └── storage/           # MinIO/S3 wrapper (saves raw images and output videos)
    │
    └── engine/                    # Tier 3: Inference Engine (Your PyTorch Code)
        ├── pyproject.toml         # Dependencies (torch, einops, fastapi, uvicorn)
        ├── checkpoints/           # Frozen base weights (SAM2, Diffusion)
        ├── cache/                 # Precomputed masks
        ├── outputs/               # Video outputs and tensorboard logs
        │
        └── src/
            ├── configs/           # YAML hardware/model configs
            ├── scripts/           # verify_env.py, train.py
            ├── tests/             # test_vram_budget.py, test_spatial_attn.py
            │
            └── spatial_dynamics/  # The Core ML Package
                │
                ├── api/           # [NEW] FastAPI wrapper to receive jobs from the queue
                │   ├── main.py    # Uvicorn entry point
                │   └── schemas.py # Pydantic models (validating the JSON vector inputs)
                │
                ├── config/        # Validates configs/*.yaml
                ├── data/          # dataset.py, trajectory.py
                ├── models/        
                │   ├── backbones/ # sam2.py, diffusion.py
                │   ├── attention/ # spatial_cross_attn.py
                │   ├── bridge.py  
                │   └── hybrid.py  
                │
                ├── training/      # losses, trainer.py
                ├── inference/     # pipeline.py
                └── utils/         # vram.py, logging.py
```