# CHIDANOO

CHIDANOO is a bilingual (English/Persian) AI interior-redesign application. It ships one canonical Python inference core, a FastAPI API, a React interface, and an optional Gradio interface. The three supported workflows are:

- **Full Redesign** — redesigns the room while using depth and edge structure guidance.
- **Targeted Inpainting** — edits only a user-painted region and restores every pixel outside the mask exactly.
- **Reference Redesign** — transfers design cues from a reference image through the SDXL IP-Adapter path.

## Recognition and project scope

CHIDANOO was developed as the three-person final capstone for Hamrah Academy's (MCI) Generative AI for Image bootcamp and placed **3rd** in the final project ranking. Faeze Ali-Mirzaee's contribution focused on model and pipeline comparison, controlled experiments, failure analysis, runtime/VRAM tracking, and integration of the selected workflow into the final deliverable.

This public repository excludes the original room-style thumbnails because their redistribution metadata was not retained. The React UI uses self-contained CSS previews instead. Model weights, benchmark images, and private evaluation artifacts are also excluded.

The application requires an NVIDIA CUDA GPU for real generation. CPU-only environments can run the static/unit checks, but generation endpoints intentionally return a clear unavailable response instead of producing mock output.

## Repository layout

```text
chidanoo-ai-interior-redesign/
├── assets/                 # Gradio styling
├── backend/                # FastAPI app, canonical core, tests, dependency metadata
├── docs/                   # Architecture, evaluation summary, known limitations
├── frontend/               # React/Vite bilingual interface
├── scripts/                # CUDA smoke-validation command
├── test_fixtures/          # Small deterministic test asset(s)
├── FINAL_CONFIG.json       # Frozen inference/model configuration
├── gradio_app.py           # Optional Gradio launcher
├── LICENSES_AND_SOURCES.md
└── README.md
```

## Frozen configuration

`backend/app/core/final_config.py` is the runtime source of the frozen defaults. `FINAL_CONFIG.json` is the machine-readable delivery mirror used for review and handoff; both representations are kept aligned.

| Workflow | Seed | Steps | CFG | Depth | Canny | Workflow-specific value |
|---|---:|---:|---:|---:|---:|---|
| Full Redesign | 42 | 30 | 7.5 | 0.50 | 0.20 | — |
| Targeted Inpainting | 42 | 30 | 7.5 | 0.40 | 0.10 | strength 1.0; threshold 128; exact outside-mask restoration |
| Reference Redesign | 42 | 30 | 7.5 | 0.50 | 0.20 | IP-Adapter scale 0.60 |

Canonical model IDs, adapter weight, resize policy, mask semantics, scheduler, and CUDA dtype are frozen across the runtime configuration and `FINAL_CONFIG.json`; do not retune them during evaluation.

## Local setup

Prerequisites: Python 3.11 or 3.12, Node.js 18+, npm, and an NVIDIA CUDA environment for generation. Model weights are downloaded by their upstream libraries on first use.

Backend (PowerShell):

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,ui]"
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend (bash):

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,ui]'
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend, in a second terminal:

```bash
cd frontend
cp .env.example .env
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The health endpoint is `http://127.0.0.1:8000/api/health`, and FastAPI documentation is at `http://127.0.0.1:8000/docs`.

The frontend always calls the real backend. No mock production mode is included.

## Optional Gradio UI

After installing the backend with the `ui` extra, run from the repository root:

```bash
python gradio_app.py
```

Gradio calls the same `generate_room_result()`, `generate_region_result()`, and `generate_reference_result()` functions as FastAPI; it does not duplicate inference logic.

## Static verification

```bash
cd backend
python -m pytest -q

cd ../frontend
npm ci
npm run typecheck
npm run build
```

CUDA smoke validation, from the repository root:

```bash
python scripts/gpu_smoke_test.py --output-dir outputs/gpu_smoke
```

The smoke tool records workflow status, runtime, peak VRAM, frozen generation values, and `outside_mask_mae` for Targeted Inpainting. It must be run on the intended NVIDIA environment before claiming GPU PASS. Kaggle-specific pinned packages are in `backend/requirements-kaggle.txt`; it contains `controlnet-aux==0.0.10` and intentionally does not replace Kaggle's CUDA-enabled PyTorch build.

## Outputs and evaluation

- FastAPI job images and metadata are written under `backend/storage/` at runtime.
- Durable UI history is stored under `backend/storage/history/` and can be exported through `/api/history/export`.
- The smoke script writes images and JSON/log evidence to the output directory supplied on the command line.
- Generated/runtime data is ignored by Git and is not included in this source package.

The final evaluator notebook is a separate deliverable and is intentionally not embedded in this repository. The formal report is likewise kept separate. See `docs/EVALUATION_SUMMARY.md` for the evidence that is safe to claim and `docs/KNOWN_LIMITATIONS.md` before evaluation.

## Documentation

- `docs/ARCHITECTURE.md` — system and workflow architecture
- `docs/EVALUATION_SUMMARY.md` — consolidated benchmark and final-validation evidence
- `docs/KNOWN_LIMITATIONS.md` — operational and evidence boundaries
- `LICENSES_AND_SOURCES.md` — models, software, fonts, images, and provenance concerns
