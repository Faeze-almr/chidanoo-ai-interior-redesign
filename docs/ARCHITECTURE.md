# Architecture

## System overview

CHIDANOO has one inference implementation and two presentation layers:

```text
React/Vite UI ──multipart HTTP──> FastAPI routes ──> single async job queue
                                                    │
Gradio UI ──────────────────────────────────────────┤
                                                    ▼
                               canonical pipeline.py result functions
                                                    │
                       SDXL + Depth/Canny ControlNet + MiDaS
                             ├─ Full Redesign
                             ├─ Targeted Inpainting
                             └─ Reference Redesign + IP-Adapter
                                                    │
                                  PNG/JSON history and downloads
```

The React interface is the full bilingual product UI. It supports English and Persian directionality, stable image/mask editing, progress polling, before/after comparison, history, gallery, and downloads. Gradio is a lightweight evaluator/demo surface over the same core.

## Backend

`backend/app/main.py` creates the FastAPI application and starts one in-process asynchronous job worker. Requests are accepted by three multipart endpoints:

- `POST /api/generate/room`
- `POST /api/edit/region`
- `POST /api/reference/room`

Clients poll `GET /api/jobs/{job_id}` and retrieve the output at `GET /api/jobs/{job_id}/image`. File-backed history is exposed by `/api/history` and its asset/export endpoints. The deployment model is intentionally single-process and single-GPU; it does not require a separate broker or database.

## Canonical inference core

`backend/app/core/pipeline.py` exposes the only three production entry points:

- `generate_room_result()`
- `generate_region_result()`
- `generate_reference_result()`

Model loading is lazy. Importing the application does not download weights. CUDA is checked before production generation. The full and inpainting workflows share the canonical SDXL components. Reference mode uses an ephemeral adapter-enabled pipeline with CPU offload and full cleanup so adapter processors do not mutate the cached Full/Targeted pipeline.

Preprocessing resizes to a 1024-pixel short side and an 8-pixel multiple, then computes MiDaS depth and Canny controls. Targeted masks are binary (`L`, threshold 128), align to the image with nearest-neighbor resampling, and treat white as editable. After generation, the original input is composited back outside the mask, enforcing exact protection.

## Configuration and contracts

Frozen runtime defaults live in `backend/app/core/final_config.py`, with `FINAL_CONFIG.json` maintained as the delivery mirror. The final package audit checks that the declared values remain aligned. Request schemas and routes expose optional advanced controls but default to the frozen values. Prompt assembly remains centralized in `backend/app/core/prompts.py`.

Errors from validation, CUDA checks, model loading, generation, and storage are mapped to user-safe responses. Successful generation is not discarded if history persistence fails; the UI receives the result plus a history warning.

## Storage

Runtime files are project-relative:

- transient job uploads/results: `backend/storage/uploads/` and `backend/storage/results/`
- durable history: `backend/storage/history/`
- explicit smoke-test output: caller-selected `--output-dir`

No workstation-specific absolute path is required. Environment overrides are optional and loaded from `backend/.env`; secrets and runtime data are excluded from the package.

## Frontend

The Vite/React frontend calls the FastAPI contracts directly through `frontend/src/lib/api.ts`. Full, Targeted, and Reference views submit multipart requests, poll jobs with bounded transient-error retries, and render completed images. The Targeted editor maintains one stable object URL for both its visible preview and canvas background, revoking it only when replaced or unmounted.

The UI uses local brand/style assets under `frontend/public/`, remote font stylesheets, and responsive EN/FA layouts. It does not start FastAPI or embed inference code.
