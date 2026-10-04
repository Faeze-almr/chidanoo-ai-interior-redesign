# Licenses and Sources

This file is an attribution and provenance inventory, not legal advice and not a license grant for CHIDANOO itself. The application was developed as a three-person capstone. No repository-level open-source license has been granted, so reuse or redistribution requires permission from the project owners.

## Models and learned components

| Component | Upstream source | Declared license / status | Use in this project |
|---|---|---|---|
| Stable Diffusion XL Base 1.0 | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 | CreativeML Open RAIL++-M / `openrail++`; review use-based restrictions | Base image model |
| SDXL Depth ControlNet | https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0 | `openrail++`; review model card and license | Structure guidance |
| SDXL Canny ControlNet | https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0 | `openrail++`; review model card and license | Edge guidance |
| SDXL VAE FP16 Fix | https://huggingface.co/madebyollin/sdxl-vae-fp16-fix | MIT | VAE replacement |
| IP-Adapter | https://huggingface.co/h94/IP-Adapter | Apache-2.0 | Reference-image guidance; SDXL ViT-H adapter weight and encoder folder |
| MiDaS annotator weights | https://huggingface.co/lllyasviel/Annotators | **License: verification required** (repository metadata says `other` and provides no complete license record) | Depth-map preprocessing |

OpenRAIL licenses are not equivalent to an unrestricted permissive software license. Users must review the applicable model license, acceptable-use restrictions, and downstream deployment obligations.

## Direct software dependencies

The package installs dependencies from their package registries; their source is not vendored here. Versions/ranges are defined in `backend/pyproject.toml`, `backend/requirements-kaggle.txt`, and `frontend/package-lock.json`.

| Dependency | Upstream | License |
|---|---|---|
| PyTorch | https://github.com/pytorch/pytorch | BSD-style; consult upstream third-party notices |
| Diffusers | https://github.com/huggingface/diffusers | Apache-2.0 |
| Transformers | https://github.com/huggingface/transformers | Apache-2.0 |
| Accelerate | https://github.com/huggingface/accelerate | Apache-2.0 |
| Safetensors | https://github.com/huggingface/safetensors | Apache-2.0 |
| huggingface_hub (Kaggle requirements) | https://github.com/huggingface/huggingface_hub | Apache-2.0 |
| controlnet-aux | https://github.com/huggingface/controlnet_aux | Apache-2.0 |
| OpenCV | https://github.com/opencv/opencv | Apache-2.0 for current 4.x releases; consult bundled third-party notices |
| NumPy | https://github.com/numpy/numpy | BSD-3-Clause; wheels include additional third-party notices |
| Pillow | https://github.com/python-pillow/Pillow | HPND |
| FastAPI | https://github.com/fastapi/fastapi | MIT |
| Uvicorn | https://github.com/Kludex/uvicorn | BSD-3-Clause |
| Pydantic / pydantic-settings | https://github.com/pydantic/pydantic | MIT |
| python-dotenv | https://github.com/theskumar/python-dotenv | BSD-3-Clause |
| python-multipart | https://github.com/Kludex/python-multipart | Apache-2.0 |
| Gradio | https://github.com/gradio-app/gradio | Apache-2.0 |
| pytest | https://github.com/pytest-dev/pytest | MIT |
| HTTPX | https://github.com/encode/httpx | BSD-3-Clause |
| React / React DOM | https://github.com/facebook/react | MIT |
| Motion (installed as `framer-motion`) | https://github.com/motiondivision/motion | MIT |
| Vite and React plugin | https://github.com/vitejs/vite | MIT; published builds may include separately noticed bundled dependencies |
| TypeScript | https://github.com/microsoft/TypeScript | Apache-2.0; consult NOTICE |
| Tailwind CSS | https://github.com/tailwindlabs/tailwindcss | MIT |
| PostCSS | https://github.com/postcss/postcss | MIT |
| Autoprefixer | https://github.com/postcss/autoprefixer | MIT |
| React/Node type packages | https://github.com/DefinitelyTyped/DefinitelyTyped | MIT (individual packages may include upstream notices) |

Transitive dependencies remain governed by their own licenses. For redistribution, preserve registry license files/notices and audit the exact resolved environment, not only this direct-dependency summary.

## Fonts

| Font | Delivery/source | License |
|---|---|---|
| Poppins | Google Fonts stylesheet; https://github.com/itfoundry/Poppins | SIL Open Font License 1.1 |
| Playfair Display | Google Fonts stylesheet; https://github.com/google/fonts/tree/main/ofl/playfairdisplay | SIL Open Font License 1.1 |
| Shabnam 5.0.1 | jsDelivr from https://github.com/rastikerdar/shabnam-font | SIL Open Font License 1.1 |

The CSS references these fonts remotely; font binaries are not included in this source ZIP.

## Images and visual assets

### Evaluation benchmark

The evaluation record describes 20 fixed images: 16 Core and 4 Stress; 15 sourced from Pexels and 5 supplied by users. The images and an item-level manifest were not present in the audited repository and are not included in this clean package. The general Pexels license is at https://www.pexels.com/license/ and its full terms at https://www.pexels.com/terms-of-service/.

**Provenance gap:** the 15 individual Pexels photo-page URLs, photographers, download dates, and contemporaneous license records were not recoverable from the project. Do not invent them. Retain or reconstruct an external item-level manifest and verify current ML/evaluation and redistribution permissions before reusing the benchmark. The 5 user-provided images likewise require documented contributor permission.

### UI style thumbnails

The original capstone package included room-style thumbnails without retained source or license metadata. They are excluded from the public Git history. The public React selector renders CSS-generated abstract previews instead.

### Brand assets

`frontend/public/branding/logo-en.png` and `logo-fa.png` were supplied as CHIDANOO brand assets. No third-party source or license was identified. Treat them as project-owned only after the owner confirms authorship/rights; trademark rights are separate from copyright.

### Test fixture

`test_fixtures/room_04_furniture_mask.png` is a binary mask annotation used to verify the Targeted workflow. The underlying benchmark room image is not embedded in this mask or included in the package. Retain the annotation's project provenance with the external benchmark manifest.

## Practical release checks

Before public or commercial distribution:

1. Confirm a project-level license or written owner permission for the CHIDANOO source.
2. Review every model card/license, especially OpenRAIL terms and the unresolved MiDaS annotator license.
3. Verify the seven style-thumbnail rights or replace them with documented project-owned/licensed assets.
4. Restore item-level benchmark provenance and verify Pexels/user permissions for the intended evaluation use.
5. Preserve dependency and font license notices for the exact built artifacts.
