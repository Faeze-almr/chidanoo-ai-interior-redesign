# Evaluation Summary

This document consolidates only recorded evidence. It does not turn a CPU/static check into a GPU claim.

## Evaluation protocol

- Fixed benchmark: 20 images, comprising 16 Core and 4 Stress cases.
- Reported provenance: 15 Pexels images and 5 user-provided images.
- Images were intended to be evaluated without cherry-picking.
- Runtime excludes initial model loading and is measured around generation.
- Peak VRAM is `torch.cuda.max_memory_allocated`.
- Targeted protection is measured as mean absolute pixel error outside the binary mask.

The benchmark images are not bundled in this clean source package. Individual Pexels source URLs were not present in the audited repository, so the external benchmark manifest must be retained separately before the dataset is redistributed or reused. See `LICENSES_AND_SOURCES.md`.

## Consolidated results

| Evaluation | Coverage | Result | Runtime | Peak VRAM | Status |
|---|---:|---:|---:|---:|---|
| T0 baseline | 20/20 images | completed | 13.34 s/image average | 4.95 GB | recorded baseline |
| Candidate B Fast | 6/6 subset | completed | ~97.17 s/image | ~12.80 GB | candidate evidence |
| Final Full Redesign | `room_16` | completed | ~101.37 s | ~13.01 GB | PASS with minor visual limitation |
| Final Targeted Inpainting | `room_04` | completed | ~100.44 s | ~13.02 GB | PASS; outside-mask MAE 0.0000 |
| Final Reference Redesign | — | no complete CUDA evidence retained | — | — | validation pending |

## Interpretation

The final Full run demonstrates the frozen SDXL + Depth/Canny configuration on the recorded case, with a minor visual limitation noted in the original evaluation evidence. The final Targeted run demonstrates both successful generation and exact outside-mask restoration (`outside_mask_mae = 0.0000`) on the recorded case.

No final Reference GPU PASS is claimed because the audited evidence did not contain a completed successful Reference output with runtime/VRAM records. Run the canonical smoke command on the target NVIDIA environment and retain its output image plus JSON/log evidence before changing that status.

## Required per-run record

For each workflow retain: `status`, `runtime_sec`, `peak_vram_gb`, `seed`, `guidance_scale`, `depth_scale`, `canny_scale`, and workflow-specific `strength` or `reference_scale`. Targeted Inpainting must additionally retain `outside_mask_mae`. A PASS also requires a visually usable output and no workflow error.

## Reproduction entry point

From the repository root on a configured CUDA machine:

```bash
python scripts/gpu_smoke_test.py --output-dir outputs/gpu_smoke
```

Do not retune `FINAL_CONFIG.json` while reproducing these records.
