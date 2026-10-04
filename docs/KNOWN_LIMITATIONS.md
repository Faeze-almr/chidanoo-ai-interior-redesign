# Known Limitations

- Real generation requires a CUDA-capable NVIDIA GPU and enough VRAM for SDXL plus dual ControlNets. CPU-only startup and tests are supported; CPU generation is intentionally rejected.
- The first generation requires network access and substantial local disk space for model downloads. Upstream model availability and access terms can change.
- The server is designed for a single process and one serialized GPU worker. In-memory job status is lost on restart; completed file-backed history remains.
- Final Reference Redesign does not yet have retained end-to-end CUDA PASS evidence in the audited evaluation record. Do not claim a Reference GPU PASS until the smoke output and metrics are captured.
- The recorded Full Redesign case passed with a minor visual limitation. Structure guidance reduces geometry drift but cannot guarantee architectural or photorealistic perfection.
- Targeted exact protection applies outside the submitted binary mask. A poorly aligned or overly broad mask can still produce an undesirable edit inside the editable region.
- Remote fonts require network access. If unavailable, the browser falls back to local font stacks; functionality remains intact but typography may differ.
- The package does not include model weights, the 20-image benchmark, the evaluator notebook, or the formal report.
- Source URLs and rights metadata for the seven bundled style thumbnails were not present in the audited source. Their licenses require verification before public redistribution.
