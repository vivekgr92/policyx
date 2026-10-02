# VLM Judge Test Fixtures

Real recorded episodes copied from `vivekgr92/lerobot-dataset` (local cache), kept here as a
standing test set for evaluating the VLM judge - no need to re-extract from the dataset each time.

| File | Source | Episode index | Real task | Notes |
|---|---|---|---|---|
| `episode_2.mp4` | `videos/observation.images.front/chunk-000/file-002.mp4` | 2 | `Pick A to B` | Real motion is visible on inspection (cups move from clustered-left to clustered-right), but the judge has repeatedly returned a false-negative `INVALID - object was never picked up or moved` on this one across several fixes (frame selection, prompt grounding, middle-frame coverage). Root cause narrowed down to a genuine model-perception limitation (likely washed-out/low-contrast starting frame + no true temporal/video encoding in Ollama's serving path - see session history). Good adversarial case for testing any future judge-accuracy fix. |
| `episode_3.mp4` | `videos/observation.images.front/chunk-000/file-003.mp4` | 3 | `Pick B to A` | |

Camera: `observation.images.front`, 1280x720, 30fps, ~900 frames each (~30s).
