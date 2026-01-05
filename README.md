# DrivingForward with gsplat

## How to use

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
uv run drivingforward-gsplat
```

### SDXL i2i strip panorama

```bash
uv run python -m drivingforward_gsplat.i2i.sdxl_panorama_i2i \
  --config configs/sdxl_panorama_i2i.yaml \
  --prompt-config configs/prompts/sunset.yaml
```

Edit `configs/sdxl_panorama_i2i.yaml` to change offload, model ids, or `blend_width`. Prompts are provided via `--prompt-config` (for example, `configs/prompts/sunset.yaml`), and reference images for IP-Adapter go in `reference_images` inside the prompt config. Use `control_nets` to specify one or more ControlNets and per-ControlNet `scale`. Canny IDs create edge control maps; depth IDs use Depth Anything 3. Outputs are saved in `output_dir`.

### Predict gaussian

```bash
uv run python -m drivingforward_gsplat.predict_gaussian \
  --predict-config configs/predict_gaussian.yaml
```

Edit `configs/predict_gaussian.yaml` to point at the model config, set the dataset split/index, and choose the torchscript directory and output path. The default `sdxl_panorama_i2i_config` is always loaded, and i2i only runs when a prompt config is provided.

#### Predict gaussian with SDXL i2i

```bash
uv run python -m drivingforward_gsplat.predict_gaussian \
  --predict-config configs/predict_gaussian.yaml \
  --sdxl-panorama-prompt-config configs/prompts/sunset.yaml
```

Use `--sdxl-panorama-prompt-config` to enable SDXL panorama i2i. The prompt config format matches the SDXL i2i tool and supports `reference_images` for IP-Adapter.

### Optimize gaussian (gsplat)

```bash
uv run python -m drivingforward_gsplat.predict_gaussian \
  --predict-config configs/predict_gaussian.yaml \
  --optimize-gaussian-config configs/optimize_gaussian.yaml
```

This runs prediction, then optimizes the exported gaussians with gsplat using raw NuScenes views as anchors and Fixer outputs as regularization. Sky masks are excluded from losses and merge/densify, and Fixer is weighted lightly to avoid overfitting. Update `configs/optimize_gaussian.yaml` to tune view counts per phase, merge frequency, sigma minimums, Fixer weights, and the per-phase camera jitter range (cm) used for viewpoint robustness.
