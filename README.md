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

Edit `configs/predict_gaussian.yaml` to point at the model config, set the dataset split/index, and choose the torchscript directory and output path.
