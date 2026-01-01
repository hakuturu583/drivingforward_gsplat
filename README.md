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
uv run python -m drivingforward_gsplat.i2i.sdxl_i2i \
  --config configs/sdxl_i2i.yaml
```

Edit `configs/sdxl_i2i.yaml` to change prompts, offload, model ids, or `blend_width`. Set `controlnet_id` to `diffusers/controlnet-canny-sdxl-1.0` to use Canny edges (depth model is ignored). Outputs are saved as `input_image.png`, `control_map.png`, and `output_image.png` in `output_dir`.
