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

Edit `configs/sdxl_i2i.yaml` to change `prompt`/`negative_prompt`, offload, model ids, or `blend_width`. Use `control_nets` to specify one or more ControlNets and per-ControlNet `scale`. Canny IDs create edge control maps; depth IDs use Depth Anything 3. Outputs are saved as `input_image.png`, `control_map_*.png`, and `output_image.png` in `output_dir`.
