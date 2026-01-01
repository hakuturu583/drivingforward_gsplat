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

Edit `configs/sdxl_i2i.yaml` to change prompts, tiling, offload, or model ids.

With tiling and CPU offload to reduce memory use:

```bash
uv run python -m drivingforward_gsplat.i2i.sdxl_i2i \
  --config configs/sdxl_i2i.yaml
```

If you still see CUDA OOM, enable sequential offload and reduce tile size:

```bash
uv run python -m drivingforward_gsplat.i2i.sdxl_i2i \
  --config configs/sdxl_i2i.yaml
```
