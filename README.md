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
