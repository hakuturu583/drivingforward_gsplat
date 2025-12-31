# DrivingForward with gsplat

## How to use

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
uv run drivingforward-gsplat
```

### Flux i2i strip panorama

```bash
uv run python -m drivingforward_gsplat.i2i.flux_i2i \
  --prompt "sunset urban street, cinematic" \
  --output flux_strip.png \
  --novel_view_mode MF \
  --sample_index 0
```

With tiling and CPU offload to reduce memory use:

```bash
uv run python -m drivingforward_gsplat.i2i.flux_i2i \
  --prompt "sunset urban street, cinematic" \
  --output flux_strip.png \
  --novel_view_mode MF \
  --sample_index 0 \
  --cpu_offload \
  --tile_size 512 \
  --tile_overlap 96
```

If you still see CUDA OOM, enable sequential offload and reduce tile size:

```bash
uv run python -m drivingforward_gsplat.i2i.flux_i2i \
  --prompt "sunset urban street, cinematic" \
  --output flux_strip.png \
  --novel_view_mode MF \
  --sample_index 0 \
  --cpu_offload \
  --sequential_offload \
  --tile_size 384 \
  --tile_overlap 64
```
