import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import viser
from plyfile import PlyData

from gsplat.rendering import rasterization
from nerfview import CameraState, RenderTabState, Viewer, apply_float_colormap


class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controlable parameters
    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: str = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: str = "turbo"
    rasterize_mode: str = "classic"
    camera_model: str = "pinhole"


class GsplatViewer(Viewer):
    """Viewer for gsplat."""

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn,
        output_dir: Path,
        mode: str = "rendering",
    ):
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                max_sh_degree_number = server.gui.add_number(
                    "Max SH",
                    initial_value=self.render_tab_state.max_sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Maximum SH degree used.",
                )

                @max_sh_degree_number.on_update
                def _(_) -> None:
                    self.render_tab_state.max_sh_degree = int(
                        max_sh_degree_number.value
                    )
                    self.rerender(_)

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @near_far_plane_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                    self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                    self.rerender(_)

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = radius_clip_slider.value
                    self.rerender(_)

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to projected covariance eigenvalues.",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    if "depth" in render_mode_dropdown.value:
                        normalize_nearfar_checkbox.disabled = False
                        inverse_checkbox.disabled = False
                    else:
                        normalize_nearfar_checkbox.disabled = True
                        inverse_checkbox.disabled = True
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    disabled=True,
                    hint="Normalize depth with near/far plane.",
                )

                @normalize_nearfar_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.normalize_nearfar = (
                        normalize_nearfar_checkbox.value
                    )
                    self.rerender(_)

                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    disabled=True,
                    hint="Inverse the depth.",
                )

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                rasterize_mode_dropdown = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Rasterization mode.",
                )

                @rasterize_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                    self.rerender(_)

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )

                @camera_model_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.camera_model = camera_model_dropdown.value
                    self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count


def _sorted_field_names(names: tuple[str, ...], prefix: str) -> list[str]:
    indexed = []
    for name in names:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            indexed.append((int(suffix), name))
    return [name for _, name in sorted(indexed)]


def _stack_fields(data, names: list[str]) -> np.ndarray:
    return np.stack([data[name] for name in names], axis=1).astype(np.float32)


def _load_gaussians_from_ply(ply_path: Path, device: torch.device):
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError("PLY file is missing 'vertex' element.")
    data = ply["vertex"].data
    names = data.dtype.names or ()

    means = _stack_fields(data, ["x", "y", "z"])
    quats = _stack_fields(data, ["rot_0", "rot_1", "rot_2", "rot_3"])
    quats_norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(quats_norm, 1e-8, None)

    if "opacity" in names:
        opacity_name = "opacity"
    elif "opacities" in names:
        opacity_name = "opacities"
    else:
        raise ValueError("PLY file is missing opacity values.")
    opacities = data[opacity_name].astype(np.float32).reshape(-1, 1)

    is_inria = "f_dc_0" in names
    if is_inria:
        scales = _stack_fields(data, ["scale_0", "scale_1", "scale_2"])
        scales = np.exp(scales)
        opacities = 1.0 / (1.0 + np.exp(-opacities))

        dc = _stack_fields(data, ["f_dc_0", "f_dc_1", "f_dc_2"])
        rest_fields = _sorted_field_names(names, "f_rest_")
        rest_dim = len(rest_fields) // 3
        if rest_dim:
            rest = _stack_fields(data, rest_fields)
            rest = rest.reshape(rest.shape[0], 3, rest_dim)
        else:
            rest = np.zeros((dc.shape[0], 3, 0), dtype=np.float32)
        sh = np.concatenate([dc[:, :, None], rest], axis=2)
        sh = np.transpose(sh, (0, 2, 1))
    else:
        scales = _stack_fields(data, ["scale_0", "scale_1", "scale_2"])
        sh_fields = _sorted_field_names(names, "sh_")
        if not sh_fields:
            raise ValueError("PLY file is missing SH coefficients.")
        sh_flat = _stack_fields(data, sh_fields)
        if sh_flat.shape[1] % 3 != 0:
            raise ValueError("Unexpected SH coefficient count in PLY file.")
        d_sh = sh_flat.shape[1] // 3
        sh = sh_flat.reshape(sh_flat.shape[0], 3, d_sh).transpose(0, 2, 1)

    means_t = torch.from_numpy(means).to(device=device)
    quats_t = torch.from_numpy(quats).to(device=device)
    scales_t = torch.from_numpy(scales).to(device=device)
    opacities_t = torch.from_numpy(opacities).to(device=device).squeeze(-1)
    sh_t = torch.from_numpy(sh).to(device=device)
    return means_t, quats_t, scales_t, opacities_t, sh_t


def _infer_sh_degree(sh: torch.Tensor) -> int | None:
    d_sh = sh.shape[1]
    degree = int(math.isqrt(d_sh) - 1)
    if (degree + 1) ** 2 != d_sh:
        return None
    return degree


def main() -> None:
    parser = argparse.ArgumentParser(description="View Gaussian splats from a PLY file.")
    parser.add_argument("ply", type=str, help="Path to a Gaussian PLY file.")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for the viewer server."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gaussian_viewer",
        help="Directory for viewer output assets.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. cuda:0). Defaults to CUDA if available.",
    )
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    ply_path = Path(args.ply)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    means, quats, scales, opacities, shs = _load_gaussians_from_ply(
        ply_path, device
    )
    sh_degree = _infer_sh_degree(shs)

    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
        viewmat = c2w.inverse()

        render_mode_map = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = rasterization(
            means,
            quats,
            scales,
            opacities,
            shs,
            viewmat[None],
            K[None],
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=render_mode_map[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
        )
        render_tab_state.total_gs_count = int(means.shape[0])
        radii = info.get("radii") if isinstance(info, dict) else None
        if radii is not None:
            render_tab_state.rendered_gs_count = (
                (radii > 0).all(-1).sum().item()
            )
        else:
            render_tab_state.rendered_gs_count = 0

        if render_tab_state.render_mode == "rgb":
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = apply_float_colormap(depth_norm, render_tab_state.colormap).cpu()
            renders = renders.numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = apply_float_colormap(alpha, render_tab_state.colormap).cpu()
            renders = renders.numpy()
        else:
            renders = render_colors[0, ..., 0:3].clamp(0, 1).cpu().numpy()
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)
