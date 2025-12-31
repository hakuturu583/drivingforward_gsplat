import inspect
import locale
import math
import os

import torch
from einops import rearrange

left_cam_dict = {2: 0, 0: 1, 4: 2, 1: 3, 5: 4, 3: 5}
right_cam_dict = {0: 2, 1: 0, 2: 4, 3: 1, 4: 5, 5: 3}


def _format_shs_for_gsplat(shs):
    if shs is None:
        return None
    if shs.dim() != 3:
        return shs
    # Legacy gsplat conventions expect (N, 3, d_sh).
    if shs.shape[-1] == 3:
        return shs.transpose(1, 2).contiguous()
    return shs


def _call_gsplat(fn, values):
    signature = inspect.signature(fn)
    kwargs = {}

    aliases = {
        "means3D": "means3D",
        "means3d": "means3D",
        "means": "means3D",
        "xyz": "means3D",
        "scales": "scales",
        "scale": "scales",
        "rotations": "rotations",
        "quats": "rotations",
        "opacities": "opacities",
        "opacity": "opacities",
        "shs": "shs",
        "sh": "shs",
        "viewmats": "viewmats",
        "viewmat": "viewmats",
        "projmats": "projmats",
        "projmat": "projmats",
        "image_height": "image_height",
        "image_width": "image_width",
        "height": "image_height",
        "width": "image_width",
        "H": "image_height",
        "W": "image_width",
        "tanfovx": "tanfovx",
        "tanfovy": "tanfovy",
        "bg": "bg",
        "background": "bg",
        "sh_degree": "sh_degree",
    }

    for name, param in signature.parameters.items():
        key = aliases.get(name, name)
        if key in values:
            kwargs[name] = values[key]
        elif param.default is inspect._empty:
            raise TypeError(f"Missing required argument '{name}' for gsplat render.")

    return fn(**kwargs)


def _ensure_utf8_locale():
    # Best-effort fix for gsplat JIT reading source files under ASCII locale.
    if os.environ.get("PYTHONUTF8") != "1":
        os.environ["PYTHONUTF8"] = "1"
    for key in ("LC_ALL", "LANG"):
        if os.environ.get(key, "") in ("", "C", "POSIX"):
            os.environ[key] = "C.UTF-8"
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, "C.UTF-8")
        except locale.Error:
            pass


def render(
    novel_FovX,
    novel_FovY,
    novel_height,
    novel_width,
    novel_world_view_transform,
    novel_full_proj_transform,
    novel_camera_center,
    pts_xyz,
    pts_rgb,
    rotations,
    scales,
    opacity,
    shs,
    bg_color,
):
    """Render with gsplat. Background tensor must be on the same device."""
    _ensure_utf8_locale()
    try:
        from gsplat import rendering
    except ImportError as exc:
        msg = str(exc)
        if "gsplat.csrc" in msg or "csrc" in msg:
            raise ImportError(
                "gsplat CUDA kernels are missing. Reinstall gsplat from source to build CUDA."
            ) from exc
        raise ImportError("gsplat is required for Gaussian rendering.") from exc

    device = pts_xyz.device
    dtype = pts_xyz.dtype
    bg = torch.tensor(bg_color, dtype=dtype, device=device)

    opacities = opacity.squeeze(-1) if opacity is not None else None
    colors = shs if shs is not None else pts_rgb
    sh_degree = None
    if colors is not None and colors.dim() == 3 and colors.shape[-1] == 3:
        sh_degree = int(math.isqrt(colors.shape[-2])) - 1

    fovx = torch.as_tensor(novel_FovX, device=device, dtype=dtype)
    fovy = torch.as_tensor(novel_FovY, device=device, dtype=dtype)
    width_t = torch.tensor(float(novel_width), device=device, dtype=dtype)
    height_t = torch.tensor(float(novel_height), device=device, dtype=dtype)
    fx = width_t / (2.0 * torch.tan(fovx * 0.5))
    fy = height_t / (2.0 * torch.tan(fovy * 0.5))
    cx = width_t * 0.5
    cy = height_t * 0.5
    zeros = torch.zeros((), device=device, dtype=dtype)
    ones = torch.ones((), device=device, dtype=dtype)
    Ks = torch.stack(
        [
            torch.stack([fx, zeros, cx]),
            torch.stack([zeros, fy, cy]),
            torch.stack([zeros, zeros, ones]),
        ]
    ).unsqueeze(0)
    viewmats = novel_world_view_transform.to(device=device, dtype=dtype).unsqueeze(0)
    backgrounds = bg.unsqueeze(0)

    if hasattr(rendering, "rasterization"):

        rendered = rendering.rasterization(
            means=pts_xyz,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=int(novel_width),
            height=int(novel_height),
            backgrounds=backgrounds,
            sh_degree=sh_degree,
        )
        rendered = rendered[0] if isinstance(rendered, (tuple, list)) else rendered
    elif hasattr(rendering, "rasterization_inria_wrapper"):
        values = {
            "means": pts_xyz,
            "quats": rotations,
            "scales": scales,
            "opacities": opacities,
            "colors": shs if shs is not None else pts_rgb,
            "viewmats": viewmats,
            "Ks": Ks,
            "width": int(novel_width),
            "height": int(novel_height),
            "backgrounds": backgrounds,
            "sh_degree": sh_degree,
        }
        rendered = rendering.rasterization_inria_wrapper(**values)
        rendered = rendered[0] if isinstance(rendered, (tuple, list)) else rendered
    else:
        raise AttributeError("gsplat.rendering does not expose a supported rasterizer.")

    if rendered.dim() == 3 and rendered.shape[-1] in (1, 3):
        rendered = rendered.permute(2, 0, 1)
    elif rendered.dim() == 4 and rendered.shape[-1] in (1, 3):
        rendered = rendered.permute(0, 3, 1, 2)
    return rendered


def get_adj_cams(cam):
    adj_cams = [cam]
    adj_cams.append(left_cam_dict[cam])
    adj_cams.append(right_cam_dict[cam])
    return adj_cams


def pts2render(inputs, outputs, cam_num, novel_cam, novel_frame_id, bg_color, mode="MF"):
    bs, _, height, width = inputs[("color", 0, 0)][:, novel_cam, ...].shape
    render_novel_list = []
    for i in range(bs):
        xyz_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        sh_i_valid = []
        if mode == "SF":
            frame_id = 0
            for cam in range(cam_num):
                valid_i = outputs[("cam", cam)][("pts_valid", frame_id, 0)][i, :]
                xyz_i = outputs[("cam", cam)][("xyz", frame_id, 0)][i, :, :]

                rot_i = (
                    outputs[("cam", cam)][("rot_maps", frame_id, 0)][i, :, :, :]
                    .permute(1, 2, 0)
                    .view(-1, 4)
                )
                scale_i = (
                    outputs[("cam", cam)][("scale_maps", frame_id, 0)][i, :, :, :]
                    .permute(1, 2, 0)
                    .view(-1, 3)
                )
                opacity_i = (
                    outputs[("cam", cam)][("opacity_maps", frame_id, 0)][i, :, :, :]
                    .permute(1, 2, 0)
                    .view(-1, 1)
                )
                sh_i = rearrange(
                    outputs[("cam", cam)][("sh_maps", frame_id, 0)][i, :, :, :],
                    "p srf r xyz d_sh -> (p srf r) d_sh xyz",
                ).contiguous()

                xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
                rot_i_valid.append(rot_i[valid_i].view(-1, 4))
                scale_i_valid.append(scale_i[valid_i].view(-1, 3))
                opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))
                sh_i_valid.append(sh_i[valid_i])

        elif mode == "MF":
            for frame_id in [-1, 1]:
                cam = novel_cam
                valid_i = outputs[("cam", cam)][("pts_valid", frame_id, 0)][i, :]
                xyz_i = outputs[("cam", cam)][("xyz", frame_id, 0)][i, :, :]

                rot_i = (
                    outputs[("cam", cam)][("rot_maps", frame_id, 0)][i, :, :, :]
                    .permute(1, 2, 0)
                    .view(-1, 4)
                )
                scale_i = (
                    outputs[("cam", cam)][("scale_maps", frame_id, 0)][i, :, :, :]
                    .permute(1, 2, 0)
                    .view(-1, 3)
                )
                opacity_i = (
                    outputs[("cam", cam)][("opacity_maps", frame_id, 0)][i, :, :, :]
                    .permute(1, 2, 0)
                    .view(-1, 1)
                )
                sh_i = rearrange(
                    outputs[("cam", cam)][("sh_maps", frame_id, 0)][i, :, :, :],
                    "p srf r xyz d_sh -> (p srf r) d_sh xyz",
                ).contiguous()

                xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
                rot_i_valid.append(rot_i[valid_i].view(-1, 4))
                scale_i_valid.append(scale_i[valid_i].view(-1, 3))
                opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))
                sh_i_valid.append(sh_i[valid_i])

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)
        sh_i = torch.concat(sh_i_valid, dim=0)

        novel_FovX_i = outputs[("cam", novel_cam)][("FovX", novel_frame_id, 0)][i]
        novel_FovY_i = outputs[("cam", novel_cam)][("FovY", novel_frame_id, 0)][i]
        novel_world_view_transform_i = outputs[("cam", novel_cam)][
            ("world_view_transform", novel_frame_id, 0)
        ][i]
        novel_function_proj_transform_i = outputs[("cam", novel_cam)][
            ("full_proj_transform", novel_frame_id, 0)
        ][i]
        novel_camera_center_i = outputs[("cam", novel_cam)][
            ("camera_center", novel_frame_id, 0)
        ][i]

        render_novel_i = render(
            novel_FovX=novel_FovX_i,
            novel_FovY=novel_FovY_i,
            novel_height=height,
            novel_width=width,
            novel_world_view_transform=novel_world_view_transform_i,
            novel_full_proj_transform=novel_function_proj_transform_i,
            novel_camera_center=novel_camera_center_i,
            pts_xyz=pts_xyz_i,
            pts_rgb=None,
            rotations=rot_i,
            scales=scale_i,
            opacity=opacity_i,
            shs=sh_i,
            bg_color=bg_color,
        )
        render_novel_list.append(render_novel_i.unsqueeze(0))

    novel = torch.concat(render_novel_list, dim=0)

    return novel
