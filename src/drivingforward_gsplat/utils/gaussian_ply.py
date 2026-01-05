import os
from typing import Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _extract_gaussians_for_render(
    outputs,
    cam_num: int,
    mode: str,
    sample_idx: int,
    novel_cam: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz_list = []
    rot_list = []
    scale_list = []
    opacity_list = []
    sh_list = []
    device = None

    if mode == "SF":
        frames = [0]
        cams = range(cam_num)
    else:
        frames = [-1, 1]
        cams = [novel_cam]

    for frame_id in frames:
        for cam in cams:
            valid = outputs[("cam", cam)][("pts_valid", frame_id, 0)][sample_idx]
            valid = valid.view(-1)
            if device is None:
                device = valid.device
            if not torch.any(valid):
                continue
            xyz = outputs[("cam", cam)][("xyz", frame_id, 0)][sample_idx]
            rot = (
                outputs[("cam", cam)][("rot_maps", frame_id, 0)][sample_idx]
                .permute(1, 2, 0)
                .reshape(-1, 4)
            )
            scale = (
                outputs[("cam", cam)][("scale_maps", frame_id, 0)][sample_idx]
                .permute(1, 2, 0)
                .reshape(-1, 3)
            )
            opacity = (
                outputs[("cam", cam)][("opacity_maps", frame_id, 0)][sample_idx]
                .permute(1, 2, 0)
                .reshape(-1, 1)
            )
            sh_maps = outputs[("cam", cam)][("sh_maps", frame_id, 0)][sample_idx]
            sh = sh_maps.reshape(-1, sh_maps.shape[-2], sh_maps.shape[-1])

            xyz_list.append(xyz[valid].view(-1, 3))
            rot_list.append(rot[valid].view(-1, 4))
            scale_list.append(scale[valid].view(-1, 3))
            opacity_list.append(opacity[valid].view(-1, 1))
            sh_list.append(sh[valid])

    if not xyz_list:
        empty_device = device or torch.device("cpu")
        empty = torch.empty((0, 3), device=empty_device)
        return empty, empty, empty, empty, empty

    xyz = torch.cat(xyz_list, dim=0)
    rot = torch.cat(rot_list, dim=0)
    scale = torch.cat(scale_list, dim=0)
    opacity = torch.cat(opacity_list, dim=0)
    sh = torch.cat(sh_list, dim=0)
    return xyz, rot, scale, opacity, sh


def _gather_gaussians(
    outputs,
    cam_num: int,
    mode: str,
    sample_idx: int,
) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    cams = range(cam_num) if mode == "MF" else range(cam_num)
    xyz_list = []
    rot_list = []
    scale_list = []
    opacity_list = []
    sh_list = []

    for cam in cams:
        xyz, rot, scale, opacity, sh = _extract_gaussians_for_render(
            outputs, cam_num, mode, sample_idx, cam
        )
        if xyz.numel() == 0:
            continue
        xyz_list.append(xyz)
        rot_list.append(rot)
        scale_list.append(scale)
        opacity_list.append(opacity)
        sh_list.append(sh)

    if not xyz_list:
        return None

    xyz = torch.cat(xyz_list, dim=0)
    rot = torch.cat(rot_list, dim=0)
    scale = torch.cat(scale_list, dim=0)
    opacity = torch.cat(opacity_list, dim=0)
    sh = torch.cat(sh_list, dim=0)
    return xyz, rot, scale, opacity, sh


def save_gaussians_as_ply(
    outputs,
    output_path: str,
    cam_num: int,
    mode: str,
    sample_idx: int = 0,
) -> Optional[str]:
    gathered = _gather_gaussians(outputs, cam_num, mode, sample_idx)
    if gathered is None:
        return None
    xyz, rot, scale, opacity, sh = gathered
    sh = sh.permute(0, 2, 1).reshape(sh.shape[0], -1)

    xyz_np = xyz.detach().cpu().numpy().astype(np.float32)
    rot_np = rot.detach().cpu().numpy().astype(np.float32)
    scale_np = scale.detach().cpu().numpy().astype(np.float32)
    opacity_np = opacity.detach().cpu().numpy().astype(np.float32)
    sh_np = sh.detach().cpu().numpy().astype(np.float32)

    num = xyz_np.shape[0]
    sh_dim = sh_np.shape[1]
    dtype_fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("opacity", "f4"),
    ] + [(f"sh_{idx}", "f4") for idx in range(sh_dim)]

    data = np.zeros(num, dtype=dtype_fields)
    data["x"] = xyz_np[:, 0]
    data["y"] = xyz_np[:, 1]
    data["z"] = xyz_np[:, 2]
    data["rot_0"] = rot_np[:, 0]
    data["rot_1"] = rot_np[:, 1]
    data["rot_2"] = rot_np[:, 2]
    data["rot_3"] = rot_np[:, 3]
    data["scale_0"] = scale_np[:, 0]
    data["scale_1"] = scale_np[:, 1]
    data["scale_2"] = scale_np[:, 2]
    data["opacity"] = opacity_np[:, 0]
    for idx in range(sh_dim):
        data[f"sh_{idx}"] = sh_np[:, idx]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(output_path)
    return output_path


def save_gaussians_as_inria_ply(
    outputs,
    output_path: str,
    cam_num: int,
    mode: str,
    sample_idx: int = 0,
) -> Optional[str]:
    gathered = _gather_gaussians(outputs, cam_num, mode, sample_idx)
    if gathered is None:
        return None
    xyz, rot, scale, opacity, sh = gathered

    scale = torch.clamp(scale, min=1e-8)
    opacity = torch.clamp(opacity, min=1e-6, max=1.0 - 1e-6)
    scale_log = torch.log(scale)
    opacity_logit = torch.log(opacity / (1.0 - opacity))

    xyz_np = xyz.detach().cpu().numpy().astype(np.float32)
    rot_np = rot.detach().cpu().numpy().astype(np.float32)
    scale_np = scale_log.detach().cpu().numpy().astype(np.float32)
    opacity_np = opacity_logit.detach().cpu().numpy().astype(np.float32)
    sh_np = sh.detach().cpu().numpy().astype(np.float32)

    num = xyz_np.shape[0]
    d_sh = sh_np.shape[2]
    rest_dim = max(d_sh - 1, 0)
    dtype_fields = (
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
        ]
        + [(f"f_rest_{idx}", "f4") for idx in range(rest_dim * 3)]
        + [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )

    data = np.zeros(num, dtype=dtype_fields)
    data["x"] = xyz_np[:, 0]
    data["y"] = xyz_np[:, 1]
    data["z"] = xyz_np[:, 2]
    data["nx"] = 0.0
    data["ny"] = 0.0
    data["nz"] = 0.0

    dc = sh_np[:, :, 0]
    data["f_dc_0"] = dc[:, 0]
    data["f_dc_1"] = dc[:, 1]
    data["f_dc_2"] = dc[:, 2]

    if rest_dim:
        rest = sh_np[:, :, 1:].reshape(num, -1)
        for idx in range(rest.shape[1]):
            data[f"f_rest_{idx}"] = rest[:, idx]

    data["opacity"] = opacity_np[:, 0]
    data["scale_0"] = scale_np[:, 0]
    data["scale_1"] = scale_np[:, 1]
    data["scale_2"] = scale_np[:, 2]
    data["rot_0"] = rot_np[:, 0]
    data["rot_1"] = rot_np[:, 1]
    data["rot_2"] = rot_np[:, 2]
    data["rot_3"] = rot_np[:, 3]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(output_path)
    return output_path


def save_gaussians_tensors_as_inria_ply(
    xyz: torch.Tensor,
    rot: torch.Tensor,
    scale: torch.Tensor,
    opacity: torch.Tensor,
    sh: torch.Tensor,
    output_path: str,
) -> str:
    scale = torch.clamp(scale, min=1e-8)
    opacity = torch.clamp(opacity, min=1e-6, max=1.0 - 1e-6)
    scale_log = torch.log(scale)
    opacity_logit = torch.log(opacity / (1.0 - opacity))

    xyz_np = xyz.detach().cpu().numpy().astype(np.float32)
    rot_np = rot.detach().cpu().numpy().astype(np.float32)
    scale_np = scale_log.detach().cpu().numpy().astype(np.float32)
    opacity_np = opacity_logit.detach().cpu().numpy().astype(np.float32)
    sh_np = sh.detach().cpu().numpy().astype(np.float32)

    num = xyz_np.shape[0]
    d_sh = sh_np.shape[1]
    rest_dim = max(d_sh - 1, 0)
    dtype_fields = (
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
        ]
        + [(f"f_dc_{i}", "f4") for i in range(3)]
        + [(f"f_rest_{i}", "f4") for i in range(rest_dim * 3)]
        + [("opacity", "f4")]
        + [(f"scale_{i}", "f4") for i in range(3)]
        + [(f"rot_{i}", "f4") for i in range(4)]
    )

    data = np.zeros(num, dtype=dtype_fields)
    data["x"] = xyz_np[:, 0]
    data["y"] = xyz_np[:, 1]
    data["z"] = xyz_np[:, 2]
    data["nx"] = 0.0
    data["ny"] = 0.0
    data["nz"] = 0.0

    data["f_dc_0"] = sh_np[:, 0, 0]
    data["f_dc_1"] = sh_np[:, 0, 1]
    data["f_dc_2"] = sh_np[:, 0, 2]

    if rest_dim > 0:
        sh_rest = sh_np[:, 1:, :].transpose(0, 2, 1).reshape(num, -1)
        for idx in range(rest_dim * 3):
            data[f"f_rest_{idx}"] = sh_rest[:, idx]

    data["opacity"] = opacity_np[:, 0] if opacity_np.ndim > 1 else opacity_np
    data["scale_0"] = scale_np[:, 0]
    data["scale_1"] = scale_np[:, 1]
    data["scale_2"] = scale_np[:, 2]
    data["rot_0"] = rot_np[:, 0]
    data["rot_1"] = rot_np[:, 1]
    data["rot_2"] = rot_np[:, 2]
    data["rot_3"] = rot_np[:, 3]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(output_path)
    return output_path
