import argparse
import os
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from drivingforward_gsplat.depth.depth import (
    dense_depth_from_anything,
    normalize_depths,
    resize_depths_to_match,
)
from drivingforward_gsplat.network.blocks import pack_cam_feat, unpack_cam_feat
from drivingforward_gsplat.models.gaussian.utils import depth2pc
from drivingforward_gsplat.utils.gaussian_ply import (
    save_gaussians_as_inria_ply,
    save_gaussians_as_ply,
)
from drivingforward_gsplat.utils.misc import ImageLike, get_config, to_pil_rgb

CamImages = Union[Sequence[ImageLike], Mapping[str, ImageLike]]

CAM_ORDER = [
    "CAM_FRONT_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def _ensure_cam_order(images: CamImages, cam_order: Sequence[str]) -> List[ImageLike]:
    if isinstance(images, Mapping):
        missing = [name for name in cam_order if name not in images]
        if missing:
            raise ValueError(f"Missing camera images for: {missing}")
        return [images[name] for name in cam_order]
    if len(images) != len(cam_order):
        raise ValueError(
            f"Expected {len(cam_order)} images, but got {len(images)} instead."
        )
    return list(images)


def _resize_to_first(images: Iterable[Image.Image]) -> List[Image.Image]:
    images = list(images)
    if not images:
        return []
    target_size = images[0].size
    return [
        img if img.size == target_size else img.resize(target_size) for img in images
    ]


def _pil_to_chw_tensor(image: Image.Image, channels: int) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32)
    if channels == 1:
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr[None, ...]
        arr = arr / 255.0
        return torch.from_numpy(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    arr = arr.transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


def _extract_depth_encoder(depth_encoder: torch.nn.Module) -> torch.nn.Module:
    return depth_encoder.encoder if hasattr(depth_encoder, "encoder") else depth_encoder


def _ensure_feature_list(
    feats: Union[Sequence[torch.Tensor], torch.Tensor]
) -> List[torch.Tensor]:
    if isinstance(feats, torch.Tensor):
        return [feats]
    return list(feats)


def _select_matching_features(
    feats: Sequence[torch.Tensor], depth_feats: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matches: List[torch.Tensor] = []
    for depth_feat in depth_feats:
        target_hw = depth_feat.shape[-2:]
        candidates = [feat for feat in feats if feat.shape[-2:] == target_hw]
        if not candidates:
            raise RuntimeError(
                f"No image features match depth feature size {target_hw}."
            )
        matches.append(candidates[0])
    if len(matches) != 3:
        raise RuntimeError(
            f"Expected 3 matched image features, got {len(matches)} instead."
        )
    return matches[0], matches[1], matches[2]


@torch.no_grad()
def predict_gaussians_from_images(
    images: CamImages,
    depth_encoder: torch.nn.Module,
    gaussian_net: torch.nn.Module,
    depth_device: torch.device,
    depth_model_id: str,
    intrinsics: Sequence[np.ndarray] | Mapping[str, np.ndarray] | None = None,
    cam_order: Sequence[str] = CAM_ORDER,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[Image.Image]]:
    """
    Predict gaussian parameters from 6 camera images in cam_order.

    Returns a dict keyed by camera name and a list of normalized depth images.
    """
    ordered_images = _ensure_cam_order(images, cam_order)
    pil_images = [to_pil_rgb(img) for img in ordered_images]
    pil_images = _resize_to_first(pil_images)

    intrinsics_list = None
    if intrinsics is not None:
        intrinsics_list = _ensure_cam_order(intrinsics, cam_order)
        intrinsics_list = [np.asarray(k) for k in intrinsics_list]
    depths = dense_depth_from_anything(
        pil_images,
        depth_device,
        depth_model_id,
        intrinsics=intrinsics_list,
    )
    depth_pils = normalize_depths(depths)
    depth_pils = resize_depths_to_match(pil_images, depth_pils)

    image_tensors = [_pil_to_chw_tensor(img, channels=3) for img in pil_images]
    depth_tensors = [_pil_to_chw_tensor(img, channels=1) for img in depth_pils]

    image_batch = torch.stack(image_tensors, dim=0).unsqueeze(0)
    depth_batch = torch.stack(depth_tensors, dim=0).unsqueeze(0)

    device = next(gaussian_net.parameters()).device
    image_batch = image_batch.to(device=device)
    depth_batch = depth_batch.to(device=device)

    packed_images = pack_cam_feat(image_batch)
    packed_depths = pack_cam_feat(depth_batch)

    encoder = _extract_depth_encoder(depth_encoder)
    encoder = encoder.to(device=device).eval()
    gaussian_net = gaussian_net.to(device=device).eval()

    feats = _ensure_feature_list(encoder(packed_images))

    depth_encoder_module = None
    if hasattr(gaussian_net, "gaussian_encoder"):
        depth_encoder_module = gaussian_net.gaussian_encoder
    elif hasattr(gaussian_net, "depth_encoder"):
        depth_encoder_module = gaussian_net.depth_encoder

    if depth_encoder_module is not None:
        depth_feats = _ensure_feature_list(depth_encoder_module(packed_depths))
        img_feat = _select_matching_features(feats, depth_feats)
    else:
        if len(feats) < 3:
            raise RuntimeError("Not enough image features to feed gaussian network.")
        img_feat = (feats[0], feats[1], feats[2])

    rot, scale, opacity, sh = gaussian_net(packed_images, packed_depths, img_feat)

    bsz = image_batch.shape[0]
    num_cams = image_batch.shape[1]
    outputs: Dict[str, Dict[str, torch.Tensor]] = {}
    rot = unpack_cam_feat(rot, bsz, num_cams)
    scale = unpack_cam_feat(scale, bsz, num_cams)
    opacity = unpack_cam_feat(opacity, bsz, num_cams)
    sh = sh.view(bsz, num_cams, *sh.shape[1:])
    depth_out = unpack_cam_feat(packed_depths, bsz, num_cams)

    for idx, cam in enumerate(cam_order):
        outputs[cam] = {
            "rot": rot[:, idx, ...],
            "scale": scale[:, idx, ...],
            "opacity": opacity[:, idx, ...],
            "sh": sh[:, idx, ...],
            "depth": depth_out[:, idx, ...],
        }

    return outputs, depth_pils


class TorchScriptGaussianNet(torch.nn.Module):
    def __init__(self, torchscript_dir: str, mode: str, device: torch.device):
        super().__init__()
        enc_path = os.path.join(torchscript_dir, f"gaussian_encoder_{mode}.pt")
        dec_path = os.path.join(torchscript_dir, f"gaussian_decoder_{mode}.pt")
        self.gaussian_encoder = torch.jit.load(enc_path, map_location=device).eval()
        self.gaussian_decoder = torch.jit.load(dec_path, map_location=device).eval()

    def forward(self, img, depth, img_feat):
        depth_feat1, depth_feat2, depth_feat3 = self.gaussian_encoder(depth)
        return self.gaussian_decoder(
            img,
            depth,
            img_feat[0],
            img_feat[1],
            img_feat[2],
            depth_feat1,
            depth_feat2,
            depth_feat3,
        )


def _stacked_images_to_list(images: torch.Tensor) -> List[torch.Tensor]:
    if images.ndim != 4:
        raise ValueError(f"Expected stacked images as (N, C, H, W), got {images.shape}")
    return [images[idx] for idx in range(images.shape[0])]


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _timestamp_token(token: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token_safe = str(token).replace("/", "_").replace(os.sep, "_")
    return f"{timestamp}_{token_safe}"


def _gaussians_to_outputs(
    sample: Dict,
    gaussians: Dict[str, Dict[str, torch.Tensor]],
    cam_order: Sequence[str],
) -> Dict[Tuple[str, int], Dict[Tuple[str, int, int], torch.Tensor]]:
    outputs: Dict[Tuple[str, int], Dict[Tuple[str, int, int], torch.Tensor]] = {}
    intrinsics = sample[("K", 0)]
    extrinsics = sample["extrinsics"]
    if isinstance(intrinsics, torch.Tensor):
        intrinsics_np = intrinsics.detach().cpu().numpy()
    else:
        intrinsics_np = intrinsics
    if isinstance(extrinsics, torch.Tensor):
        extrinsics_np = extrinsics.detach().cpu().numpy()
    else:
        extrinsics_np = extrinsics
    for cam_idx, cam in enumerate(cam_order):
        cam_out: Dict[Tuple[str, int, int], torch.Tensor] = {}
        cam_gauss = gaussians[cam]
        depth = cam_gauss["depth"]
        device = depth.device
        intr = torch.from_numpy(intrinsics_np[cam_idx]).to(
            device=device, dtype=depth.dtype
        )
        extr = torch.from_numpy(extrinsics_np[cam_idx]).to(
            device=device, dtype=depth.dtype
        )
        if intr.ndim == 2:
            intr = intr.unsqueeze(0)
        if extr.ndim == 2:
            extr = extr.unsqueeze(0)
        extr_inv = torch.inverse(extr)
        xyz = depth2pc(depth, extr_inv, intr)
        pts_valid = depth.view(depth.shape[0], -1) != 0.0

        cam_out[("xyz", 0, 0)] = xyz
        cam_out[("pts_valid", 0, 0)] = pts_valid
        cam_out[("rot_maps", 0, 0)] = cam_gauss["rot"]
        cam_out[("scale_maps", 0, 0)] = cam_gauss["scale"]
        cam_out[("opacity_maps", 0, 0)] = cam_gauss["opacity"]
        cam_out[("sh_maps", 0, 0)] = cam_gauss["sh"]
        outputs[("cam", cam_idx)] = cam_out
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict gaussian parameters from NuScenes images."
    )
    parser.add_argument(
        "--config",
        default="configs/nuscenes/main.yaml",
        help="Config yaml file path.",
    )
    parser.add_argument(
        "--split",
        default="eval_MF",
        help="NuScenes split name (train/eval_MF/eval_SF).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index to load.",
    )
    parser.add_argument(
        "--torchscript-dir",
        default="torchscript",
        help="Torchscript directory (relative or absolute).",
    )
    parser.add_argument(
        "--depth-encoder-path",
        default=None,
        help="Optional path to a depth encoder torchscript module.",
    )
    parser.add_argument(
        "--novel-view-mode",
        default="MF",
        choices=("MF", "SF"),
        help="Model variant for torchscript files.",
    )
    parser.add_argument(
        "--depth-model-id",
        default="depth-anything/DA3METRIC-LARGE",
        help="Depth Anything 3 model id.",
    )
    parser.add_argument(
        "--output-path",
        default="output/gaussians",
        help="Output directory for gaussian ply files.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    args = parser.parse_args()

    from drivingforward_gsplat.dataset import EnvNuScenesDataset, get_transforms

    cfg = get_config(args.config, mode="eval", novel_view_mode=args.novel_view_mode)
    augmentation = {
        "image_shape": (int(cfg["training"]["height"]), int(cfg["training"]["width"])),
        "jittering": (0.0, 0.0, 0.0, 0.0),
        "crop_train_borders": (),
        "crop_eval_borders": (),
    }
    dataset = EnvNuScenesDataset(
        args.split,
        cameras=CAM_ORDER,
        back_context=cfg["data"]["back_context"],
        forward_context=cfg["data"]["forward_context"],
        data_transform=get_transforms("train", **augmentation),
        depth_type=None,
        with_pose=True,
        with_ego_pose=True,
        with_mask=True,
    )

    sample = dataset[args.index]
    images = _stacked_images_to_list(sample[("color", 0, 0)])
    intrinsics = [k[:3, :3] for k in sample[("K", 0)]]

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    depth_device = device

    depth_encoder_path = args.depth_encoder_path or os.path.join(
        args.torchscript_dir, f"depth_encoder_{args.novel_view_mode}.pt"
    )
    depth_encoder = torch.jit.load(depth_encoder_path, map_location=device).eval()
    gaussian_net = TorchScriptGaussianNet(
        args.torchscript_dir, args.novel_view_mode, device
    )

    gaussians, _ = predict_gaussians_from_images(
        images=images,
        depth_encoder=depth_encoder,
        gaussian_net=gaussian_net,
        depth_device=depth_device,
        depth_model_id=args.depth_model_id,
        cam_order=CAM_ORDER,
        intrinsics=intrinsics,
    )

    token = sample.get("token", f"index_{args.index}")
    outputs = _gaussians_to_outputs(sample, gaussians, CAM_ORDER)
    output_dir = os.path.join(args.output_path, _timestamp_token(str(token)))
    _ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "output.ply")
    output_inria_path = os.path.join(output_dir, "output_inria.ply")
    save_gaussians_as_ply(
        outputs,
        output_path,
        cam_num=len(CAM_ORDER),
        mode="SF",
        sample_idx=0,
    )
    save_gaussians_as_inria_ply(
        outputs,
        output_inria_path,
        cam_num=len(CAM_ORDER),
        mode="SF",
        sample_idx=0,
    )
    print(f"Saved gaussians: {output_path}")
    print(f"Saved gaussians (inria): {output_inria_path}")


if __name__ == "__main__":
    main()
