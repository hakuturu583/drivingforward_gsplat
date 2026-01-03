import argparse
import os
import shutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import yaml

from drivingforward_gsplat.models.drivingforward_model import (
    DrivingForwardModel,
    _NO_DEVICE_KEYS,
)
from drivingforward_gsplat.utils.gaussian_ply import (
    save_gaussians_as_inria_ply,
    save_gaussians_as_ply,
)
from drivingforward_gsplat.models.gaussian import (
    depth2pc,
    focal2fov,
    getProjectionMatrix,
    pts2render,
    rotate_sh,
)
from drivingforward_gsplat.utils.misc import get_config
from drivingforward_gsplat.utils.misc import to_pil_rgb
from einops import rearrange

CAM_ORDER = [
    "CAM_FRONT_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class PredictGaussianConfig:
    model_config: str = "configs/nuscenes/main.yaml"
    split: str = "eval_MF"
    index: int = 0
    torchscript_dir: str = "torchscript"
    novel_view_mode: str = "MF"
    output_path: str = "output/gaussians"
    cpu: bool = False
    sdxl_panorama_i2i_config: Optional[str] = "configs/sdxl_panorama_i2i.yaml"
    sdxl_panorama_prompt_config: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "PredictGaussianConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cpu_value = data.get("cpu", cls.cpu)
        if isinstance(cpu_value, str):
            cpu_value = cpu_value.strip().lower() in ("1", "true", "yes", "y")
        return cls(
            model_config=data.get("model_config", cls.model_config),
            split=data.get("split", cls.split),
            index=int(data.get("index", cls.index)),
            torchscript_dir=data.get("torchscript_dir", cls.torchscript_dir),
            novel_view_mode=data.get("novel_view_mode", cls.novel_view_mode),
            output_path=data.get("output_path", cls.output_path),
            cpu=bool(cpu_value),
            sdxl_panorama_i2i_config=data.get(
                "sdxl_panorama_i2i_config", cls.sdxl_panorama_i2i_config
            ),
            sdxl_panorama_prompt_config=data.get(
                "sdxl_panorama_prompt_config", cls.sdxl_panorama_prompt_config
            ),
        )


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _timestamp_token(token: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token_safe = str(token).replace("/", "_").replace(os.sep, "_")
    return f"{timestamp}_{token_safe}"


def _save_input_images(
    raw_images: Optional[torch.Tensor],
    i2i_images: Optional[torch.Tensor],
    output_root: str,
    output_token: str,
) -> None:
    if raw_images is None and i2i_images is None:
        return
    output_dir = os.path.join(output_root, output_token)
    _ensure_dir(output_dir)
    for cam_name, cam_idx in zip(CAM_ORDER, range(len(CAM_ORDER))):
        if raw_images is not None:
            image = to_pil_rgb(raw_images[cam_idx])
            image.save(os.path.join(output_dir, f"{cam_name}.png"))
        if i2i_images is not None:
            image = to_pil_rgb(i2i_images[cam_idx])
            image.save(os.path.join(output_dir, f"{cam_name}_i2i.png"))


def _ensure_render_params(
    outputs: dict,
    inputs: dict,
    cam: int,
    frame_id: int,
    zfar: float,
    znear: float,
) -> None:
    cam_outputs = outputs[("cam", cam)]
    if ("FovX", frame_id, 0) in cam_outputs:
        return
    bs, _, height, width = inputs[("color", 0, 0)][:, cam, ...].shape
    fovx_list = []
    fovy_list = []
    world_view_transform_list = []
    full_proj_transform_list = []
    camera_center_list = []
    for i in range(bs):
        intr = inputs[("K", 0)][:, cam, ...][i, :]
        extr = inputs["extrinsics_inv"][:, cam, ...][i, :]
        fovx = focal2fov(intr[0, 0].item(), width)
        fovy = focal2fov(intr[1, 1].item(), height)
        projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, K=intr, h=height, w=width
        ).to(device=intr.device, dtype=intr.dtype)
        projection_matrix = projection_matrix.transpose(0, 1)
        world_view_transform = extr.transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0)
            .bmm(projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        camera_center = world_view_transform.inverse()[3, :3]

        fovx_list.append(fovx)
        fovy_list.append(fovy)
        world_view_transform_list.append(world_view_transform.unsqueeze(0))
        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
        camera_center_list.append(camera_center.unsqueeze(0))
    cam_outputs[("FovX", frame_id, 0)] = torch.tensor(
        fovx_list, device=intr.device, dtype=intr.dtype
    )
    cam_outputs[("FovY", frame_id, 0)] = torch.tensor(
        fovy_list, device=intr.device, dtype=intr.dtype
    )
    cam_outputs[("world_view_transform", frame_id, 0)] = torch.cat(
        world_view_transform_list, dim=0
    )
    cam_outputs[("full_proj_transform", frame_id, 0)] = torch.cat(
        full_proj_transform_list, dim=0
    )
    cam_outputs[("camera_center", frame_id, 0)] = torch.cat(camera_center_list, dim=0)


def _save_rendered_images(
    outputs: dict,
    inputs: dict,
    cam_num: int,
    novel_view_mode: str,
    output_root: str,
    output_token: str,
    zfar: float,
) -> None:
    output_dir = os.path.join(output_root, output_token)
    output_dir_path = Path(output_dir)
    _ensure_dir(output_dir)
    frame_id = 0
    rendered_raw_list = []
    fixer_input_dir = output_dir_path / "fixer_input"
    fixer_output_dir = output_dir_path / "fixer_output"
    fixer_input_dir.mkdir(parents=True, exist_ok=True)
    fixer_output_dir.mkdir(parents=True, exist_ok=True)
    for cam_name, cam in zip(CAM_ORDER, range(cam_num)):
        _ensure_render_params(outputs, inputs, cam, frame_id, zfar=zfar, znear=0.01)
        rendered_raw = pts2render(
            inputs=inputs,
            outputs=outputs,
            cam_num=cam_num,
            novel_cam=cam,
            novel_frame_id=frame_id,
            bg_color=[1.0, 1.0, 1.0],
            mode=novel_view_mode,
            with_postprocess=False,
        )
        rendered_raw_list.append(rendered_raw)

    if not rendered_raw_list:
        return

    for cam_name, rendered_raw in zip(CAM_ORDER, rendered_raw_list):
        raw_image = to_pil_rgb(rendered_raw[0])
        raw_image.save(os.path.join(output_dir, f"{cam_name}_render_raw.png"))
        raw_image.save(fixer_input_dir / f"{cam_name}_render_raw.png")

    from fixerpy.fixer import setup_and_infer

    dest_root = Path(os.environ.get("FIXER_WORK_DIR", ".fixer_work"))
    setup_and_infer(
        dest_root=dest_root,
        input_dir=fixer_input_dir,
        output_dir=fixer_output_dir,
        timestep=250,
        batch_size=1,
        use_gpus=True,
        platform=None,
    )

    for cam_name in CAM_ORDER:
        expected_name = f"{cam_name}_render_raw.png"
        output_path = fixer_output_dir / expected_name
        if not output_path.exists():
            stem = Path(expected_name).stem
            matches = [
                p
                for p in fixer_output_dir.glob(f"{stem}.*")
                if p.suffix.lower() in _IMAGE_EXTS
            ]
            if not matches:
                raise FileNotFoundError(
                    f"Fixer output missing for {cam_name} in {fixer_output_dir}"
                )
            output_path = sorted(matches)[0]
        shutil.copyfile(output_path, output_dir_path / f"{cam_name}_render.png")


def _to_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, list):
        return [_to_tensor(item) for item in value]
    return value


def _add_batch_dim(value):
    if torch.is_tensor(value):
        return value.unsqueeze(0)
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, 0)
    if isinstance(value, list) and value:
        if torch.is_tensor(value[0]):
            return [item.unsqueeze(0) for item in value]
        if isinstance(value[0], np.ndarray):
            return [np.expand_dims(item, 0) for item in value]
    return value


def _add_batch_dim_to_inputs(inputs: dict) -> dict:
    for key, value in inputs.items():
        if key in _NO_DEVICE_KEYS:
            continue
        inputs[key] = _add_batch_dim(value)
    return inputs


def _move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    for key, value in inputs.items():
        if key in _NO_DEVICE_KEYS:
            continue
        value = _to_tensor(value)
        if torch.is_tensor(value):
            inputs[key] = value.float().to(device)
        elif isinstance(value, list):
            inputs[key] = [
                item.float().to(device) if torch.is_tensor(item) else item
                for item in value
            ]
        else:
            inputs[key] = value
    return inputs


def _generate_context_gaussians_mf(
    model: DrivingForwardModel, inputs: dict, outputs: dict, cam: int
) -> None:
    bs, _, _, _ = inputs[("color", 0, 0)][:, cam, ...].shape
    for frame_id in (-1, 1):
        if ("color", frame_id, 0) not in inputs:
            raise KeyError(f"Missing frame_id {frame_id} for MF gaussian generation.")
        outputs[("cam", cam)][("e2c_extr", frame_id, 0)] = torch.matmul(
            outputs[("cam", cam)][("cam_T_cam", 0, frame_id)],
            inputs["extrinsics_inv"][:, cam, ...],
        )
        outputs[("cam", cam)][("c2e_extr", frame_id, 0)] = torch.matmul(
            inputs["extrinsics"][:, cam, ...],
            torch.inverse(outputs[("cam", cam)][("cam_T_cam", 0, frame_id)]),
        )
        outputs[("cam", cam)][("xyz", frame_id, 0)] = depth2pc(
            outputs[("cam", cam)][("depth", frame_id, 0)],
            outputs[("cam", cam)][("e2c_extr", frame_id, 0)],
            inputs[("K", 0)][:, cam, ...],
        )
        valid = outputs[("cam", cam)][("depth", frame_id, 0)] != 0.0
        outputs[("cam", cam)][("pts_valid", frame_id, 0)] = valid.view(bs, -1)
        rot_maps, scale_maps, opacity_maps, sh_maps = model.gs_net(
            inputs[("color", frame_id, 0)][:, cam, ...],
            outputs[("cam", cam)][("depth", frame_id, 0)],
            outputs[("cam", cam)][("img_feat", frame_id, 0)],
        )
        c2w_rotations = rearrange(
            outputs[("cam", cam)][("c2e_extr", frame_id, 0)][..., :3, :3],
            "k i j -> k () () () i j",
        )
        sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])
        outputs[("cam", cam)][("rot_maps", frame_id, 0)] = rot_maps
        outputs[("cam", cam)][("scale_maps", frame_id, 0)] = scale_maps
        outputs[("cam", cam)][("opacity_maps", frame_id, 0)] = opacity_maps
        outputs[("cam", cam)][("sh_maps", frame_id, 0)] = sh_maps


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def _apply_sdxl_panorama_i2i(predict_cfg: PredictGaussianConfig, sample: dict) -> dict:
    from PIL import Image
    import torchvision.transforms as transforms

    from drivingforward_gsplat.i2i.sdxl_panorama_i2i import sdxl_panorama_i2i
    from drivingforward_gsplat.i2i.sdxl_panorama_i2i_config import (
        SdxlPanoramaI2IConfig,
    )

    cfg_path = _resolve_path(predict_cfg.sdxl_panorama_i2i_config)
    if cfg_path is None:
        raise ValueError("sdxl_panorama_i2i_config is required.")
    i2i_cfg = SdxlPanoramaI2IConfig.from_yaml(cfg_path)
    if predict_cfg.sdxl_panorama_prompt_config:
        i2i_cfg.prompt_config = predict_cfg.sdxl_panorama_prompt_config
    i2i_cfg.prompt_config = _resolve_path(i2i_cfg.prompt_config)
    if i2i_cfg.prompt_config is None:
        return sample
    if predict_cfg.novel_view_mode == "SF":
        desired_frame_ids = [0]
    else:
        desired_frame_ids = [0, -1, 1]
    frame_ids = [
        frame_id for frame_id in desired_frame_ids if ("color", frame_id, 0) in sample
    ]
    to_tensor = transforms.ToTensor()
    for frame_id in frame_ids:
        target = sample[("color", frame_id, 0)]
        target_height, target_width = target.shape[-2], target.shape[-1]
        images = [target[idx] for idx in range(len(CAM_ORDER))]
        generated_images = sdxl_panorama_i2i(i2i_cfg, images)
        converted = []
        for image in generated_images:
            if image.size != (target_width, target_height):
                image = image.resize(
                    (target_width, target_height), resample=Image.BICUBIC
                )
            converted.append(to_tensor(image).type_as(target))
        if len(converted) != target.shape[0]:
            raise ValueError(
                f"Expected {target.shape[0]} i2i images, got {len(converted)}."
            )
        stacked = torch.stack(converted, dim=0)
        sample[("color", frame_id, 0)] = stacked
        sample[("color_aug", frame_id, 0)] = stacked.clone()
    return sample


class TorchScriptDepthNet(torch.nn.Module):
    def __init__(self, cfg, torchscript_dir, mode, device):
        super().__init__()
        self.num_cams = cfg["data"]["num_cams"]
        self.fusion_level = cfg["model"]["fusion_level"]
        self.mode = mode
        enc_path = os.path.join(torchscript_dir, f"depth_encoder_{mode}.pt")
        dec_path = os.path.join(torchscript_dir, f"depth_decoder_{mode}.pt")
        self.depth_encoder = torch.jit.load(enc_path, map_location=device).eval()
        self.depth_decoder = torch.jit.load(dec_path, map_location=device).eval()

    def _run_one(self, images, mask, k, inv_k, extrinsics, extrinsics_inv):
        (
            feat0,
            feat1,
            proj_feat,
            img_feat0,
            img_feat1,
            img_feat2,
        ) = self.depth_encoder(images, mask, k, inv_k, extrinsics, extrinsics_inv)
        disp = self.depth_decoder(feat0, feat1, proj_feat)
        img_feat = (img_feat0, img_feat1, img_feat2)
        return disp, img_feat

    def forward(self, inputs):
        outputs = {}
        for cam in range(self.num_cams):
            outputs[("cam", cam)] = {}

        images = inputs[("color_aug", 0, 0)]
        mask = inputs["mask"]
        k = inputs[("K", self.fusion_level + 1)]
        inv_k = inputs[("inv_K", self.fusion_level + 1)]
        extrinsics = inputs["extrinsics"]
        extrinsics_inv = inputs["extrinsics_inv"]

        if self.mode == "MF":
            images_last = inputs[("color_aug", -1, 0)]
            images_next = inputs[("color_aug", 1, 0)]
            enc_out = self.depth_encoder(
                images,
                images_last,
                images_next,
                mask,
                k,
                inv_k,
                extrinsics,
                extrinsics_inv,
            )
            feat0, feat1, proj_feat, img_feat0, img_feat1, img_feat2 = enc_out[:6]
            (
                feat0_last,
                feat1_last,
                proj_feat_last,
                img_feat0_last,
                img_feat1_last,
                img_feat2_last,
            ) = enc_out[6:12]
            (
                feat0_next,
                feat1_next,
                proj_feat_next,
                img_feat0_next,
                img_feat1_next,
                img_feat2_next,
            ) = enc_out[12:18]
            disp_cur = self.depth_decoder(feat0, feat1, proj_feat)
            disp_last = self.depth_decoder(feat0_last, feat1_last, proj_feat_last)
            disp_next = self.depth_decoder(feat0_next, feat1_next, proj_feat_next)
            img_feat_cur = (img_feat0, img_feat1, img_feat2)
            img_feat_last = (img_feat0_last, img_feat1_last, img_feat2_last)
            img_feat_next = (img_feat0_next, img_feat1_next, img_feat2_next)
        else:
            disp_cur, img_feat_cur = self._run_one(
                images, mask, k, inv_k, extrinsics, extrinsics_inv
            )

        for cam in range(self.num_cams):
            outputs[("cam", cam)][("disp", 0)] = disp_cur[:, cam, ...]
            outputs[("cam", cam)][("img_feat", 0, 0)] = [
                feat[:, cam, ...] for feat in img_feat_cur
            ]
            if self.mode == "MF":
                outputs[("cam", cam)][("disp", -1, 0)] = disp_last[:, cam, ...]
                outputs[("cam", cam)][("disp", 1, 0)] = disp_next[:, cam, ...]
                outputs[("cam", cam)][("img_feat", -1, 0)] = [
                    feat[:, cam, ...] for feat in img_feat_last
                ]
                outputs[("cam", cam)][("img_feat", 1, 0)] = [
                    feat[:, cam, ...] for feat in img_feat_next
                ]

        return outputs


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


class GtPoseDrivingForwardModel(DrivingForwardModel):
    def __init__(self, cfg, torchscript_dir, device):
        self.torchscript_dir = torchscript_dir
        self.device = device
        super().__init__(cfg, rank=0)

    def prepare_dataset(self, cfg, rank):
        return

    def prepare_model(self, cfg, rank):
        mode = cfg["model"]["novel_view_mode"]
        models = {
            "pose_net": torch.nn.Identity(),
            "depth_net": TorchScriptDepthNet(
                cfg, self.torchscript_dir, mode, self.device
            ),
        }
        if self.gaussian:
            models["gs_net"] = TorchScriptGaussianNet(
                self.torchscript_dir, mode, self.device
            )
        return models

    def load_weights(self):
        if self.rank == 0:
            print("Skipping weight loading for torchscript mode.")

    def predict_pose(self, inputs):
        outputs = {}
        for cam in range(self.num_cams):
            outputs[("cam", cam)] = {}
            for frame_id in self.frame_ids[1:]:
                outputs[("cam", cam)][("cam_T_cam", 0, frame_id)] = inputs[
                    ("cam_T_cam", 0, frame_id)
                ][:, cam, ...]
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict gaussian parameters from NuScenes images."
    )
    parser.add_argument(
        "--predict-config",
        default="configs/predict_gaussian.yaml",
        help="Predict gaussian config yaml file path.",
    )
    parser.add_argument(
        "--sdxl-panorama-prompt-config",
        default=None,
        help="SDXL panorama prompt yaml file path. When set, i2i is applied.",
    )
    args = parser.parse_args()
    predict_cfg = PredictGaussianConfig.from_yaml(args.predict_config)
    if args.sdxl_panorama_prompt_config:
        predict_cfg.sdxl_panorama_prompt_config = args.sdxl_panorama_prompt_config

    from drivingforward_gsplat.dataset import EnvNuScenesDataset, get_transforms

    cfg = get_config(
        predict_cfg.model_config,
        mode="eval",
        novel_view_mode=predict_cfg.novel_view_mode,
    )
    augmentation = {
        "image_shape": (int(cfg["training"]["height"]), int(cfg["training"]["width"])),
        "jittering": (0.0, 0.0, 0.0, 0.0),
        "crop_train_borders": (),
        "crop_eval_borders": (),
    }
    dataset = EnvNuScenesDataset(
        predict_cfg.split,
        cameras=CAM_ORDER,
        back_context=cfg["data"]["back_context"],
        forward_context=cfg["data"]["forward_context"],
        data_transform=get_transforms("train", **augmentation),
        depth_type=None,
        with_pose=True,
        with_ego_pose=True,
        with_mask=True,
    )

    sample = dataset[predict_cfg.index]
    raw_images = sample.get(("color", 0, 0))
    if raw_images is not None:
        raw_images = raw_images.clone()
    if predict_cfg.sdxl_panorama_prompt_config:
        sample = _apply_sdxl_panorama_i2i(predict_cfg, sample)
    if predict_cfg.cpu or not torch.cuda.is_available():
        raise RuntimeError("TorchScript DrivingForward inference requires CUDA.")
    device = torch.device("cuda")

    token = sample.get("token", f"index_{predict_cfg.index}")
    output_token = _timestamp_token(str(token))
    i2i_images = None
    if predict_cfg.sdxl_panorama_prompt_config:
        i2i_images = sample.get(("color", 0, 0))
    _save_input_images(raw_images, i2i_images, "output/images", output_token)
    inputs = _add_batch_dim_to_inputs(sample)
    inputs = _move_inputs_to_device(inputs, device)
    model = GtPoseDrivingForwardModel(cfg, predict_cfg.torchscript_dir, device)
    model.set_eval()
    with torch.no_grad():
        outputs = model.estimate(inputs)
        if getattr(model, "gaussian", False):
            model.gs_net = model.models["gs_net"]
            for cam in range(model.num_cams):
                if model.novel_view_mode == "MF":
                    _generate_context_gaussians_mf(model, inputs, outputs, cam)
                else:
                    model.get_gaussian_data(inputs, outputs, cam)
            _save_rendered_images(
                outputs,
                inputs,
                cam_num=model.num_cams,
                novel_view_mode=model.novel_view_mode,
                output_root="output/images",
                output_token=output_token,
                zfar=model.max_depth,
            )
    output_dir = os.path.join(predict_cfg.output_path, output_token)
    _ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "output.ply")
    output_inria_path = os.path.join(output_dir, "output_inria.ply")
    save_gaussians_as_ply(
        outputs,
        output_path,
        cam_num=model.num_cams,
        mode=model.novel_view_mode,
        sample_idx=0,
    )
    save_gaussians_as_inria_ply(
        outputs,
        output_inria_path,
        cam_num=model.num_cams,
        mode=model.novel_view_mode,
        sample_idx=0,
    )
    print(f"Saved gaussians: {output_path}")
    print(f"Saved gaussians (inria): {output_inria_path}")


if __name__ == "__main__":
    main()
