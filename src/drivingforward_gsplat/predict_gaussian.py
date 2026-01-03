import argparse
import os
from datetime import datetime
from dataclasses import dataclass
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
from drivingforward_gsplat.utils.misc import get_config

CAM_ORDER = [
    "CAM_FRONT_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


@dataclass
class PredictGaussianConfig:
    model_config: str = "configs/nuscenes/main.yaml"
    split: str = "eval_MF"
    index: int = 0
    torchscript_dir: str = "torchscript"
    novel_view_mode: str = "MF"
    output_path: str = "output/gaussians"
    cpu: bool = False

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
        )


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _timestamp_token(token: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token_safe = str(token).replace("/", "_").replace(os.sep, "_")
    return f"{timestamp}_{token_safe}"


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
    args = parser.parse_args()
    predict_cfg = PredictGaussianConfig.from_yaml(args.predict_config)

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
    if predict_cfg.cpu or not torch.cuda.is_available():
        raise RuntimeError("TorchScript DrivingForward inference requires CUDA.")
    device = torch.device("cuda")

    token = sample.get("token", f"index_{predict_cfg.index}")
    inputs = _add_batch_dim_to_inputs(sample)
    inputs = _move_inputs_to_device(inputs, device)
    model = GtPoseDrivingForwardModel(cfg, predict_cfg.torchscript_dir, device)
    model.set_eval()
    with torch.no_grad():
        outputs = model.estimate(inputs)
        if getattr(model, "gaussian", False):
            model.gs_net = model.models["gs_net"]
            for cam in range(model.num_cams):
                model.get_gaussian_data(inputs, outputs, cam)
    output_dir = os.path.join(predict_cfg.output_path, _timestamp_token(str(token)))
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
