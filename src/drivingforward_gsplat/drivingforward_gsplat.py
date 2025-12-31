import argparse
import locale
import os
import sys
import shutil

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset

if os.environ.get("DRIVINGFORWARD_UTF8_REEXEC") != "1":
    encoding = locale.getpreferredencoding(False).lower()
    if sys.flags.utf8_mode == 0 and encoding == "ascii":
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        if env.get("LANG", "") in ("", "C", "POSIX"):
            env["LANG"] = "C.UTF-8"
        if env.get("LC_ALL", "") in ("", "C", "POSIX"):
            env["LC_ALL"] = "C.UTF-8"
        env["DRIVINGFORWARD_UTF8_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

_TORCHSCRIPT_FILES = (
    "depth_decoder_MF.pt",
    "depth_decoder_SF.pt",
    "depth_encoder_MF.pt",
    "depth_encoder_SF.pt",
    "gaussian_decoder_MF.pt",
    "gaussian_decoder_SF.pt",
    "gaussian_encoder_MF.pt",
    "gaussian_encoder_SF.pt",
    "pose_net_MF.pt",
    "pose_net_SF.pt",
)


def parse_args():
    parser = argparse.ArgumentParser(description="TorchScript inference (NuScenes)")
    parser.add_argument(
        "--drivingforward_root",
        default="/home/masaya/workspace/DrivingForward",
        type=str,
        help="Path to the DrivingForward repository",
    )
    parser.add_argument(
        "--config_file",
        default="configs/nuscenes/main.yaml",
        type=str,
        help="Config yaml file path (relative to this repo)",
    )
    parser.add_argument(
        "--weight_path",
        default="weights",
        type=str,
        help="Weight path (kept for config compatibility)",
    )
    parser.add_argument(
        "--novel_view_mode",
        default="MF",
        type=str,
        help="MF or SF",
    )
    parser.add_argument(
        "--torchscript_dir",
        default="torchscript",
        type=str,
        help="Directory for torchscript modules (relative to this repo)",
    )
    parser.add_argument(
        "--torchscript_repo",
        default="hakuturu583/DrivingForward",
        type=str,
        help="Hugging Face repo with torchscript modules",
    )
    return parser.parse_args()


def resolve_path(base, path):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(base, path)


def load_nuscenes_root():
    load_dotenv()
    data_root = os.getenv("NUSCENES_DATA_ROOT")
    if not data_root:
        raise ValueError(
            "Missing NUSCENES_DATA_ROOT. Set it via .env to /mnt/sata_ssd/nuscenes_full/v1.0"
        )
    return data_root


def ensure_torchscript_modules(torchscript_dir, repo_id):
    missing = [
        name
        for name in _TORCHSCRIPT_FILES
        if not os.path.isfile(os.path.join(torchscript_dir, name))
    ]
    if not missing:
        return
    os.makedirs(torchscript_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=torchscript_dir,
        local_dir_use_symlinks=False,
        allow_patterns=list(_TORCHSCRIPT_FILES),
    )


def build_inference(cfg, args, torchscript_dir, drivingforward_root):
    from drivingforward_gsplat.dataset import NuScenesdataset, get_transforms
    from drivingforward_gsplat.models import DrivingForwardModel
    from drivingforward_gsplat.trainer import DrivingForwardTrainer

    class EnvNuScenesDataset(Dataset):
        def __init__(
            self,
            split,
            cameras=None,
            back_context=0,
            forward_context=0,
            data_transform=None,
            depth_type=None,
            scale_range=2,
            with_pose=None,
            with_ego_pose=None,
            with_mask=None,
        ):
            load_dotenv()
            data_root = os.getenv("NUSCENES_DATA_ROOT")
            if not data_root:
                raise ValueError(
                    "Missing NUSCENES_DATA_ROOT. Set it via .env to /mnt/sata_ssd/nuscenes_full/v1.0"
                )
            self._dataset = NuScenesdataset(
                data_root,
                split,
                cameras=cameras,
                back_context=back_context,
                forward_context=forward_context,
                data_transform=data_transform,
                depth_type=depth_type,
                scale_range=scale_range,
                with_pose=with_pose,
                with_ego_pose=with_ego_pose,
                with_mask=with_mask,
            )

        def __len__(self):
            return len(self._dataset)

        def __getitem__(self, idx):
            return self._dataset[idx]

    def construct_env_dataset(cfg, mode, **kwargs):
        if mode == "train":
            dataset_args = {
                "cameras": cfg["data"]["cameras"],
                "back_context": cfg["data"]["back_context"],
                "forward_context": cfg["data"]["forward_context"],
                "data_transform": get_transforms("train", **kwargs),
                "depth_type": cfg["data"]["depth_type"]
                if "gt_depth" in cfg["data"]["train_requirements"]
                else None,
                "with_pose": "gt_pose" in cfg["data"]["train_requirements"],
                "with_ego_pose": "gt_ego_pose" in cfg["data"]["train_requirements"],
                "with_mask": "mask" in cfg["data"]["train_requirements"],
            }
        elif mode in ("val", "eval"):
            dataset_args = {
                "cameras": cfg["data"]["cameras"],
                "back_context": cfg["data"]["back_context"],
                "forward_context": cfg["data"]["forward_context"],
                "data_transform": get_transforms("train", **kwargs),
                "depth_type": cfg["data"]["depth_type"]
                if "gt_depth" in cfg["data"]["val_requirements"]
                else None,
                "with_pose": "gt_pose" in cfg["data"]["val_requirements"],
                "with_ego_pose": "gt_ego_pose" in cfg["data"]["val_requirements"],
                "with_mask": "mask" in cfg["data"]["val_requirements"],
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if cfg["data"]["dataset"] != "nuscenes":
            raise ValueError(f"Unknown dataset: {cfg['data']['dataset']}")

        if mode == "train":
            split = "train"
        else:
            if cfg["model"]["novel_view_mode"] == "MF":
                split = "eval_MF"
            elif cfg["model"]["novel_view_mode"] == "SF":
                split = "eval_SF"
            else:
                raise ValueError(
                    f"Unknown novel view mode: {cfg['model']['novel_view_mode']}"
                )

        return EnvNuScenesDataset(split, **dataset_args)

    class TorchScriptPoseNet(torch.nn.Module):
        def __init__(self, torchscript_dir, mode, device, fusion_level):
            super().__init__()
            path = os.path.join(torchscript_dir, f"pose_net_{mode}.pt")
            self.pose_net = torch.jit.load(path, map_location=device).eval()
            self.fusion_level = fusion_level

        def forward(self, inputs, frame_ids, _):
            cur_image = inputs[("color_aug", frame_ids[0], 0)]
            next_image = inputs[("color_aug", frame_ids[1], 0)]
            mask = inputs["mask"]
            k = inputs[("K", self.fusion_level + 1)]
            inv_k = inputs[("inv_K", self.fusion_level + 1)]
            extrinsics = inputs["extrinsics"]
            extrinsics_inv = inputs["extrinsics_inv"]
            axis_angle, translation = self.pose_net(
                cur_image,
                next_image,
                mask,
                k,
                inv_k,
                extrinsics,
                extrinsics_inv,
            )
            return axis_angle, translation

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
            feat0, feat1, proj_feat, img_feat0, img_feat1, img_feat2 = (
                self.depth_encoder(images, mask, k, inv_k, extrinsics, extrinsics_inv)
            )
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
                feat0_last, feat1_last, proj_feat_last, img_feat0_last, img_feat1_last, img_feat2_last = enc_out[6:12]
                feat0_next, feat1_next, proj_feat_next, img_feat0_next, img_feat1_next, img_feat2_next = enc_out[12:18]
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
        def __init__(self, torchscript_dir, mode, device):
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

    class TorchScriptDrivingForwardModel(DrivingForwardModel):
        def __init__(self, cfg, rank, torchscript_dir):
            self.torchscript_dir = torchscript_dir
            super().__init__(cfg, rank)

        def prepare_model(self, cfg, rank):
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            mode = cfg["model"]["novel_view_mode"]
            models = {
                "pose_net": TorchScriptPoseNet(
                    self.torchscript_dir,
                    mode,
                    device,
                    cfg["model"]["fusion_level"],
                ),
                "depth_net": TorchScriptDepthNet(
                    cfg, self.torchscript_dir, mode, device
                ),
            }
            if self.gaussian:
                models["gs_net"] = TorchScriptGaussianNet(
                    self.torchscript_dir, mode, device
                )
            return models

        def load_weights(self):
            if self.rank == 0:
                print("Skipping weight loading for torchscript mode.")

    class EnvTorchScriptDrivingForwardModel(TorchScriptDrivingForwardModel):
        def __init__(self, cfg, rank, torchscript_dir):
            super().__init__(cfg, rank, torchscript_dir)

        def set_eval_dataloader(self, cfg):
            augmentation = {
                "image_shape": (int(self.height), int(self.width)),
                "jittering": (0.0, 0.0, 0.0, 0.0),
                "crop_train_borders": (),
                "crop_eval_borders": (),
            }

            dataloader_opts = {
                "batch_size": self.eval_batch_size,
                "shuffle": False,
                "num_workers": self.eval_num_workers,
                "pin_memory": True,
                "drop_last": True,
            }

            eval_dataset = construct_env_dataset(cfg, "eval", **augmentation)
            self._dataloaders["eval"] = DataLoader(eval_dataset, **dataloader_opts)

    trainer = DrivingForwardTrainer(cfg, 0, use_tb=False)
    model = EnvTorchScriptDrivingForwardModel(cfg, 0, torchscript_dir)
    return trainer, model


def main():
    args = parse_args()
    repo_root = os.getcwd()
    drivingforward_root = os.path.abspath(args.drivingforward_root)

    config_file = resolve_path(repo_root, args.config_file)
    torchscript_dir = resolve_path(repo_root, args.torchscript_dir)
    weight_path = resolve_path(repo_root, args.weight_path)

    nuscenes_root = load_nuscenes_root()
    depth_map_root = os.path.join(os.path.dirname(nuscenes_root), "samples", "DEPTH_MAP")
    shutil.rmtree(depth_map_root, ignore_errors=True)

    from drivingforward_gsplat import utils

    cfg = utils.get_config(
        config_file,
        mode="eval",
        weight_path=weight_path,
        novel_view_mode=args.novel_view_mode,
    )
    cfg["data"]["data_path"] = nuscenes_root

    ensure_torchscript_modules(torchscript_dir, args.torchscript_repo)
    trainer, model = build_inference(cfg, args, torchscript_dir, drivingforward_root)
    trainer.evaluate(model)


if __name__ == "__main__":
    main()
