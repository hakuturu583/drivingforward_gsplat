import os
from typing import Union

import numpy as np
import torch
import yaml
from PIL import Image
from collections import defaultdict

ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]

_NUSC_CAM_LIST = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
]
_REL_CAM_DICT = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]}


def camera2ind(cameras):
    """
    This function transforms camera name list to indices
    """
    indices = []
    for cam in cameras:
        if cam in _NUSC_CAM_LIST:
            ind = _NUSC_CAM_LIST.index(cam)
        else:
            ind = None
        indices.append(ind)
    return indices


def get_relcam(cameras):
    """
    This function returns relative camera indices from given camera list
    """
    relcam_dict = defaultdict(list)
    indices = camera2ind(cameras)
    for ind in indices:
        relcam_dict[ind] = []
        relcam_cand = _REL_CAM_DICT[ind]
        for cand in relcam_cand:
            if cand in indices:
                relcam_dict[ind].append(cand)
    return relcam_dict


def get_config(config, mode="train", weight_path="./", novel_view_mode="MF"):
    """
    This function reads the configuration file and return as dictionary
    """
    with open(config, "r") as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

        cfg_name = os.path.splitext(os.path.basename(config))[0]
        print("Experiment: ", cfg_name)

        _log_path = os.path.join(cfg["data"]["log_dir"], cfg_name)
        cfg["data"]["log_path"] = _log_path
        cfg["data"]["save_weights_root"] = os.path.join(_log_path, "models")
        cfg["data"]["num_cams"] = len(cfg["data"]["cameras"])
        cfg["data"]["rel_cam_list"] = get_relcam(cfg["data"]["cameras"])

        cfg["model"]["mode"] = mode
        cfg["model"]["novel_view_mode"] = novel_view_mode

        cfg["load"]["load_weights_dir"] = weight_path

        if mode == "eval":
            cfg["ddp"]["world_size"] = 1
            cfg["ddp"]["gpus"] = [0]
            cfg["training"]["batch_size"] = cfg["eval"]["eval_batch_size"]
            cfg["training"]["depth_flip"] = False
    return cfg


def pretty_ts(ts):
    """
    This function prints amount of time taken in user friendly way.
    """
    second = int(ts)
    minute = second // 60
    hour = minute // 60
    return f"{hour:02d}h{(minute%60):02d}m{(second%60):02d}s"


def cal_depth_error(pred, target):
    """
    This function calculates depth error using various metrics.
    """
    abs_rel = torch.mean(torch.abs(pred - target) / target)
    sq_rel = torch.mean((pred - target).pow(2) / target)
    rmse = torch.sqrt(torch.mean((pred - target).pow(2)))
    rmse_log = torch.sqrt(torch.mean((torch.log(target) - torch.log(pred)).pow(2)))

    thresh = torch.max((target / pred), (pred / target))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def find_project_root(start_dir: str) -> str:
    current = os.path.abspath(start_dir)
    while True:
        if os.path.isfile(os.path.join(current, "pyproject.toml")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("pyproject.toml not found in parent directories.")
        current = parent


def tensor_to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.permute(1, 2, 0)
    arr = arr.numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def to_pil_rgb(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        return Image.fromarray(tensor_to_numpy_uint8(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")
