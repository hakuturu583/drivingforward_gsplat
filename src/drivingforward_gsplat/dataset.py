import os
import random
import shutil
import tempfile
from functools import partial

import cv2
import numpy as np
import PIL.Image as pil

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

_PIL_INTERPOLATION = pil.Resampling.LANCZOS


_DEL_KEYS = [
    "rgb",
    "rgb_context",
    "rgb_original",
    "rgb_context_original",
    "intrinsics",
    "contexts",
    "splitname",
    "ego_pose",
]


def is_numpy(data):
    """Checks if data is a numpy array."""
    return isinstance(data, np.ndarray)


def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor


def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list)


def is_tuple(data):
    """Checks if data is a tuple."""
    return isinstance(data, tuple)


def is_int(data):
    """Checks if data is an integer."""
    return isinstance(data, int)


def is_seq(data):
    """Checks if data is a list or tuple."""
    return is_tuple(data) or is_list(data)


def filter_dict(dictionary, keywords):
    """Returns only the keywords that are part of a dictionary."""
    return [key for key in keywords if key in dictionary]


def parse_crop_borders(borders, shape):
    """
    Calculate borders for cropping.

    Parameters
    ----------
    borders : tuple
        Border input for parsing. Can be one of the following forms:
        (int, int, int, int): y, height, x, width
        (int, int): y, x --> y, height = image_height - y, x, width = image_width - x
        Negative numbers are taken from image borders, according to the shape argument
        Float numbers for y and x are treated as percentage, according to the shape argument,
            and in this case height and width are centered at that point.
    shape : tuple
        Image shape (image_height, image_width), used to determine negative crop boundaries

    Returns
    -------
    borders : tuple (left, top, right, bottom)
        Parsed borders for cropping
    """
    if len(borders) == 0:
        return 0, 0, shape[1], shape[0]
    borders = list(borders).copy()
    if len(borders) == 4:
        borders = [borders[2], borders[0], borders[3], borders[1]]
        if is_int(borders[0]):
            borders[0] += shape[1] if borders[0] < 0 else 0
            borders[2] += shape[1] if borders[2] <= 0 else borders[0]
        else:
            center_w, half_w = borders[0] * shape[1], borders[2] / 2
            borders[0] = int(center_w - half_w)
            borders[2] = int(center_w + half_w)
        if is_int(borders[1]):
            borders[1] += shape[0] if borders[1] < 0 else 0
            borders[3] += shape[0] if borders[3] <= 0 else borders[1]
        else:
            center_h, half_h = borders[1] * shape[0], borders[3] / 2
            borders[1] = int(center_h - half_h)
            borders[3] = int(center_h + half_h)
    elif len(borders) == 2:
        borders = [borders[1], borders[0]]
        if is_int(borders[0]):
            borders = (
                max(0, borders[0]),
                max(0, borders[1]),
                shape[1] + min(0, borders[0]),
                shape[0] + min(0, borders[1]),
            )
        else:
            center_w, half_w = borders[0] * shape[1], borders[1] / 2
            center_h, half_h = borders[0] * shape[0], borders[1] / 2
            borders = (
                int(center_w - half_w),
                int(center_h - half_h),
                int(center_w + half_w),
                int(center_h + half_h),
            )
    else:
        raise NotImplementedError("Crop tuple must have 2 or 4 values.")
    assert (
        0 <= borders[0] < borders[2] <= shape[1]
        and 0 <= borders[1] < borders[3] <= shape[0]
    ), f"Crop borders {borders} are invalid"
    return borders


def resize_image(image, shape, interpolation=_PIL_INTERPOLATION):
    """Resizes input image."""
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)


def resize_depth(depth, shape):
    """Resizes depth map."""
    depth = cv2.resize(depth, dsize=shape[::-1], interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(depth, axis=2)


def resize_depth_preserve(depth, shape):
    """Resizes depth map preserving all valid depth pixels."""
    if depth is None:
        return depth
    if not is_seq(shape):
        shape = tuple(int(s * shape) for s in depth.shape)
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    idx = x > 0
    crd, val = uv[idx], x[idx]
    crd[:, 0] = (crd[:, 0] * (shape[0] / h)).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w)).astype(np.int32)
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    return np.expand_dims(depth, axis=2)


def resize_sample_image_and_intrinsics(
    sample, shape, image_interpolation=_PIL_INTERPOLATION
):
    """Resizes the image and intrinsics of a sample."""
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    orig_w, orig_h = sample["rgb"].size
    out_h, out_w = shape
    for key in filter_dict(sample, ["intrinsics"]):
        intrinsics = np.copy(sample[key])
        intrinsics[0] *= out_w / orig_w
        intrinsics[1] *= out_h / orig_h
        sample[key] = intrinsics
    for key in filter_dict(sample, ["rgb", "rgb_original"]):
        sample[key] = image_transform(sample[key])
    for key in filter_dict(sample, ["rgb_context", "rgb_context_original"]):
        sample[key] = [image_transform(k) for k in sample[key]]
    return sample


def resize_sample(sample, shape, image_interpolation=_PIL_INTERPOLATION):
    """Resizes a sample, including image, intrinsics and depth maps."""
    sample = resize_sample_image_and_intrinsics(sample, shape, image_interpolation)
    for key in filter_dict(sample, ["depth", "input_depth"]):
        sample[key] = resize_depth_preserve(sample[key], shape)
    for key in filter_dict(sample, ["depth_context"]):
        sample[key] = [resize_depth_preserve(k, shape) for k in sample[key]]
    return sample


def to_tensor_sample(sample, tensor_type="torch.FloatTensor"):
    """Casts the keys of sample to tensors."""
    transform = transforms.ToTensor()
    for key in filter_dict(sample, ["rgb", "rgb_original", "depth", "input_depth"]):
        sample[key] = transform(sample[key]).type(tensor_type)
    for key in filter_dict(
        sample, ["rgb_context", "rgb_context_original", "depth_context"]
    ):
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]
    return sample


def duplicate_sample(sample):
    """Duplicates sample images and contexts to preserve their unaugmented versions."""
    for key in filter_dict(sample, ["rgb"]):
        sample[f"{key}_original"] = sample[key].copy()
    for key in filter_dict(sample, ["rgb_context"]):
        sample[f"{key}_original"] = [k.copy() for k in sample[key]]
    return sample


def random_color_jitter_transform(parameters):
    """Creates a reusable color jitter transformation."""
    brightness, contrast, saturation, hue = parameters
    brightness = [max(0, 1 - brightness), 1 + brightness]
    contrast = [max(0, 1 - contrast), 1 + contrast]
    saturation = [max(0, 1 - saturation), 1 + saturation]
    hue = [-hue, hue]

    all_transforms = []
    if brightness is not None:
        brightness_factor = random.uniform(brightness[0], brightness[1])
        all_transforms.append(
            transforms.Lambda(
                lambda img: transforms.functional.adjust_brightness(
                    img, brightness_factor
                )
            )
        )
    if contrast is not None:
        contrast_factor = random.uniform(contrast[0], contrast[1])
        all_transforms.append(
            transforms.Lambda(
                lambda img: transforms.functional.adjust_contrast(img, contrast_factor)
            )
        )
    if saturation is not None:
        saturation_factor = random.uniform(saturation[0], saturation[1])
        all_transforms.append(
            transforms.Lambda(
                lambda img: transforms.functional.adjust_saturation(
                    img, saturation_factor
                )
            )
        )
    if hue is not None:
        hue_factor = random.uniform(hue[0], hue[1])
        all_transforms.append(
            transforms.Lambda(
                lambda img: transforms.functional.adjust_hue(img, hue_factor)
            )
        )
    random.shuffle(all_transforms)
    return transforms.Compose(all_transforms)


def colorjitter_sample(sample, parameters, prob=1.0):
    """Jitters input images as data augmentation."""
    if random.random() < prob:
        color_jitter_transform = random_color_jitter_transform(parameters[:4])
        if len(parameters) > 4 and parameters[4] > 0:
            matrix = (
                random.uniform(1.0 - parameters[4], 1 + parameters[4]),
                0,
                0,
                0,
                0,
                random.uniform(1.0 - parameters[4], 1 + parameters[4]),
                0,
                0,
                0,
                0,
                random.uniform(1.0 - parameters[4], 1 + parameters[4]),
                0,
            )
        else:
            matrix = None
        for key in filter_dict(sample, ["rgb"]):
            sample[key] = color_jitter_transform(sample[key])
            if matrix is not None:
                sample[key] = sample[key].convert("RGB", matrix)
        for key in filter_dict(sample, ["rgb_context"]):
            sample[key] = [color_jitter_transform(k) for k in sample[key]]
            if matrix is not None:
                sample[key] = [k.convert("RGB", matrix) for k in sample[key]]
    return sample


def crop_image(image, borders):
    """Crop a PIL Image."""
    return image.crop(borders)


def crop_intrinsics(intrinsics, borders):
    """Crop camera intrinsics matrix."""
    intrinsics = np.copy(intrinsics)
    intrinsics[0, 2] -= borders[0]
    intrinsics[1, 2] -= borders[1]
    return intrinsics


def crop_depth(depth, borders):
    """Crop a numpy depth map."""
    if depth is None:
        return depth
    return depth[borders[1] : borders[3], borders[0] : borders[2]]


def crop_sample_input(sample, borders):
    """Crops the input information of a sample."""
    for key in filter_dict(sample, ["intrinsics"]):
        if key + "_full" not in sample:
            sample[key + "_full"] = np.copy(sample[key])
        sample[key] = crop_intrinsics(sample[key], borders)
    for key in filter_dict(sample, ["rgb", "rgb_original", "warped_rgb"]):
        sample[key] = crop_image(sample[key], borders)
    for key in filter_dict(sample, ["rgb_context", "rgb_context_original"]):
        sample[key] = [crop_image(val, borders) for val in sample[key]]
    for key in filter_dict(sample, ["input_depth", "bbox2d_depth", "bbox3d_depth"]):
        sample[key] = crop_depth(sample[key], borders)
    for key in filter_dict(sample, ["input_depth_context"]):
        sample[key] = [crop_depth(val, borders) for val in sample[key]]
    return sample


def crop_sample_supervision(sample, borders):
    """Crops the output information of a sample."""
    for key in filter_dict(
        sample,
        [
            "depth",
            "bbox2d_depth",
            "bbox3d_depth",
            "semantic",
            "bwd_optical_flow",
            "fwd_optical_flow",
            "valid_fwd_optical_flow",
            "bwd_scene_flow",
            "fwd_scene_flow",
        ],
    ):
        sample[key] = crop_depth(sample[key], borders)
    for key in filter_dict(
        sample,
        [
            "depth_context",
            "semantic_context",
            "bwd_optical_flow_context",
            "fwd_optical_flow_context",
            "bwd_scene_flow_context",
            "fwd_scene_flow_context",
        ],
    ):
        sample[key] = [crop_depth(k, borders) for k in sample[key]]
    return sample


def crop_sample(sample, borders):
    """Crops a sample, including image, intrinsics and depth maps."""
    sample = crop_sample_input(sample, borders)
    sample = crop_sample_supervision(sample, borders)
    return sample


def train_transforms(sample, image_shape, jittering, crop_train_borders):
    """Training data augmentation transformations."""
    if len(crop_train_borders) > 0:
        borders = parse_crop_borders(crop_train_borders, sample["rgb"].size[::-1])
        sample = crop_sample(sample, borders)
    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering)
    sample = to_tensor_sample(sample)
    return sample


def validation_transforms(sample, image_shape, crop_eval_borders):
    """Validation data augmentation transformations."""
    if len(crop_eval_borders) > 0:
        borders = parse_crop_borders(crop_eval_borders, sample["rgb"].size[::-1])
        sample = crop_sample_input(sample, borders)
    if len(image_shape) > 0:
        sample["rgb"] = resize_image(sample["rgb"], image_shape)
        if "input_depth" in sample:
            sample["input_depth"] = resize_depth_preserve(
                sample["input_depth"], image_shape
            )
    sample = to_tensor_sample(sample)
    return sample


def test_transforms(sample, image_shape, crop_eval_borders):
    """Test data augmentation transformations."""
    if len(crop_eval_borders) > 0:
        borders = parse_crop_borders(crop_eval_borders, sample["rgb"].size[::-1])
        sample = crop_sample_input(sample, borders)
    if len(image_shape) > 0:
        sample["rgb"] = resize_image(sample["rgb"], image_shape)
        if "input_depth" in sample:
            sample["input_depth"] = resize_depth(sample["input_depth"], image_shape)
    sample = to_tensor_sample(sample)
    return sample


def get_transforms(
    mode, image_shape, jittering, crop_train_borders, crop_eval_borders, **kwargs
):
    """Get data augmentation transformations for each split."""
    if mode == "train":
        return partial(
            train_transforms,
            image_shape=image_shape,
            jittering=jittering,
            crop_train_borders=crop_train_borders,
        )
    if mode == "validation":
        return partial(
            validation_transforms,
            crop_eval_borders=crop_eval_borders,
            image_shape=image_shape,
        )
    if mode == "test":
        return partial(
            test_transforms,
            crop_eval_borders=crop_eval_borders,
            image_shape=image_shape,
        )
    raise ValueError(f"Unknown mode {mode}")


def transform_mask_sample(sample, data_transform):
    """Transforms masks to match input rgb images."""
    image_shape = data_transform.keywords["image_shape"]
    resize_transform = transforms.Resize(image_shape, interpolation=_PIL_INTERPOLATION)
    sample["mask"] = resize_transform(sample["mask"])
    tensor_transform = transforms.ToTensor()
    sample["mask"] = tensor_transform(sample["mask"])
    return sample


def img_loader(path):
    """Loads rgb image."""
    with open(path, "rb") as f:
        with pil.open(f) as img:
            return img.convert("RGB")


def mask_loader_scene(path, mask_idx, cam):
    """Loads mask that correspondes to the scene and camera."""
    fname = os.path.join(path, str(mask_idx), f"{cam.upper()}_mask.png")
    with open(fname, "rb") as f:
        with pil.open(f) as img:
            return img.convert("L")


def align_dataset(sample, scales, contexts):
    """Reorganize samples to match our trainer configuration."""
    K = sample["intrinsics"]
    aug_images = sample["rgb"]
    aug_contexts = sample["rgb_context"]
    org_images = sample["rgb_original"]
    org_contexts = sample["rgb_context_original"]
    ego_poses = sample["ego_pose"]

    n_cam, _, w, h = aug_images.shape

    resized_K = np.expand_dims(np.eye(4), 0).repeat(n_cam, axis=0)
    resized_K[:, :3, :3] = K

    for scale in scales:
        scaled_K = resized_K.copy()
        scaled_K[:, :2, :] /= 2 ** scale

        sample[("K", scale)] = scaled_K.copy()
        sample[("inv_K", scale)] = np.linalg.pinv(scaled_K).copy()

        resized_org = F.interpolate(
            org_images,
            size=(w // (2 ** scale), h // (2 ** scale)),
            mode="bilinear",
            align_corners=False,
        )
        resized_aug = F.interpolate(
            aug_images,
            size=(w // (2 ** scale), h // (2 ** scale)),
            mode="bilinear",
            align_corners=False,
        )

        sample[("color", 0, scale)] = resized_org
        sample[("color_aug", 0, scale)] = resized_aug

    for idx, frame in enumerate(contexts):
        sample[("color", frame, 0)] = org_contexts[idx]
        sample[("color_aug", frame, 0)] = aug_contexts[idx]
        sample[("cam_T_cam", 0, frame)] = ego_poses[idx]

    for key in list(sample.keys()):
        if key in _DEL_KEYS:
            del sample[key]
    return sample


def stack_sample(sample):
    """Stack a sample from multiple sensors."""
    if len(sample) == 1:
        return sample[0]

    stacked_sample = {}
    for key in sample[0]:
        if key in ["idx", "dataset_idx", "sensor_name", "filename", "token"]:
            stacked_sample[key] = sample[0][key]
        else:
            if is_tensor(sample[0][key]):
                stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
            elif is_numpy(sample[0][key]):
                stacked_sample[key] = np.stack([s[key] for s in sample], 0)
            elif is_list(sample[0][key]):
                stacked_sample[key] = []
                if is_tensor(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            torch.stack([s[key][i] for s in sample], 0)
                        )
                if is_numpy(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            np.stack([s[key][i] for s in sample], 0)
                        )

    return stacked_sample


class NuScenesdataset(Dataset):
    """Loaders for NuScenes dataset."""

    def __init__(
        self,
        path,
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
        super().__init__()
        version = "v1.0-trainval"
        self.path = path
        self.split = split
        self.dataset_idx = 0

        self.cameras = cameras
        self.scales = np.arange(scale_range + 2)
        self.num_cameras = len(cameras)

        self.bwd = back_context
        self.fwd = forward_context

        self.has_context = back_context + forward_context > 0
        self.data_transform = data_transform

        self.with_depth = depth_type is not None
        self.with_pose = with_pose
        self.with_ego_pose = with_ego_pose

        self.loader = img_loader

        self.with_mask = with_mask
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.mask_path = os.path.join(cur_path, "nuscenes_mask")
        self.mask_loader = mask_loader_scene

        self.dataset = NuScenes(version=version, dataroot=self.path, verbose=True)

        split_list_path = self._resolve_split_list_path(self.split)
        with open(split_list_path, "r") as f:
            self.filenames = f.readlines()

    def _resolve_split_list_path(self, split):
        tmp_dir = os.path.join(
            tempfile.gettempdir(), "drivingforward_gsplat", "nuscenes"
        )
        tmp_path = os.path.join(tmp_dir, f"{split}.txt")
        if os.path.exists(tmp_path):
            return tmp_path

        legacy_path = os.path.join("dataset", "nuscenes", f"{split}.txt")
        if os.path.exists(legacy_path):
            os.makedirs(tmp_dir, exist_ok=True)
            shutil.copyfile(legacy_path, tmp_path)
            return tmp_path

        if split in ("eval_MF", "eval_SF"):
            os.makedirs(tmp_dir, exist_ok=True)
            with open(tmp_path, "w") as f:
                for sample in self.dataset.sample:
                    if not self._has_required_context(sample):
                        continue
                    f.write(sample["token"] + "\n")
            return tmp_path

        return legacy_path

    def _has_required_context(self, sample):
        if not self.has_context:
            return True

        for cam in self.cameras:
            cam_sample = self.dataset.get("sample_data", sample["data"][cam])
            if self.bwd and not cam_sample["prev"]:
                return False
            if self.fwd and not cam_sample["next"]:
                return False
        return True

    def get_current(self, key, cam_sample):
        """This function returns samples for current contexts."""
        if key == "rgb":
            rgb_path = cam_sample["filename"]
            return self.loader(os.path.join(self.path, rgb_path))
        if key == "intrinsics":
            cam_param = self.dataset.get(
                "calibrated_sensor", cam_sample["calibrated_sensor_token"]
            )
            return np.array(cam_param["camera_intrinsic"], dtype=np.float32)
        if key == "extrinsics":
            cam_param = self.dataset.get(
                "calibrated_sensor", cam_sample["calibrated_sensor_token"]
            )
            return self.get_tranformation_mat(cam_param)
        raise ValueError(f"Unknown key: {key}")

    def get_context(self, key, cam_sample):
        """This function returns samples for backward and forward contexts."""
        bwd_context, fwd_context = [], []
        if self.bwd != 0:
            if self.split == "eval_SF":
                bwd_sample = cam_sample
            else:
                bwd_sample = self.dataset.get("sample_data", cam_sample["prev"])
            bwd_context = [self.get_current(key, bwd_sample)]

        if self.fwd != 0:
            fwd_sample = self.dataset.get("sample_data", cam_sample["next"])
            fwd_context = [self.get_current(key, fwd_sample)]
        return bwd_context + fwd_context

    def get_cam_T_cam(self, cam_sample):
        cam_to_ego = self.dataset.get(
            "calibrated_sensor", cam_sample["calibrated_sensor_token"]
        )
        cam_to_ego_rotation = Quaternion(cam_to_ego["rotation"])
        cam_to_ego_translation = np.array(cam_to_ego["translation"])[:, None]
        cam_to_ego = np.vstack(
            [
                np.hstack(
                    (cam_to_ego_rotation.rotation_matrix, cam_to_ego_translation)
                ),
                np.array([0, 0, 0, 1]),
            ]
        )

        world_to_ego = self.dataset.get("ego_pose", cam_sample["ego_pose_token"])
        world_to_ego_rotation = Quaternion(world_to_ego["rotation"]).inverse
        world_to_ego_translation = -np.array(world_to_ego["translation"])[:, None]
        world_to_ego = np.vstack(
            [
                np.hstack(
                    (
                        world_to_ego_rotation.rotation_matrix,
                        world_to_ego_rotation.rotation_matrix
                        @ world_to_ego_translation,
                    )
                ),
                np.array([0, 0, 0, 1]),
            ]
        )
        ego_to_world = np.linalg.inv(world_to_ego)

        cam_T_cam = []

        if self.bwd != 0:
            if self.split == "eval_SF":
                bwd_sample = cam_sample
            else:
                bwd_sample = self.dataset.get("sample_data", cam_sample["prev"])

            world_to_ego_bwd = self.dataset.get(
                "ego_pose", bwd_sample["ego_pose_token"]
            )
            world_to_ego_bwd_rotation = Quaternion(world_to_ego_bwd["rotation"]).inverse
            world_to_ego_bwd_translation = -np.array(world_to_ego_bwd["translation"])[
                :, None
            ]
            world_to_ego_bwd = np.vstack(
                [
                    np.hstack(
                        (
                            world_to_ego_bwd_rotation.rotation_matrix,
                            world_to_ego_bwd_rotation.rotation_matrix
                            @ world_to_ego_bwd_translation,
                        )
                    ),
                    np.array([0, 0, 0, 1]),
                ]
            )

            cam_to_ego_bwd = self.dataset.get(
                "calibrated_sensor", bwd_sample["calibrated_sensor_token"]
            )
            cam_to_ego_bwd_rotation = Quaternion(cam_to_ego_bwd["rotation"])
            cam_to_ego_bwd_translation = np.array(cam_to_ego_bwd["translation"])[
                :, None
            ]
            cam_to_ego_bwd = np.vstack(
                [
                    np.hstack(
                        (
                            cam_to_ego_bwd_rotation.rotation_matrix,
                            cam_to_ego_bwd_translation,
                        )
                    ),
                    np.array([0, 0, 0, 1]),
                ]
            )
            ego_to_cam_bwd = np.linalg.inv(cam_to_ego_bwd)

            cam_T_cam_bwd = (
                ego_to_cam_bwd @ world_to_ego_bwd @ ego_to_world @ cam_to_ego
            )

            cam_T_cam.append(cam_T_cam_bwd)

        if self.fwd != 0:
            fwd_sample = self.dataset.get("sample_data", cam_sample["next"])

            world_to_ego_fwd = self.dataset.get(
                "ego_pose", fwd_sample["ego_pose_token"]
            )
            world_to_ego_fwd_rotation = Quaternion(world_to_ego_fwd["rotation"]).inverse
            world_to_ego_fwd_translation = -np.array(world_to_ego_fwd["translation"])[
                :, None
            ]
            world_to_ego_fwd = np.vstack(
                [
                    np.hstack(
                        (
                            world_to_ego_fwd_rotation.rotation_matrix,
                            world_to_ego_fwd_rotation.rotation_matrix
                            @ world_to_ego_fwd_translation,
                        )
                    ),
                    np.array([0, 0, 0, 1]),
                ]
            )

            cam_to_ego_fwd = self.dataset.get(
                "calibrated_sensor", fwd_sample["calibrated_sensor_token"]
            )
            cam_to_ego_fwd_rotation = Quaternion(cam_to_ego_fwd["rotation"])
            cam_to_ego_fwd_translation = np.array(cam_to_ego_fwd["translation"])[
                :, None
            ]
            cam_to_ego_fwd = np.vstack(
                [
                    np.hstack(
                        (
                            cam_to_ego_fwd_rotation.rotation_matrix,
                            cam_to_ego_fwd_translation,
                        )
                    ),
                    np.array([0, 0, 0, 1]),
                ]
            )
            ego_to_cam_fwd = np.linalg.inv(cam_to_ego_fwd)

            cam_T_cam_fwd = (
                ego_to_cam_fwd @ world_to_ego_fwd @ ego_to_world @ cam_to_ego
            )

            cam_T_cam.append(cam_T_cam_fwd)

        return cam_T_cam

    def generate_depth_map(self, sample, sensor, cam_sample):
        """Returns depth map for nuscenes dataset."""
        filename = "{}/{}.npz".format(
            os.path.join(os.path.dirname(self.path), "samples"),
            f"DEPTH_MAP/{sensor}/{cam_sample['filename']}",
        )

        if os.path.exists(filename):
            try:
                loaded = np.load(filename, allow_pickle=True)
                depth = loaded["depth"]
                if depth.size == 0:
                    raise ValueError("empty depth cache")
                return depth
            except (EOFError, ValueError, KeyError, OSError):
                try:
                    os.remove(filename)
                except OSError:
                    pass

        lidar_sample = self.dataset.get("sample_data", sample["data"]["LIDAR_TOP"])

        lidar_file = os.path.join(self.path, lidar_sample["filename"])
        lidar_points = np.fromfile(lidar_file, dtype=np.float32)
        lidar_points = lidar_points.reshape(-1, 5)[:, :3]

        lidar_pose = self.dataset.get("ego_pose", lidar_sample["ego_pose_token"])
        lidar_rotation = Quaternion(lidar_pose["rotation"])
        lidar_translation = np.array(lidar_pose["translation"])[:, None]
        lidar_to_world = np.vstack(
            [
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1]),
            ]
        )

        sensor_sample = self.dataset.get(
            "calibrated_sensor", lidar_sample["calibrated_sensor_token"]
        )
        lidar_to_ego_rotation = Quaternion(sensor_sample["rotation"]).rotation_matrix
        lidar_to_ego_translation = np.array(sensor_sample["translation"]).reshape(1, 3)

        ego_lidar_points = np.dot(lidar_points[:, :3], lidar_to_ego_rotation.T)
        ego_lidar_points += lidar_to_ego_translation

        homo_ego_lidar_points = np.concatenate(
            (ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1
        )

        ego_pose = self.dataset.get("ego_pose", cam_sample["ego_pose_token"])
        ego_rotation = Quaternion(ego_pose["rotation"]).inverse
        ego_translation = -np.array(ego_pose["translation"])[:, None]
        world_to_ego = np.vstack(
            [
                np.hstack(
                    (
                        ego_rotation.rotation_matrix,
                        ego_rotation.rotation_matrix @ ego_translation,
                    )
                ),
                np.array([0, 0, 0, 1]),
            ]
        )

        sensor_sample = self.dataset.get(
            "calibrated_sensor", cam_sample["calibrated_sensor_token"]
        )
        sensor_rotation = Quaternion(sensor_sample["rotation"])
        sensor_translation = np.array(sensor_sample["translation"])[:, None]
        sensor_to_ego = np.vstack(
            [
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1]),
            ]
        )
        ego_to_sensor = np.linalg.inv(sensor_to_ego)

        lidar_to_sensor = ego_to_sensor @ world_to_ego @ lidar_to_world
        homo_ego_lidar_points = torch.from_numpy(homo_ego_lidar_points).float()
        cam_lidar_points = np.matmul(lidar_to_sensor, homo_ego_lidar_points.T).T

        depth_mask = cam_lidar_points[:, 2] > 0
        cam_lidar_points = cam_lidar_points[depth_mask]

        intrinsics = np.eye(4)
        intrinsics[:3, :3] = sensor_sample["camera_intrinsic"]
        pixel_points = np.matmul(intrinsics, cam_lidar_points.T).T
        pixel_points[:, :2] /= pixel_points[:, 2:3]

        image_filename = os.path.join(self.path, cam_sample["filename"])
        img = pil.open(image_filename)
        h, w, _ = np.array(img).shape

        pixel_mask = (
            (pixel_points[:, 0] >= 0)
            & (pixel_points[:, 0] <= w - 1)
            & (pixel_points[:, 1] >= 0)
            & (pixel_points[:, 1] <= h - 1)
        )
        valid_points = pixel_points[pixel_mask].round().int()
        valid_depth = cam_lidar_points[:, 2][pixel_mask]

        depth = np.zeros([h, w])
        depth[valid_points[:, 1], valid_points[:, 0]] = valid_depth

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez_compressed(filename, depth=depth)
        return depth

    def get_tranformation_mat(self, pose):
        """Transforms pose information in accordance with DDAD dataset format."""
        extrinsics = Quaternion(pose["rotation"]).transformation_matrix
        extrinsics[:3, 3] = np.array(pose["translation"])
        return extrinsics.astype(np.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        frame_idx = self.filenames[idx].strip().split()[0]
        sample_nusc = self.dataset.get("sample", frame_idx)

        sample = []
        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)

        for cam in self.cameras:
            cam_sample = self.dataset.get("sample_data", sample_nusc["data"][cam])

            data = {
                "idx": idx,
                "token": frame_idx,
                "sensor_name": cam,
                "contexts": contexts,
                "filename": cam_sample["filename"],
                "rgb": self.get_current("rgb", cam_sample),
                "intrinsics": self.get_current("intrinsics", cam_sample),
            }

            if self.with_depth:
                data.update(
                    {"depth": self.generate_depth_map(sample_nusc, cam, cam_sample)}
                )
            if self.with_pose:
                data.update({"extrinsics": self.get_current("extrinsics", cam_sample)})
            if self.with_ego_pose:
                data.update({"ego_pose": self.get_cam_T_cam(cam_sample)})
            if self.with_mask:
                data.update({"mask": self.mask_loader(self.mask_path, "", cam)})
            if self.has_context:
                data.update({"rgb_context": self.get_context("rgb", cam_sample)})

            sample.append(data)

        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]
            sample = [transform_mask_sample(smp, self.data_transform) for smp in sample]

        sample = stack_sample(sample)
        sample = align_dataset(sample, self.scales, contexts)
        return sample


def construct_dataset(cfg, mode, **kwargs):
    """Construct datasets based on config and mode."""
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

    if cfg["data"]["dataset"] == "nuscenes":
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
        dataset = NuScenesdataset(cfg["data"]["data_path"], split, **dataset_args)
    else:
        raise ValueError(f"Unknown dataset: {cfg['data']['dataset']}")
    return dataset


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
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
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


_PIL_INTERPOLATION = pil.Resampling.LANCZOS
