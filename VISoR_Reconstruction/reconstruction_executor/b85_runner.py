import json
import os
import re
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import SimpleITK as sitk
import tifffile

from VISoR_Brain.format.visor_data import VISoRData
from VISoR_Brain.positioning.visor_brain import VISoRBrain
from VISoR_Brain.positioning.visor_sample import VISoRSample
from VISoR_Reconstruction.misc import VERSION


REFINEMENT_PRE_STEP_NAMES = [
    "reconstruct_sample",
    "reconstruct_image",
]

B85_REFINEMENT_STEP_NAMES = [
    "step1_1",
    "step1_2",
    "step1_3",
    "step2",
    "extract_surface_failed",
    "step3",
    "check_xy",
    "step4",
    "step4_channel",
]

B85_STEP_NAMES = REFINEMENT_PRE_STEP_NAMES + B85_REFINEMENT_STEP_NAMES


@dataclass
class B85ChannelConfig:
    channel_id: str
    channel_name: str
    laser_wavelength: str
    name_format: str
    image_format: str
    transform_format: str


@dataclass
class B85Config:
    dataset_file: str
    dataset_path: str
    dataset_name: str
    output_root: str
    temp_root: str
    save_res_root: str
    reference_channel_id: str
    output_channel_ids: List[str]
    start_slice: int
    end_slice_exclusive: int
    pixel_size: float = 4.0
    block_size: int = 250
    gap: int = 500
    selected_steps: List[str] = field(default_factory=lambda: list(B85_STEP_NAMES))
    channels: Dict[str, B85ChannelConfig] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "B85Config":
        data = json.loads(text.lstrip("\ufeff"))
        data["reference_channel_id"] = str(data["reference_channel_id"])
        data["output_channel_ids"] = [str(c) for c in data.get("output_channel_ids", [])]
        data["channels"] = {
            k: B85ChannelConfig(**v) for k, v in data.get("channels", {}).items()
        }
        return cls(**data)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _b85_root() -> Path:
    return _repo_root() / "YQReconstructionScripts" / "B85_Test"


def _pixel_dir_name(pixel_size: float) -> str:
    if float(pixel_size).is_integer():
        return f"{float(pixel_size):.1f}"
    return str(pixel_size)


def _channel_wavelength(channel: dict) -> float:
    value = channel.get("LaserWavelength") or channel.get("ChannelName") or ""
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else 0.0


def _install_common0313_shim():
    def preprocess(surface, threshold):
        surface = sitk.Threshold(surface, threshold, 65535, threshold)
        back_log_value = np.log(threshold)
        return sitk.Clamp(
            (sitk.Log(sitk.Cast(surface + 1, sitk.sitkFloat32)) - back_log_value) * 39.4,
            sitk.sitkUInt8,
            0,
            255,
        )

    def fill_outside_yq(img, value: int):
        img[0, 0] = 0
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
        cv2.floodFill(img, mask, (0, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
        img[img.shape[0] - 1, 0] = 0
        cv2.floodFill(img, mask, (0, img.shape[0] - 1), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
        img[img.shape[0] - 1, img.shape[1] - 1] = 0
        cv2.floodFill(
            img,
            mask,
            (img.shape[1] - 1, img.shape[0] - 1),
            value,
            value,
            value,
            cv2.FLOODFILL_FIXED_RANGE,
        )
        img[0, img.shape[1] - 1] = 0
        cv2.floodFill(img, mask, (img.shape[1] - 1, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
        return img

    package_name = "YQReconstructionScripts.CRH"
    module_name = f"{package_name}.common0313"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.Preprocess = preprocess
        module.fill_outside_yq = fill_outside_yq
        sys.modules[module_name] = module


def _prepare_b85_imports():
    repo_root = str(_repo_root())
    b85_root = str(_b85_root())
    for path in [repo_root, b85_root]:
        if path not in sys.path:
            sys.path.insert(0, path)
    _install_common0313_shim()


def _import_b85_modules():
    _prepare_b85_imports()
    import importlib

    return {
        "step1_1": importlib.import_module("utils.step1_1_methods"),
        "step1_2": importlib.import_module("utils.step1_2_use_block"),
        "step1_3": importlib.import_module("utils.step1_3_CalLoss"),
        "step2": importlib.import_module("utils.step2_extract"),
        "step3": importlib.import_module("utils.step3_align"),
        "step4": importlib.import_module("utils.step4_ContiuneProcessTransform"),
    }


def _derive_name_format(dataset: VISoRData, channel_id: str, image_dir: Path) -> str:
    channel_name = dataset.channels[channel_id]["ChannelName"]
    slices = sorted(int(i) for i in dataset.acquisition_results.get(channel_id, {}))
    for slice_id in slices:
        standard = f"{dataset.name}_{slice_id:03d}_{channel_name}.tif"
        if (image_dir / standard).exists():
            return f"{dataset.name}_{{:03d}}_{channel_name}"

        token = f"{slice_id:03d}"
        for candidate in image_dir.glob(f"*{token}*{channel_name}.tif"):
            stem = candidate.name[:-4]
            pos = stem.find(token)
            if pos >= 0:
                return f"{stem[:pos]}{{:03d}}{stem[pos + len(token):]}"

    return f"{dataset.name}_{{:03d}}_{channel_name}"


def _available_channel_config(
    dataset: VISoRData,
    channel_id: str,
    image_dir: Path,
    transform_dir: Path,
) -> B85ChannelConfig:
    channel = dataset.channels[channel_id]
    normalized_channel_id = str(channel["ChannelId"])
    name_format = _derive_name_format(dataset, channel_id, image_dir)
    return B85ChannelConfig(
        channel_id=normalized_channel_id,
        channel_name=channel["ChannelName"],
        laser_wavelength=str(channel.get("LaserWavelength", "")),
        name_format=name_format,
        image_format=str(image_dir / (name_format + ".tif")),
        transform_format=str(transform_dir / (name_format + ".txt")),
    )


def _has_any_slice_image(config: B85ChannelConfig, slices: List[int]) -> bool:
    return any(os.path.isfile(config.image_format.format(i)) for i in slices)


def infer_b85_config(
    dataset: VISoRData,
    output_root: Optional[str] = None,
    temp_root: Optional[str] = None,
    save_res_root: Optional[str] = None,
    reference_channel_id: Optional[str] = None,
    output_channel_ids: Optional[List[str]] = None,
    start_slice: Optional[int] = None,
    end_slice_exclusive: Optional[int] = None,
    pixel_size: float = 4.0,
    block_size: int = 250,
    gap: int = 500,
    selected_steps: Optional[List[str]] = None,
    validate: bool = True,
) -> B85Config:
    if dataset.file is None:
        raise ValueError("Refinement requires a loaded VISoR dataset file.")

    steps = selected_steps or list(B85_STEP_NAMES)
    invalid_steps = [s for s in steps if s not in B85_STEP_NAMES]
    if invalid_steps:
        raise ValueError(f"Unknown Refinement steps: {invalid_steps}")

    image_dir = Path(dataset.path) / "Reconstruction" / "SliceImage" / _pixel_dir_name(pixel_size)
    transform_dir = Path(dataset.path) / "Reconstruction" / "SliceTransform"
    needs_b85_inputs = any(step in B85_REFINEMENT_STEP_NAMES for step in steps)
    if validate and needs_b85_inputs and "reconstruct_image" not in steps and not image_dir.exists():
        raise FileNotFoundError(f"Slice image directory does not exist: {image_dir}")
    if validate and needs_b85_inputs and "reconstruct_sample" not in steps and not transform_dir.exists():
        raise FileNotFoundError(f"Slice transform directory does not exist: {transform_dir}")

    dataset_channel_keys = {str(dataset.channels[c]["ChannelId"]): c for c in dataset.channels}
    channels = {
        str(dataset.channels[c]["ChannelId"]): _available_channel_config(dataset, c, image_dir, transform_dir)
        for c in dataset.channels
    }
    all_slices = sorted(
        {int(i) for c in dataset.acquisition_results for i in dataset.acquisition_results[c]}
    )
    if len(all_slices) < 2:
        raise ValueError("Refinement requires at least two slice indices.")

    if start_slice is None:
        start_slice = all_slices[0]
    if end_slice_exclusive is None:
        end_slice_exclusive = all_slices[-1]
    if start_slice >= end_slice_exclusive:
        raise ValueError("Refinement end_slice_exclusive must be greater than start_slice.")

    if reference_channel_id is None:
        preferred = [
            str(v["ChannelId"]) for c, v in dataset.channels.items()
            if "640" in str(v.get("LaserWavelength", "")) or "640" in v.get("ChannelName", "")
        ]
        if preferred:
            reference_channel_id = preferred[0]
        else:
            reference_channel_id = str(dataset.channels[
                max(dataset.channels, key=lambda c: _channel_wavelength(dataset.channels[c]))
            ]["ChannelId"])
    else:
        reference_channel_id = str(reference_channel_id)

    selected_slices = list(range(start_slice, end_slice_exclusive + 1))
    if output_channel_ids is None:
        if "reconstruct_image" in steps:
            output_channel_ids = [
                str(dataset.channels[c]["ChannelId"])
                for c in dataset.channels
                if c in dataset.acquisition_results
                and any(int(i) in selected_slices for i in dataset.acquisition_results[c])
            ]
        else:
            output_channel_ids = [
                c for c, cfg in channels.items()
                if _has_any_slice_image(cfg, selected_slices)
            ]
        if not output_channel_ids and not validate:
            output_channel_ids = [str(dataset.channels[c]["ChannelId"]) for c in dataset.channels]
    output_channel_ids = [str(c) for c in output_channel_ids]

    if reference_channel_id not in output_channel_ids:
        output_channel_ids = [reference_channel_id] + list(output_channel_ids)

    missing_channels = [c for c in output_channel_ids if c not in dataset_channel_keys]
    if missing_channels:
        raise ValueError(f"Unknown channel ids: {missing_channels}")

    if output_root is None or len(str(output_root)) == 0:
        output_root = os.path.join(dataset.path, "Reconstruction", "Refinement")
    if temp_root is None or len(str(temp_root)) == 0:
        temp_root = os.path.join(output_root, "temp_block")
    if save_res_root is None or len(str(save_res_root)) == 0:
        save_res_root = os.path.join(output_root, "minus10")

    config = B85Config(
        dataset_file=dataset.file,
        dataset_path=dataset.path,
        dataset_name=dataset.name,
        output_root=output_root,
        temp_root=temp_root,
        save_res_root=save_res_root,
        reference_channel_id=reference_channel_id,
        output_channel_ids=output_channel_ids,
        start_slice=int(start_slice),
        end_slice_exclusive=int(end_slice_exclusive),
        pixel_size=float(pixel_size),
        block_size=int(block_size),
        gap=int(gap),
        selected_steps=steps,
        channels={c: channels[c] for c in output_channel_ids},
    )

    if validate:
        _validate_b85_config(config)
    return config


def _validate_b85_config(config: B85Config):
    ref = config.channels[config.reference_channel_id]
    required_images = range(config.start_slice, config.end_slice_exclusive + 1)
    missing_images = [ref.image_format.format(i) for i in required_images if not os.path.isfile(ref.image_format.format(i))]
    if missing_images and "reconstruct_image" not in config.selected_steps:
        raise FileNotFoundError(f"Missing reference channel slice image: {missing_images[0]}")

    required_transforms = range(config.start_slice, config.end_slice_exclusive)
    missing_transforms = [
        ref.transform_format.format(i)
        for i in required_transforms
        if not os.path.isfile(ref.transform_format.format(i))
    ]
    if missing_transforms and "step4" in config.selected_steps and "reconstruct_sample" not in config.selected_steps:
        raise FileNotFoundError(f"Missing reference channel slice transform: {missing_transforms[0]}")
    if "step4_channel" in config.selected_steps and "step4" not in config.selected_steps:
        brain_path = os.path.join(_brain_output_root(config), "BrainTransform", "visor_brain.txt")
        if not os.path.isfile(brain_path):
            raise FileNotFoundError(f"Missing Refinement brain transform for step4_channel: {brain_path}")


def _read_flsm_bounds(flsm_path: str):
    with open(flsm_path, "r") as file:
        doc = json.load(file)
    left = [
        float(doc["lefttop_x"]) * 1000,
        float(doc["lefttop_y"]) * 1000,
        float(doc["lefttop_z"]) * 1000,
    ]
    right = [
        float(doc["rightbottom_x"]) * 1000,
        float(doc["rightbottom_y"]) * 1000,
        float(doc["rightbottom_z"]) * 1000,
    ]
    return left, right


def _read_offsets(config: B85Config):
    with open(config.dataset_file, "r") as file:
        doc = json.load(file)
    channel_ids = [str(c["ChannelId"]) for c in doc["Channels"]]
    channel_index = channel_ids.index(str(config.reference_channel_id))
    by_slice = {}
    for item in doc["Acquisition Results"]:
        slice_id = int(item["SliceID"])
        if slice_id < config.start_slice or slice_id > config.end_slice_exclusive:
            continue
        flsm_list = item["FlsmList"]
        if channel_index >= len(flsm_list) or len(flsm_list[channel_index]) == 0:
            continue
        flsm_path = flsm_list[channel_index]
        if not os.path.isabs(flsm_path):
            flsm_path = os.path.join(config.dataset_path, flsm_path)
        by_slice[slice_id] = _read_flsm_bounds(flsm_path)

    missing = [
        i for i in range(config.start_slice, config.end_slice_exclusive + 1)
        if i not in by_slice
    ]
    if missing:
        raise FileNotFoundError(f"Missing Refinement FLSM offset source for slice {missing[0]}.")

    left = np.array([by_slice[i][0] for i in range(config.start_slice, config.end_slice_exclusive + 1)])
    right = np.array([by_slice[i][1] for i in range(config.start_slice, config.end_slice_exclusive + 1)])
    return left, right


def _calc_reference_geometry(config: B85Config):
    left_list, right_list = _read_offsets(config)
    spacing = [config.pixel_size, config.pixel_size, config.pixel_size]
    lefttop = left_list.min(axis=0)
    rightbottom = right_list.max(axis=0)
    lefttop = np.array([lefttop[0], lefttop[1], 0.0])
    ref_size = [
        int((rightbottom[0] - lefttop[0]) // spacing[0]) + config.gap,
        int((rightbottom[1] - lefttop[1]) // spacing[1]) + config.gap,
    ]
    return left_list, right_list, spacing, lefttop, ref_size


def _send(pipe, payload):
    if pipe is not None:
        pipe.send(payload)
    elif "message" in payload:
        print(payload["message"])


def _send_message(pipe, message: str):
    _send(pipe, {"message": message})


def _check_stop(pipe):
    if pipe is None:
        return False
    if pipe.poll():
        payload = pipe.recv()
        return "stop" in payload
    return False


@contextmanager
def _pushd(path: Path):
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


def _brain_output_root(config: B85Config) -> str:
    return os.path.join(
        config.output_root,
        f"BrainTrans_{config.start_slice}_{config.end_slice_exclusive - 1}",
    )


def _build_standard_pre_pipeline(config: B85Config, task_type: str):
    metadata_key = {
        "reconstruct_sample": "SliceTransform",
        "reconstruct_image": "SliceImage",
    }.get(task_type)
    if metadata_key is None:
        raise ValueError(f"Unsupported Refinement pre-step: {task_type}")

    from VISoR_Reconstruction.reconstruction_executor.generator import gen_brain_reconstruction_pipeline

    dataset = VISoRData()
    dataset.load(config.dataset_file)

    ref = config.channels[config.reference_channel_id]
    selected_slice_ids = set(range(config.start_slice, config.end_slice_exclusive + 1))
    selected_channel_names = {channel.channel_name for channel in config.channels.values()}
    selected_channel_names.add(ref.channel_name)

    pipeline_json = gen_brain_reconstruction_pipeline(
        dataset,
        output_path=config.dataset_path,
        reference_channel=ref.channel_name,
        output_pixel_size=config.pixel_size,
        internal_pixel_size=config.pixel_size,
        generate_projection=False,
        reconstruct_brain=False,
    )
    pipeline = json.loads(pipeline_json)
    filtered_tasks = {}
    for task_name, task in pipeline["tasks"].items():
        if task["type"] != task_type:
            continue
        if not task["output_targets"]:
            continue
        metadata = task["output_targets"][0].get("metadata", {})
        try:
            slice_id = int(metadata.get("SliceID", -1))
        except (TypeError, ValueError):
            continue
        channel_name = metadata.get("ChannelName")
        if slice_id not in selected_slice_ids:
            continue
        if channel_name not in selected_channel_names:
            continue
        filtered_tasks[task_name] = task

    pipeline["tasks"] = filtered_tasks
    pipeline["metadata"] = {metadata_key: pipeline.get("metadata", {}).get(metadata_key, {})}
    return pipeline


def _validate_pipeline_inputs(pipeline: dict, task_type: str):
    generated_targets = {
        output["name"]
        for task in pipeline["tasks"].values()
        for output in task.get("output_targets", [])
    }
    missing_inputs = []
    for task in pipeline["tasks"].values():
        for target in task.get("input_targets", {}).values():
            if target.get("type") == "null":
                continue
            if target.get("name") in generated_targets:
                continue
            path = target.get("path")
            if path is None or not os.path.exists(path):
                missing_inputs.append(path or target.get("name", "<unknown>"))
    if missing_inputs:
        raise FileNotFoundError(f"Missing input for {task_type}: {missing_inputs[0]}")


def _run_standard_pre_step(config: B85Config, task_type: str, pipe=None):
    from VISoR_Reconstruction.reconstruction_executor.executor import main as executor_main

    pipeline = _build_standard_pre_pipeline(config, task_type)
    filtered_tasks = pipeline["tasks"]
    if not filtered_tasks:
        _send(pipe, {"message": f"No {task_type} tasks matched the Refinement slice/channel selection."})
        return

    _validate_pipeline_inputs(pipeline, task_type)
    _send(pipe, {
        "message": "Running {} {} task(s) for Refinement.".format(len(filtered_tasks), task_type)
    })
    executor_main(json.dumps(pipeline), pipe)


def _run_reconstruct_sample(config: B85Config, pipe=None):
    _run_standard_pre_step(config, "reconstruct_sample", pipe)


def _run_reconstruct_image(config: B85Config, pipe=None):
    _run_standard_pre_step(config, "reconstruct_image", pipe)


def _run_step1_1(modules, config, ref, left_list, lefttop, ref_size, spacing):
    tasks = []
    npy_format = os.path.join(config.temp_root, "refine_{}_pars.npy")
    for i in range(config.start_slice, config.end_slice_exclusive):
        if os.path.exists(npy_format.format(i)):
            print(f"exist {npy_format.format(i)}")
            continue
        prev_index = i
        next_index = i + 1
        up_origin = left_list[prev_index - config.start_slice].copy()
        down_origin = left_list[next_index - config.start_slice].copy()
        up_origin[2] = 0
        down_origin[2] = 0
        tasks.append((
            ref.image_format.format(prev_index),
            ref.image_format.format(next_index),
            up_origin,
            down_origin,
            lefttop,
            ref_size,
            spacing,
            i,
            config.output_root,
        ))
    modules["step1_1"].step1_1_multiprocess(3, tasks)


def _run_step1_2(modules, config, ref_size):
    step1_2 = modules["step1_2"]
    tasks = []
    names = {}
    npy_format = os.path.join(config.temp_root, "refine_{}_pars.npy")
    for i in range(config.start_slice, config.end_slice_exclusive):
        if os.path.exists(npy_format.format(i)):
            print(f"exist {npy_format.format(i)}")
            continue
        temp_block_path = os.path.join(config.temp_root, f"{i}_{i + 1}")
        if not os.path.exists(temp_block_path):
            print(f"path is wrong {temp_block_path}")
            continue
        names[f"{i}_{i + 1}"] = step1_2.get_block_name(
            ref_size,
            slices_index=i,
            block_size=config.block_size,
            save_root=config.output_root,
            tempName="temp_block",
        )

    for _, value in names.items():
        for pair in value:
            folder_name = os.path.basename(os.path.dirname(pair[0]))
            filename = os.path.basename(pair[0])
            match = re.match(r"^(\d+)_([0-9]+)", filename)
            if match is None:
                continue
            row, col = int(match.group(1)), int(match.group(2))
            save_res_folder = os.path.join(config.save_res_root, folder_name)
            os.makedirs(save_res_folder, exist_ok=True)
            tasks.append((pair[0], pair[1], row, col, save_res_folder))
    step1_2.step1_2_multiprocess(40, tasks)


def _run_step1_3(modules, config, ref_size):
    step1_3 = modules["step1_3"]
    transformed_data = []
    root = config.temp_root
    txt_root = config.save_res_root
    for slice_index in range(config.start_slice, config.end_slice_exclusive):
        print(f"start {slice_index}")
        res_path = os.path.join(root, f"refine_{slice_index}_pars.npy")
        if os.path.exists(res_path):
            print(f"exists {res_path}")
            continue
        offsets = {}
        txt_block_folder = os.path.join(txt_root, f"{slice_index}_{slice_index + 1}")
        row = int(np.floor(ref_size[0] / config.block_size))
        col = int(np.floor(ref_size[1] / config.block_size))
        for i in range(row):
            for j in range(col):
                pos_path = os.path.join(txt_block_folder, f"pos_{i}_{j}.txt")
                if os.path.exists(pos_path):
                    res = step1_3.read_coordinates(pos_path)
                    offsets[(int(i), int(j))] = res[0]
        if not offsets:
            print(f"index {slice_index} is empty")
            continue

        refine_path = os.path.join(root, f"{slice_index}_{slice_index + 1}")
        os.makedirs(refine_path, exist_ok=True)
        moving_format = os.path.join(root, f"{slice_index}_{slice_index + 1}", "{}_{}down_temp_all.tif")
        moved_format = os.path.join(root, f"{slice_index}_{slice_index + 1}", "{}_{}moved.tif")
        fixed_format = os.path.join(root, f"{slice_index}_{slice_index + 1}", "{}_{}up_temp_all.tif")
        fixed_save_format = os.path.join(refine_path, "save_{}_{}up_temp_all.tif")
        moving_save_format = os.path.join(refine_path, "save_{}_{}moved.tif")
        save_loss_format = os.path.join(txt_block_folder, "loss_{}_{}.txt")
        step1_3.CalNCC(
            [4.0, 4.0, 4.0],
            moving_format,
            moved_format,
            offsets,
            fixed_format,
            moved_format,
            fixed_save_format,
            moving_save_format,
            save_loss_format,
            rate=4,
        )

        used_list = []
        count = 0
        for i in range(row):
            for j in range(col):
                loss_path = os.path.join(txt_block_folder, f"loss_{i}_{j}.txt")
                pos = (i, j)
                if os.path.exists(loss_path):
                    ncc, ssim = step1_3.read_loss(loss_path)
                    if ncc > 0.80 and ssim > 0.50:
                        count += 1
                        value = offsets[pos]
                        used_list.append(value[2])
                        print(f"key is {pos}; value is {value}; ncc: {ncc}; ssim: {ssim} ct is {count}")
        if used_list:
            npy_array = np.zeros((row, col, 3))
            step1_3.rest(used_list, offsets, slice_index, count, npy_array, 1, transformed_data, root)


def _run_step2(modules, config, ref, left_list, lefttop, ref_size, spacing):
    tasks = []
    npy_format = os.path.join(config.temp_root, "refine_{}_pars.npy")
    for slice_index in range(config.start_slice, config.end_slice_exclusive):
        img_index = slice_index + 1
        img_origin = left_list[img_index - config.start_slice]
        img_origin = [img_origin[0], img_origin[1], 0]
        tasks.append((
            npy_format.format(slice_index),
            ref.image_format.format(img_index),
            config.temp_root,
            slice_index,
            lefttop,
            img_origin,
            spacing,
            ref_size,
            config.block_size,
            int(round(_channel_wavelength({"LaserWavelength": ref.laser_wavelength}))),
            ref.name_format,
        ))
    modules["step2"].step2_multiprocess(4, tasks)


def _run_extract_surface_failed(modules, config, ref, left_list, lefttop, ref_size, spacing):
    copy_extract_surface = modules["step2"].copy_extract_surface
    uz_format = os.path.join(config.temp_root, ref.name_format + "_uz.mha")
    lz_format = os.path.join(config.temp_root, ref.name_format + "_lz.mha")
    us_format = os.path.join(config.temp_root, ref.name_format + "_us.mha")
    ls_format = os.path.join(config.temp_root, ref.name_format + "_ls.mha")
    for i in range(config.start_slice, config.end_slice_exclusive):
        uz_path = uz_format.format(i)
        lz_path = lz_format.format(i)
        if os.path.exists(uz_path) and os.path.exists(lz_path):
            print(f"{uz_path} exists")
            continue
        img_origin = left_list[i - config.start_slice]
        img_origin = [img_origin[0], img_origin[1], 0]
        img = sitk.ReadImage(ref.image_format.format(i))
        img.SetOrigin(img_origin)
        img.SetSpacing(spacing)
        img_size = img.GetSize()
        img = sitk.Resample(img, [ref_size[0], ref_size[1], img_size[2]], sitk.Transform(), sitk.sitkLinear, lefttop, spacing)
        gap = 30
        height_range = [img_size[2] - 100 - gap, img_size[2] - gap]
        img.SetOrigin([0, 0, 0])
        img.SetSpacing([1, 1, 1])
        umap_x = sitk.Image(ref_size, sitk.sitkFloat32)
        umap_y = sitk.Image(ref_size, sitk.sitkFloat32)
        uz = sitk.Compose(umap_x, umap_y, sitk.Image(ref_size, sitk.sitkFloat32) + height_range[0])
        lz = sitk.Compose(
            sitk.Image(ref_size, sitk.sitkFloat32),
            sitk.Image(ref_size, sitk.sitkFloat32),
            sitk.Image(ref_size, sitk.sitkFloat32) + height_range[1],
        )
        surfaces = copy_extract_surface(img, uz, lz)
        sitk.WriteImage(uz, uz_path)
        sitk.WriteImage(lz, lz_path)
        sitk.WriteImage(surfaces[:, :, 0], us_format.format(i))
        sitk.WriteImage(surfaces[:, :, 1], ls_format.format(i))


def _run_step3(modules, config, ref, ref_size):
    tasks = []
    expected_outputs = []
    prev_format = os.path.join(config.temp_root, ref.name_format) + "_ls.mha"
    next_format = os.path.join(config.temp_root, ref.name_format) + "_us.mha"
    for i in range(config.start_slice, config.end_slice_exclusive):
        prev_index = i
        next_index = i + 1
        save_prev_df = os.path.join(config.temp_root, ref.name_format.format(prev_index) + "_lxy.mha")
        save_next_df = os.path.join(config.temp_root, ref.name_format.format(next_index) + "_uxy.mha")
        expected_outputs.extend([save_prev_df, save_next_df])
        if os.path.exists(save_prev_df) and os.path.exists(save_next_df):
            print(f"finished : {prev_index} {next_index}")
            continue
        tasks.append((
            prev_format.format(prev_index),
            next_format.format(next_index),
            1,
            ref_size,
            None,
            None,
            save_prev_df,
            save_next_df,
            os.path.join(config.temp_root, f"{prev_index:03d}_ls_re.mha"),
            os.path.join(config.temp_root, f"{next_index:03d}_us_re.mha"),
            os.path.join(config.temp_root, f"2_{prev_index:03d}_ls_re.mha"),
            os.path.join(config.temp_root, f"2_{next_index:03d}_us_re.mha"),
        ))
    if tasks:
        modules["step3"].step3_multiprocess(8, tasks)
    missing_outputs = [path for path in expected_outputs if not os.path.exists(path)]
    if missing_outputs:
        raise RuntimeError("Refinement step3 did not generate expected files: " + ", ".join(missing_outputs))


def _run_check_xy(config, ref):
    save_prev_df = os.path.join(config.temp_root, ref.name_format.format(config.start_slice) + "_lxy.mha")
    save_next_df = os.path.join(config.temp_root, ref.name_format.format(config.start_slice) + "_uxy.mha")
    if not os.path.exists(save_next_df):
        print(save_next_df)
        if os.path.exists(save_prev_df):
            import shutil
            shutil.copy(save_prev_df, save_next_df)
    end_prev_df = os.path.join(config.temp_root, ref.name_format.format(config.end_slice_exclusive - 1) + "_lxy.mha")
    end_next_df = os.path.join(config.temp_root, ref.name_format.format(config.end_slice_exclusive - 1) + "_uxy.mha")
    if not os.path.exists(end_prev_df):
        print(end_prev_df)
        if os.path.exists(end_next_df):
            import shutil
            shutil.copy(end_next_df, end_prev_df)


def _process_transforms(modules, config, ref):
    modules["step4"].ROI_ProcessTranform(
        config.start_slice,
        config.end_slice_exclusive,
        config.temp_root,
        ref.name_format,
        config.temp_root,
    )


def _create_brain(config, ref, left_list, lefttop, output_root):
    os.makedirs(os.path.join(output_root, "BrainTransform"), exist_ok=True)
    output = os.path.join(output_root, "BrainTransform", "visor_brain.txt")
    slice_offset_list = []
    for left in left_list:
        left = left - lefttop
        slice_offset_list.append([left[0], left[1], 0])

    inputs = {}
    udf_format = os.path.join(config.temp_root, ref.name_format + "_udf.mha")
    ldf_format = os.path.join(config.temp_root, ref.name_format + "_ldf.mha")
    for index in range(config.start_slice, config.end_slice_exclusive):
        inputs[f"{index},sl"] = VISoRSample()
        inputs[f"{index},sl"].load(ref.transform_format.format(index))
        inputs[f"{index},u"] = sitk.ReadImage(udf_format.format(index))
        inputs[f"{index},l"] = sitk.ReadImage(ldf_format.format(index))

    brain = VISoRBrain()
    slices, ud, ld = {}, {}, {}
    for key, value in inputs.items():
        index = int(key.split(",")[0])
        kind = key.split(",")[1]
        if kind == "sl":
            slices[index] = value
        elif kind == "u":
            ud[index] = value
        elif kind == "l":
            ld[index] = value

    for index in ud:
        sl = slices[index]
        u = ud[index]
        l = ld[index]
        slice_offset = slice_offset_list[index - config.start_slice]
        u = sitk.Compose(
            sitk.VectorIndexSelectionCast(u, 0) * config.pixel_size + sl.sphere[0][0] - slice_offset[0],
            sitk.VectorIndexSelectionCast(u, 1) * config.pixel_size + sl.sphere[0][1] - slice_offset[1],
            sitk.VectorIndexSelectionCast(u, 2) * config.pixel_size + (sl.sphere[0][2] - (index - 1) * 400),
        )
        l = sitk.Compose(
            sitk.VectorIndexSelectionCast(l, 0) * config.pixel_size + sl.sphere[0][0] - slice_offset[0],
            sitk.VectorIndexSelectionCast(l, 1) * config.pixel_size + sl.sphere[0][1] - slice_offset[1],
            sitk.VectorIndexSelectionCast(l, 2) * config.pixel_size + (sl.sphere[0][2] - index * 400),
        )
        df = sitk.JoinSeries([u[:, :, 0], l[:, :, 0]])
        df.SetOrigin([0, 0, (index - 1) * 400])
        df.SetSpacing([config.pixel_size, config.pixel_size, 400])
        size = df.GetSize()
        df = sitk.DisplacementFieldTransform(sitk.Cast(df, sitk.sitkVectorFloat64))
        brain.slices[index] = sl
        brain.set_transform(index, df)
        brain.slice_spheres[index] = [
            [0, 0, (index - 1) * 400],
            [size[0] * config.pixel_size, size[1] * config.pixel_size, index * 400],
        ]
        brain.save(output)
        brain.release_transform(index)
    brain.calculate_sphere()
    brain.save(output)
    return output


def _generate_brain_image_task(args):
    brain_path, img_path, slice_index, pixel_size, name_format, n_start = args
    brain = VISoRBrain()
    brain.load(brain_path)
    img = sitk.ReadImage(img_path)
    img.SetOrigin(brain.slices[slice_index].sphere[0])
    img.SetSpacing([pixel_size, pixel_size, pixel_size])
    roi = brain.slice_spheres[slice_index]
    size = [int((roi[1][j] - roi[0][j]) / pixel_size) for j in range(3)]
    print(size)
    res = sitk.Resample(img, size, brain.transform(slice_index), sitk.sitkLinear, roi[0], [pixel_size, pixel_size, pixel_size])
    res.SetSpacing([j / 1000 for j in res.GetSpacing()])
    paths = [name_format.format(n_start + j) for j in range(size[2])]
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    for i in range(size[2]):
        image = sitk.GetArrayFromImage(res[:, :, i])
        image = np.left_shift(np.right_shift((image + 8), 4), 4)
        tifffile.imwrite(paths[i], image, compress=1)
    return "\n".join(paths)


def _generate_brain_images(config, channel, output_root, channel_id):
    brain_path = os.path.join(output_root, "BrainTransform", "visor_brain.txt")
    save_dir = os.path.join(output_root, "BrainImage", _pixel_dir_name(config.pixel_size))
    os.makedirs(save_dir, exist_ok=True)
    name_format = os.path.join(save_dir, "Z{:05d}_" + f"C{channel_id}.tif")
    tasks = []
    for slice_index in range(config.start_slice, config.end_slice_exclusive):
        tasks.append((
            brain_path,
            channel.image_format.format(slice_index),
            slice_index,
            config.pixel_size,
            name_format,
            int(400 / config.pixel_size) * (slice_index - 1),
        ))
    file_lists = [_generate_brain_image_task(task) for task in tasks]
    list_file = os.path.join(save_dir, f"{config.dataset_name}_C{channel_id}_files.txt")
    with open(list_file, "w") as file:
        file.write("\n".join([p for group in file_lists for p in group.splitlines()]))
    return list_file


def _write_metadata(config: B85Config, output_root: str, generated_lists: Dict[str, str]):
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "Parameters.json"), "w") as file:
        json.dump(asdict(config), file, indent=2)

    brain_transform_dir = os.path.join(output_root, "BrainTransform")
    brain_image_dir = os.path.join(output_root, "BrainImage")
    os.makedirs(brain_transform_dir, exist_ok=True)
    os.makedirs(brain_image_dir, exist_ok=True)

    brain_transform_path = os.path.join(brain_transform_dir, "BrainTransform.json")
    with open(brain_transform_path, "w") as file:
        json.dump({
            "BrainTransformInfo": {
                "Type": "BrainTransform",
                "Software": "VISOR_Reconstruction",
                "Parameter": "../Parameters.json",
                "Version": VERSION,
                "Time": time.asctime(),
            },
            "BrainTransform": {
                "visor_brain.txt": {}
            },
        }, file, indent=2)

    brain_image_entries = {}
    for channel_id, list_file in generated_lists.items():
        channel = config.channels[channel_id]
        brain_image_entries[os.path.relpath(list_file, brain_image_dir)] = {
            "ChannelName": channel.channel_name,
            "PixelSize": config.pixel_size,
            "SliceStart": config.start_slice,
            "SliceEndExclusive": config.end_slice_exclusive,
        }
    brain_image_path = os.path.join(brain_image_dir, "BrainImage.json")
    with open(brain_image_path, "w") as file:
        json.dump({
            "BrainImageInfo": {
                "Type": "BrainImage",
                "Software": "VISOR_Reconstruction",
                "Parameter": "../Parameters.json",
                "Version": VERSION,
                "Time": time.asctime(),
                "Transform": "../BrainTransform/BrainTransform.json",
            },
            "BrainImage": brain_image_entries,
        }, file, indent=2)

    summary = {
        "name": "Refinement",
        "time": time.asctime(),
        "output_root": output_root,
        "brain_transform": os.path.join(output_root, "BrainTransform", "visor_brain.txt"),
        "brain_transform_metadata": brain_transform_path,
        "brain_image_metadata": brain_image_path,
        "generated_lists": generated_lists,
        "selected_steps": config.selected_steps,
    }
    with open(os.path.join(output_root, "RunSummary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    return summary


class B85Runner:
    def __init__(self, config: B85Config, pipe=None):
        self.config = config
        self.pipe = pipe
        self.modules = None
        self.output_root = _brain_output_root(config)
        self.generated_lists = {}

    def _step(self, name, index, total, func):
        if name not in self.config.selected_steps:
            return
        if _check_stop(self.pipe):
            _send(self.pipe, {"status": "Stopped"})
            raise InterruptedError()
        _send(self.pipe, {"status": name, "message": f"[{time.asctime()}] Start Refinement {name}"})
        func()
        _send(self.pipe, {
            "progress": index / total,
            "message": f"[{time.asctime()}] Finish Refinement {name}",
        })

    def run(self):
        _send(self.pipe, {"status": "Preparing Refinement"})
        os.makedirs(self.config.output_root, exist_ok=True)
        os.makedirs(self.config.temp_root, exist_ok=True)
        os.makedirs(self.config.save_res_root, exist_ok=True)
        needs_b85_steps = any(s in B85_REFINEMENT_STEP_NAMES for s in self.config.selected_steps)
        ref = self.config.channels[self.config.reference_channel_id]
        left_list, spacing, lefttop, ref_size = None, None, None, None
        if needs_b85_steps:
            self.modules = _import_b85_modules()
        total = max(1, len([s for s in B85_STEP_NAMES if s in self.config.selected_steps]))
        current = 0

        def in_b85_root(func):
            def wrapped():
                with _pushd(_b85_root()):
                    func()
            return wrapped

        for name, func in [
            ("reconstruct_sample", lambda: _run_reconstruct_sample(self.config, self.pipe)),
            ("reconstruct_image", lambda: _run_reconstruct_image(self.config, self.pipe)),
            ("step1_1", in_b85_root(lambda: _run_step1_1(self.modules, self.config, ref, left_list, lefttop, ref_size, spacing))),
            ("step1_2", in_b85_root(lambda: _run_step1_2(self.modules, self.config, ref_size))),
            ("step1_3", in_b85_root(lambda: _run_step1_3(self.modules, self.config, ref_size))),
            ("step2", in_b85_root(lambda: _run_step2(self.modules, self.config, ref, left_list, lefttop, ref_size, spacing))),
            ("extract_surface_failed", in_b85_root(lambda: _run_extract_surface_failed(self.modules, self.config, ref, left_list, lefttop, ref_size, spacing))),
            ("step3", in_b85_root(lambda: _run_step3(self.modules, self.config, ref, ref_size))),
            ("check_xy", in_b85_root(lambda: _run_check_xy(self.config, ref))),
            ("step4", in_b85_root(lambda: self._run_step4(ref, left_list, lefttop))),
            ("step4_channel", in_b85_root(self._run_step4_channel)),
        ]:
            if name in self.config.selected_steps:
                current += 1
            if (
                name in self.config.selected_steps
                and name in B85_REFINEMENT_STEP_NAMES
                and name != "step4_channel"
                and left_list is None
            ):
                left_list, _, spacing, lefttop, ref_size = _calc_reference_geometry(self.config)
            self._step(name, current, total, func)

        summary = _write_metadata(self.config, self.output_root, self.generated_lists)
        _send(self.pipe, {"status": "Finished", "progress": 1, "result": summary})

    def _run_step4(self, ref, left_list, lefttop):
        _process_transforms(self.modules, self.config, ref)
        _create_brain(self.config, ref, left_list, lefttop, self.output_root)
        self.generated_lists[self.config.reference_channel_id] = _generate_brain_images(
            self.config,
            ref,
            self.output_root,
            self.config.reference_channel_id,
        )

    def _run_step4_channel(self):
        for channel_id in self.config.output_channel_ids:
            if channel_id == self.config.reference_channel_id:
                continue
            channel = self.config.channels[channel_id]
            self.generated_lists[channel_id] = _generate_brain_images(
                self.config,
                channel,
                self.output_root,
                channel_id,
            )


def main(config_json: str, pipe=None):
    if hasattr(__import__("multiprocessing"), "get_start_method"):
        import multiprocessing
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    config = B85Config.from_json(config_json)
    runner = B85Runner(config, pipe)
    runner.run()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m VISoR_Reconstruction.reconstruction_executor.b85_runner <config.json>")
    with open(sys.argv[1], "r", encoding="utf-8-sig") as fp:
        main(fp.read())
