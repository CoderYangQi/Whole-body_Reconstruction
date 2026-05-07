import json
import os
import re
import shutil
import sys
import time
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
    overwrite_existing: bool = False
    selected_steps: List[str] = field(default_factory=lambda: list(B85_STEP_NAMES))
    channels: Dict[str, B85ChannelConfig] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "B85Config":
        data = json.loads(text.lstrip("\ufeff"))
        data.setdefault("overwrite_existing", False)
        data["reference_channel_id"] = str(data["reference_channel_id"])
        data["output_channel_ids"] = [str(c) for c in data.get("output_channel_ids", [])]
        data["channels"] = {
            k: B85ChannelConfig(**v) for k, v in data.get("channels", {}).items()
        }
        return cls(**data)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pixel_dir_name(pixel_size: float) -> str:
    if float(pixel_size).is_integer():
        return f"{float(pixel_size):.1f}"
    return str(pixel_size)


def _channel_wavelength(channel: dict) -> float:
    value = channel.get("LaserWavelength") or channel.get("ChannelName") or ""
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else 0.0


def _prepare_b85_imports():
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _import_b85_modules():
    _prepare_b85_imports()
    import importlib

    package = "VISoR_Reconstruction.reconstruction_executor.b85_utils"
    return {
        "step1_1": importlib.import_module(f"{package}.step1_1_methods"),
        "step1_2": importlib.import_module(f"{package}.step1_2_use_block"),
        "step1_3": importlib.import_module(f"{package}.step1_3_CalLoss"),
        "step2": importlib.import_module(f"{package}.step2_extract"),
        "step3": importlib.import_module(f"{package}.step3_align"),
        "step4": importlib.import_module(f"{package}.step4_ContiuneProcessTransform"),
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
    overwrite_existing: bool = False,
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
        overwrite_existing=bool(overwrite_existing),
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
        try:
            pipe.send(payload)
        except (BrokenPipeError, EOFError, OSError):
            return False
        return True
    elif "message" in payload:
        print(payload["message"])
    return False


def _send_message(pipe, message: str):
    _send(pipe, {"message": message})


class _StandardExecutorPipe:
    def __init__(self, parent_pipe):
        self.parent_pipe = parent_pipe
        self.pending = None
        self.closed = False

    def send(self, payload):
        if self.closed:
            return
        if not _send(self.parent_pipe, payload):
            self.closed = True

    def poll(self, timeout=0):
        if self.parent_pipe is None or self.closed:
            if timeout:
                time.sleep(timeout)
            return False
        if self.pending is not None:
            return True
        try:
            if self.parent_pipe.poll(timeout):
                self.pending = self.parent_pipe.recv()
                return True
        except (BrokenPipeError, EOFError, OSError):
            self.closed = True
        return False

    def recv(self):
        if self.pending is None:
            return {}
        payload = self.pending
        self.pending = None
        return payload


def _check_stop(pipe):
    if pipe is None:
        return False
    try:
        if pipe.poll():
            payload = pipe.recv()
            return "stop" in payload
    except (BrokenPipeError, EOFError, OSError):
        return False
    return False


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
    executor_main(json.dumps(pipeline), _StandardExecutorPipe(pipe))


def _run_reconstruct_sample(config: B85Config, pipe=None):
    _run_standard_pre_step(config, "reconstruct_sample", pipe)


def _run_reconstruct_image(config: B85Config, pipe=None):
    _run_standard_pre_step(config, "reconstruct_image", pipe)


def _slice_pair_indices(config: B85Config):
    return range(config.start_slice, config.end_slice_exclusive)


def _pair_folder(root: str, slice_index: int) -> str:
    return os.path.join(root, f"{slice_index}_{slice_index + 1}")


def _is_fresh(path: str, min_mtime: Optional[float] = None) -> bool:
    if min_mtime is None:
        return True
    try:
        return os.path.getmtime(path) >= min_mtime
    except OSError:
        return False


def _matching_step1_1_blocks(folder: str, min_mtime: Optional[float] = None) -> List[str]:
    if not os.path.isdir(folder):
        return []
    result = []
    for name in os.listdir(folder):
        if not name.endswith("up_temp_all.tif"):
            continue
        down_name = name.replace("up_temp_all.tif", "down_temp_all.tif")
        up_path = os.path.join(folder, name)
        down_path = os.path.join(folder, down_name)
        if (
            os.path.isfile(down_path)
            and _is_fresh(up_path, min_mtime)
            and _is_fresh(down_path, min_mtime)
        ):
            result.append(up_path)
    return result


def _validate_step1_1_outputs(config: B85Config, min_mtime: Optional[float] = None):
    missing = []
    for slice_index in _slice_pair_indices(config):
        folder = _pair_folder(config.temp_root, slice_index)
        if not _matching_step1_1_blocks(folder, min_mtime):
            missing.append(folder)
    if missing:
        raise RuntimeError("Refinement step1_1 did not generate block pairs: " + missing[0])


def _validate_step1_2_outputs(config: B85Config, min_mtime: Optional[float] = None):
    missing = []
    for slice_index in _slice_pair_indices(config):
        folder = _pair_folder(config.save_res_root, slice_index)
        if not os.path.isdir(folder) or not any(
            name.startswith("pos_")
            and name.endswith(".txt")
            and _is_fresh(os.path.join(folder, name), min_mtime)
            for name in os.listdir(folder)
        ):
            missing.append(folder)
    if missing:
        raise RuntimeError("Refinement step1_2 did not generate position results: " + missing[0])


def _validate_step1_3_outputs(config: B85Config, min_mtime: Optional[float] = None):
    missing = []
    for slice_index in _slice_pair_indices(config):
        path = os.path.join(config.temp_root, f"refine_{slice_index}_pars.npy")
        if not os.path.isfile(path) or not _is_fresh(path, min_mtime):
            missing.append(path)
    if missing:
        raise RuntimeError("Refinement step1_3 did not generate refine parameters: " + missing[0])


def _clear_path(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    except OSError as exc:
        print(f"Cannot delete existing Refinement output, will overwrite if possible: {path} ({exc})")


def _clear_step1_1_outputs(config: B85Config):
    for slice_index in _slice_pair_indices(config):
        _clear_path(_pair_folder(config.temp_root, slice_index))


def _clear_step1_2_outputs(config: B85Config):
    for slice_index in _slice_pair_indices(config):
        _clear_path(_pair_folder(config.save_res_root, slice_index))


def _clear_step1_3_outputs(config: B85Config):
    for slice_index in _slice_pair_indices(config):
        _clear_path(os.path.join(config.temp_root, f"refine_{slice_index}_pars.npy"))
        _clear_path(os.path.join(config.temp_root, f"{slice_index}_np_array.npy"))
        refine_folder = _pair_folder(config.temp_root, slice_index)
        if os.path.isdir(refine_folder):
            for name in os.listdir(refine_folder):
                if name.endswith("moved.tif") or name.startswith("save_"):
                    _clear_path(os.path.join(refine_folder, name))
        txt_block_folder = _pair_folder(config.save_res_root, slice_index)
        if os.path.isdir(txt_block_folder):
            for name in os.listdir(txt_block_folder):
                if (
                    name.startswith("loss_") and name.endswith(".txt")
                ) or name == "step1_3_block_status.txt":
                    _clear_path(os.path.join(txt_block_folder, name))


def _read_image_with_tifffile_fallback(path: str) -> sitk.Image:
    try:
        return sitk.ReadImage(path)
    except RuntimeError:
        if not str(path).lower().endswith((".tif", ".tiff")):
            raise
        array = _read_tiff_stack_with_page_fallback(path)
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        return sitk.GetImageFromArray(array)


def _read_tiff_stack_with_page_fallback(path: str):
    arrays = []
    last_valid = None
    expected_shape = None
    with tifffile.TiffFile(path) as tif:
        for page_index, page in enumerate(tif.pages):
            try:
                array = page.asarray()
            except Exception as exc:
                if last_valid is None:
                    raise
                print(
                    "TIFF page {} could not be read from {}; using previous page. {}".format(
                        page_index,
                        path,
                        exc,
                    )
                )
                array = last_valid.copy()
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                if last_valid is None:
                    raise RuntimeError(
                        "TIFF page {} has unexpected shape {} in {}; expected {}".format(
                            page_index,
                            array.shape,
                            path,
                            expected_shape,
                        )
                    )
                print(
                    "TIFF page {} has unexpected shape {} in {}; using previous page.".format(
                        page_index,
                        array.shape,
                        path,
                    )
                )
                array = last_valid.copy()
            last_valid = array
            arrays.append(array)
    if not arrays:
        raise RuntimeError(f"No readable TIFF pages: {path}")
    return np.stack(arrays, axis=0)


def _read_tiff_page_with_fallback(path: str, page_index: int):
    with tifffile.TiffFile(path) as tif:
        depth = len(tif.pages)
        if depth <= 0:
            raise RuntimeError(f"No TIFF pages: {path}")
        page_index = max(0, min(page_index, depth - 1))
        first_error = None
        candidates = [page_index]
        for distance in range(1, min(depth, 16)):
            candidates.extend([page_index - distance, page_index + distance])
        for candidate in candidates:
            if candidate < 0 or candidate >= depth:
                continue
            try:
                array = tif.pages[candidate].asarray()
                if array.ndim == 2:
                    return array, depth, candidate
                first_error = RuntimeError(
                    f"TIFF page {candidate} has unsupported shape {array.shape}: {path}"
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        raise RuntimeError(f"No readable 2D TIFF page near {page_index}: {path}")


def _resample_tiff_page_2d(path: str, page_index: int, img_origin, lefttop, spacing, ref_size):
    array, depth, used_index = _read_tiff_page_with_fallback(path, page_index)
    if used_index != page_index:
        print(f"TIFF page {page_index} could not be used from {path}; using page {used_index}.")
    img = sitk.GetImageFromArray(array)
    img.SetOrigin([float(img_origin[0]), float(img_origin[1])])
    img.SetSpacing([float(spacing[0]), float(spacing[1])])
    return sitk.Resample(
        img,
        [int(ref_size[0]), int(ref_size[1])],
        sitk.Transform(),
        sitk.sitkLinear,
        [float(lefttop[0]), float(lefttop[1])],
        [float(spacing[0]), float(spacing[1])],
    ), depth


def _run_step1_1(modules, config, ref, left_list, lefttop, ref_size, spacing):
    if config.overwrite_existing:
        _clear_step1_1_outputs(config)
    tasks = []
    temp_parent = os.path.dirname(os.path.normpath(config.temp_root))
    temp_name = os.path.basename(os.path.normpath(config.temp_root))
    for i in range(config.start_slice, config.end_slice_exclusive):
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
            temp_parent,
            temp_name,
            config.block_size,
        ))
    modules["step1_1"].step1_1_multiprocess(1, tasks)
    for i in range(config.start_slice, config.end_slice_exclusive):
        if not modules["step1_1"].step1_1_task_complete(temp_parent, temp_name, i, ref_size, config.block_size):
            raise RuntimeError(
                "Refinement step1_1 block status file is incomplete: "
                + os.path.join(config.temp_root, f"{i}_{i + 1}", "step1_1_block_status.txt")
            )
    _validate_step1_1_outputs(config)


def _run_step1_2(modules, config, ref_size):
    if config.overwrite_existing:
        _clear_step1_2_outputs(config)
    step1_2 = modules["step1_2"]
    tasks = []
    expected_by_folder = {}
    temp_parent = os.path.dirname(os.path.normpath(config.temp_root))
    temp_name = os.path.basename(os.path.normpath(config.temp_root))
    row_count = int(np.floor(ref_size[0] / config.block_size))
    col_count = int(np.floor(ref_size[1] / config.block_size))
    all_blocks = [(row, col) for row in range(row_count) for col in range(col_count)]
    for i in range(config.start_slice, config.end_slice_exclusive):
        temp_block_path = os.path.join(config.temp_root, f"{i}_{i + 1}")
        if not os.path.exists(temp_block_path):
            raise FileNotFoundError(f"Refinement step1_2 requires step1_1 block output: {temp_block_path}")
        block_pairs = step1_2.get_block_name(
            ref_size,
            slices_index=i,
            block_size=config.block_size,
            save_root=temp_parent,
            tempName=temp_name,
        )
        folder_name = f"{i}_{i + 1}"
        save_res_folder = os.path.join(config.save_res_root, folder_name)
        os.makedirs(save_res_folder, exist_ok=True)
        expected_blocks = set()
        parsed_pairs = []
        for pair in block_pairs:
            folder_name = os.path.basename(os.path.dirname(pair[0]))
            filename = os.path.basename(pair[0])
            match = re.match(r"^(\d+)_([0-9]+)", filename)
            if match is None:
                continue
            row, col = int(match.group(1)), int(match.group(2))
            expected_blocks.add((row, col))
            parsed_pairs.append((pair, row, col))

        step1_2.bootstrap_step1_2_status(save_res_folder, expected_blocks, all_blocks)
        expected_by_folder[save_res_folder] = expected_blocks
        for pair, row, col in parsed_pairs:
            if step1_2.step1_2_block_complete(save_res_folder, row, col):
                continue
            tasks.append((pair[0], pair[1], row, col, save_res_folder))

    step1_2.step1_2_multiprocess(40, tasks)
    for save_res_folder, expected_blocks in expected_by_folder.items():
        incomplete = step1_2.finalize_step1_2_status(
            save_res_folder,
            expected_blocks,
            all_blocks,
            mark_missing_failed=True,
        )
        if incomplete:
            preview = ", ".join(
                "{}_{}:{} pos_ready={}".format(
                    block["row"],
                    block["col"],
                    block["status"],
                    block["pos_ready"],
                )
                for block in incomplete[:20]
            )
            raise RuntimeError(
                "Refinement step1_2 block status file is incomplete: "
                + os.path.join(save_res_folder, "step1_2_block_status.txt")
                + (f" ({preview})" if preview else "")
            )
    _validate_step1_2_outputs(config)


def _run_step1_3(modules, config, ref_size):
    if config.overwrite_existing:
        _clear_step1_3_outputs(config)
    step1_3 = modules["step1_3"]
    transformed_data = []
    root = config.temp_root
    txt_root = config.save_res_root
    for slice_index in range(config.start_slice, config.end_slice_exclusive):
        print(f"start {slice_index}")
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
            raise RuntimeError(f"Refinement step1_3 has no step1_2 positions for slice {slice_index}.")

        refine_path = os.path.join(root, f"{slice_index}_{slice_index + 1}")
        os.makedirs(refine_path, exist_ok=True)
        moving_format = os.path.join(root, f"{slice_index}_{slice_index + 1}", "{}_{}down_temp_all.tif")
        moved_format = os.path.join(root, f"{slice_index}_{slice_index + 1}", "{}_{}moved.tif")
        fixed_format = os.path.join(root, f"{slice_index}_{slice_index + 1}", "{}_{}up_temp_all.tif")
        fixed_save_format = os.path.join(refine_path, "save_{}_{}up_temp_all.tif")
        moving_save_format = os.path.join(refine_path, "save_{}_{}moved.tif")
        save_loss_format = os.path.join(txt_block_folder, "loss_{}_{}.txt")
        step1_3.bootstrap_step1_3_status(
            txt_block_folder,
            offsets,
            moved_format,
            fixed_save_format,
            moving_save_format,
            save_loss_format,
        )
        refine_pars_path = os.path.join(root, f"refine_{slice_index}_pars.npy")
        np_array_path = os.path.join(root, f"{slice_index}_np_array.npy")
        if (
            not config.overwrite_existing
            and os.path.isfile(refine_pars_path)
            and os.path.isfile(np_array_path)
            and step1_3.step1_3_pair_complete(
                txt_block_folder,
                offsets,
                moved_format,
                fixed_save_format,
                moving_save_format,
                save_loss_format,
            )
        ):
            print(f"Refinement step1_3 slice {slice_index} already complete from status file.")
            continue
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
            status_folder=txt_block_folder,
            force=config.overwrite_existing,
        )
        step1_3.bootstrap_step1_3_status(
            txt_block_folder,
            offsets,
            moved_format,
            fixed_save_format,
            moving_save_format,
            save_loss_format,
        )
        if not step1_3.step1_3_pair_complete(
            txt_block_folder,
            offsets,
            moved_format,
            fixed_save_format,
            moving_save_format,
            save_loss_format,
        ):
            raise RuntimeError(
                "Refinement step1_3 block status file is incomplete: "
                + os.path.join(txt_block_folder, "step1_3_block_status.txt")
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
        print(
            "Refinement step1_3 slice {} usable blocks: {} / {}".format(
                slice_index,
                count,
                len(offsets),
            )
        )
        if used_list:
            npy_array = np.zeros((row, col, 3))
            step1_3.rest(used_list, offsets, slice_index, count, npy_array, 1, transformed_data, root)
        else:
            raise RuntimeError(f"Refinement step1_3 did not find usable blocks for slice {slice_index}.")
    _validate_step1_3_outputs(config)


def _run_step2(modules, config, ref, left_list, lefttop, ref_size, spacing):
    tasks = []
    expected_outputs = []
    npy_format = os.path.join(config.temp_root, "refine_{}_pars.npy")
    failed_marker = os.path.join(config.temp_root, "step2_failed_slices.txt")
    if config.overwrite_existing:
        _clear_path(failed_marker)
    failed_slices = set()
    if os.path.isfile(failed_marker):
        with open(failed_marker, "r", encoding="utf-8-sig", errors="replace") as file:
            for line in file:
                parts = line.strip().split("\t")
                if parts and parts[0].isdigit():
                    failed_slices.add(int(parts[0]))
    for slice_index in range(config.start_slice, config.end_slice_exclusive):
        img_index = slice_index + 1
        img_origin = left_list[img_index - config.start_slice]
        img_origin = [img_origin[0], img_origin[1], 0]
        output_stem = os.path.join(config.temp_root, ref.name_format.format(img_index))
        outputs = [
            output_stem + "_uz.mha",
            output_stem + "_lz.mha",
            output_stem + "_us.mha",
            output_stem + "_ls.mha",
        ]
        expected_outputs.extend(outputs)
        if not config.overwrite_existing and all(os.path.exists(path) for path in outputs):
            print(f"Refinement step2 slice {img_index} already complete.")
            continue
        if (
            not config.overwrite_existing
            and img_index in failed_slices
            and "extract_surface_failed" in config.selected_steps
        ):
            print(f"Refinement step2 slice {img_index} previously failed; extract_surface_failed will rebuild it.")
            continue
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
    failures = modules["step2"].step2_multiprocess(4, tasks)
    if failures:
        with open(failed_marker, "a", encoding="utf-8") as file:
            for slice_index, reason in failures:
                file.write(f"{slice_index}\t{reason}\t{time.asctime()}\n")
    missing_outputs = [path for path in expected_outputs if not os.path.exists(path)]
    if missing_outputs:
        if "extract_surface_failed" in config.selected_steps:
            print(
                "Refinement step2 has missing outputs; extract_surface_failed will handle: "
                + ", ".join(missing_outputs)
            )
        else:
            raise RuntimeError("Refinement step2 did not generate expected files: " + ", ".join(missing_outputs))


def _run_extract_surface_failed(modules, config, ref, left_list, lefttop, ref_size, spacing):
    copy_extract_surface = modules["step2"].copy_extract_surface
    uz_format = os.path.join(config.temp_root, ref.name_format + "_uz.mha")
    lz_format = os.path.join(config.temp_root, ref.name_format + "_lz.mha")
    us_format = os.path.join(config.temp_root, ref.name_format + "_us.mha")
    ls_format = os.path.join(config.temp_root, ref.name_format + "_ls.mha")
    expected_outputs = []
    for i in range(config.start_slice, config.end_slice_exclusive + 1):
        uz_path = uz_format.format(i)
        lz_path = lz_format.format(i)
        us_path = us_format.format(i)
        ls_path = ls_format.format(i)
        expected_outputs.extend([uz_path, lz_path, us_path, ls_path])
        if (
            not config.overwrite_existing
            and os.path.exists(uz_path)
            and os.path.exists(lz_path)
            and os.path.exists(us_path)
            and os.path.exists(ls_path)
        ):
            print(f"{uz_path} exists")
            continue
        img_origin = left_list[i - config.start_slice]
        img_origin = [img_origin[0], img_origin[1], 0]
        img_path = ref.image_format.format(i)
        gap = 30
        if str(img_path).lower().endswith((".tif", ".tiff")):
            _, depth, _ = _read_tiff_page_with_fallback(img_path, 0)
            height_range = [depth - 100 - gap, depth - gap]
            upper_index = int(np.clip(round(height_range[0]), 0, depth - 1))
            lower_index = int(np.clip(round(height_range[1]), 0, depth - 1))
            us = _resample_tiff_page_2d(
                img_path,
                upper_index,
                img_origin,
                lefttop,
                spacing,
                ref_size,
            )[0]
            ls = _resample_tiff_page_2d(
                img_path,
                lower_index,
                img_origin,
                lefttop,
                spacing,
                ref_size,
            )[0]
        else:
            img = _read_image_with_tifffile_fallback(img_path)
            img.SetOrigin(img_origin)
            img.SetSpacing(spacing)
            img_size = img.GetSize()
            img = sitk.Resample(
                img,
                [ref_size[0], ref_size[1], img_size[2]],
                sitk.Transform(),
                sitk.sitkLinear,
                lefttop,
                spacing,
            )
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
            us = surfaces[:, :, 0]
            ls = surfaces[:, :, 1]
        umap_x = sitk.Image(ref_size, sitk.sitkFloat32)
        umap_y = sitk.Image(ref_size, sitk.sitkFloat32)
        uz = sitk.Compose(umap_x, umap_y, sitk.Image(ref_size, sitk.sitkFloat32) + height_range[0])
        lz = sitk.Compose(
            sitk.Image(ref_size, sitk.sitkFloat32),
            sitk.Image(ref_size, sitk.sitkFloat32),
            sitk.Image(ref_size, sitk.sitkFloat32) + height_range[1],
        )
        sitk.WriteImage(uz, uz_path)
        sitk.WriteImage(lz, lz_path)
        sitk.WriteImage(us, us_path)
        sitk.WriteImage(ls, ls_path)
    missing_outputs = [path for path in expected_outputs if not os.path.exists(path)]
    if missing_outputs:
        raise RuntimeError("Refinement extract_surface_failed did not generate expected files: " + ", ".join(missing_outputs))


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
    img = _read_image_with_tifffile_fallback(img_path)
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


def _write_freesia_input(config, channel, save_dir, channel_id, image_paths):
    if not image_paths:
        return None

    def image_index(path):
        match = re.search(r"Z(\d+)_C", os.path.basename(path))
        if match is None:
            return 0
        return int(match.group(1))

    image_paths = sorted(image_paths, key=image_index)
    first_image = tifffile.imread(image_paths[0])
    height, width = first_image.shape[-2], first_image.shape[-1]
    doc = {
        "group_size": max(1, int(round(400 / config.pixel_size))),
        "image_path": os.path.basename(save_dir),
        "images": [],
        "pixel_size": config.pixel_size,
        "slide_thickness": config.pixel_size,
        "version": "1.1.2",
    }
    for path in image_paths:
        doc["images"].append({
            "index": image_index(path),
            "height": int(height),
            "width": int(width),
            "file_name": os.path.basename(path),
        })

    freesia_path = os.path.join(
        os.path.dirname(save_dir),
        f"freesia_{_pixel_dir_name(config.pixel_size)}_C{channel_id}_{channel.channel_name}.json",
    )
    with open(freesia_path, "w") as file:
        json.dump(doc, file, indent=2)
    return freesia_path


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
    image_paths = [p for group in file_lists for p in group.splitlines()]
    list_file = os.path.join(save_dir, f"{config.dataset_name}_C{channel_id}_files.txt")
    with open(list_file, "w") as file:
        file.write("\n".join(image_paths))
    _write_freesia_input(config, channel, save_dir, channel_id, image_paths)
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
    freesia_entries = {}
    for channel_id, list_file in generated_lists.items():
        channel = config.channels[channel_id]
        brain_image_entries[os.path.relpath(list_file, brain_image_dir)] = {
            "ChannelName": channel.channel_name,
            "PixelSize": config.pixel_size,
            "SliceStart": config.start_slice,
            "SliceEndExclusive": config.end_slice_exclusive,
        }
        freesia_file = os.path.join(
            brain_image_dir,
            f"freesia_{_pixel_dir_name(config.pixel_size)}_C{channel_id}_{channel.channel_name}.json",
        )
        if os.path.isfile(freesia_file):
            freesia_entries[os.path.relpath(freesia_file, brain_image_dir)] = {
                "ChannelName": channel.channel_name,
                "PixelSize": config.pixel_size,
            }
    brain_image_path = os.path.join(brain_image_dir, "BrainImage.json")
    with open(brain_image_path, "w") as file:
        payload = {
            "BrainImageInfo": {
                "Type": "BrainImage",
                "Software": "VISOR_Reconstruction",
                "Parameter": "../Parameters.json",
                "Version": VERSION,
                "Time": time.asctime(),
                "Transform": "../BrainTransform/BrainTransform.json",
            },
            "BrainImage": brain_image_entries,
        }
        if freesia_entries:
            payload["FreesiaFile"] = freesia_entries
        json.dump(payload, file, indent=2)

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

        for name, func in [
            ("reconstruct_sample", lambda: _run_reconstruct_sample(self.config, self.pipe)),
            ("reconstruct_image", lambda: _run_reconstruct_image(self.config, self.pipe)),
            ("step1_1", lambda: _run_step1_1(self.modules, self.config, ref, left_list, lefttop, ref_size, spacing)),
            ("step1_2", lambda: _run_step1_2(self.modules, self.config, ref_size)),
            ("step1_3", lambda: _run_step1_3(self.modules, self.config, ref_size)),
            ("step2", lambda: _run_step2(self.modules, self.config, ref, left_list, lefttop, ref_size, spacing)),
            ("extract_surface_failed", lambda: _run_extract_surface_failed(self.modules, self.config, ref, left_list, lefttop, ref_size, spacing)),
            ("step3", lambda: _run_step3(self.modules, self.config, ref, ref_size)),
            ("check_xy", lambda: _run_check_xy(self.config, ref)),
            ("step4", lambda: self._run_step4(ref, left_list, lefttop)),
            ("step4_channel", self._run_step4_channel),
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
