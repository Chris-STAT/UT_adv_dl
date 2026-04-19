
import json
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.
    """
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)

    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        color = (255, 0, 0) if track_id == 0 else COLORS.get(class_id, (255, 255, 255))
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    return np.array(pil_image)


def _load_info(info_path: str) -> dict[str, Any]:
    with open(info_path) as f:
        return json.load(f)


def _clean_name(name: Any) -> str:
    if name is None:
        return "unknown"
    if not isinstance(name, str):
        name = str(name)
    return name.replace("_", " ").replace("-", " ").strip()


def _title_name(name: Any) -> str:
    cleaned = _clean_name(name)
    return cleaned.title() if cleaned else "Unknown"


def _search_first(obj: Any, target_keys: set[str]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in target_keys and isinstance(v, (str, int, float)):
                return v
        for v in obj.values():
            result = _search_first(v, target_keys)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _search_first(item, target_keys)
            if result is not None:
                return result
    return None


def _collect_kart_name_map(obj: Any, mapping: dict[int, str] | None = None) -> dict[int, str]:
    if mapping is None:
        mapping = {}

    if isinstance(obj, dict):
        lowered = {str(k).lower(): v for k, v in obj.items()}
        id_value = None
        for id_key in ("track_id", "instance_id", "id", "kart_id", "player_id", "index"):
            if id_key in lowered and isinstance(lowered[id_key], (int, float, str)):
                try:
                    id_value = int(lowered[id_key])
                    break
                except (TypeError, ValueError):
                    pass

        name_value = None
        for name_key in ("kart_name", "name", "kart", "label", "character"):
            if name_key in lowered and isinstance(lowered[name_key], str):
                name_value = lowered[name_key]
                break

        if id_value is not None and name_value:
            mapping[id_value] = _title_name(name_value)

        for key, value in obj.items():
            if isinstance(value, dict):
                if key.isdigit():
                    maybe_name = None
                    if isinstance(value, dict):
                        lowered_inner = {k.lower(): v for k, v in value.items()}
                        for name_key in ("kart_name", "name", "kart", "label", "character"):
                            if name_key in lowered_inner and isinstance(lowered_inner[name_key], str):
                                maybe_name = lowered_inner[name_key]
                                break
                    if maybe_name:
                        mapping[int(key)] = _title_name(maybe_name)
                _collect_kart_name_map(value, mapping)
            elif isinstance(value, list):
                _collect_kart_name_map(value, mapping)
    elif isinstance(obj, list):
        for item in obj:
            _collect_kart_name_map(item, mapping)

    return mapping


def _extract_ego_id(info: dict[str, Any]) -> int | None:
    candidate = _search_first(
        info,
        {
            "ego_id",
            "ego_kart_id",
            "player_id",
            "current_kart_id",
            "self_id",
            "agent_id",
        },
    )
    if candidate is None:
        return None
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    info = _load_info(info_path)
    detections = info.get("detections", [])
    if view_index >= len(detections):
        return []

    frame_detections = detections[view_index]
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    image_center = (img_width / 2.0, img_height / 2.0)

    kart_name_map = _collect_kart_name_map(info)
    ego_id = _extract_ego_id(info)

    karts = []
    for detection in frame_detections:
        if len(detection) < 6:
            continue

        class_id, track_id, x1, y1, x2, y2 = detection[:6]
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = float(x1) * scale_x
        y1_scaled = float(y1) * scale_y
        x2_scaled = float(x2) * scale_x
        y2_scaled = float(y2) * scale_y

        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        center_x = (x1_scaled + x2_scaled) / 2.0
        center_y = (y1_scaled + y2_scaled) / 2.0
        kart_name = kart_name_map.get(track_id, f"Kart {track_id}")

        karts.append(
            {
                "instance_id": track_id,
                "kart_name": _title_name(kart_name),
                "center": (center_x, center_y),
                "bbox": (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                "distance_to_center": float((center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2),
                "is_center_kart": False,
            }
        )

    if not karts:
        return []

    if ego_id is not None:
        found = False
        for kart in karts:
            if kart["instance_id"] == ego_id:
                kart["is_center_kart"] = True
                found = True
                break
        if not found:
            min(karts, key=lambda k: k["distance_to_center"])["is_center_kart"] = True
    else:
        min(karts, key=lambda k: k["distance_to_center"])["is_center_kart"] = True

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    info = _load_info(info_path)

    candidate = _search_first(
        info,
        {
            "track",
            "track_name",
            "trackname",
            "course",
            "course_name",
            "map_name",
            "arena",
            "level",
        },
    )

    if candidate is None:
        return "unknown"

    return _title_name(candidate)


def _left_right_answer(dx: float, threshold: float = 2.0) -> str:
    if abs(dx) <= threshold:
        return "same"
    return "left" if dx < 0 else "right"


def _front_behind_answer(dy: float, threshold: float = 2.0) -> str:
    if abs(dy) <= threshold:
        return "same"
    return "front" if dy < 0 else "behind"


def _relative_position(dx: float, dy: float, threshold: float = 2.0) -> str:
    horiz = _left_right_answer(dx, threshold)
    vert = _front_behind_answer(dy, threshold)

    if horiz == "same" and vert == "same":
        return "same position"
    if horiz == "same":
        return "in front of the ego car" if vert == "front" else "behind the ego car"
    if vert == "same":
        return f"to the {horiz} of the ego car"
    return f"{vert}-{horiz}"


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not kart_objects:
        return []

    ego = next((kart for kart in kart_objects if kart["is_center_kart"]), kart_objects[0])
    others = [kart for kart in kart_objects if kart["instance_id"] != ego["instance_id"]]
    track_name = extract_track_info(info_path)

    qa_pairs = []

    qa_pairs.append({"question": "What kart is the ego car?", "answer": ego["kart_name"]})
    qa_pairs.append({"question": "How many karts are there in the scenario?", "answer": str(len(kart_objects))})
    qa_pairs.append({"question": "What track is this?", "answer": track_name})

    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0

    for kart in others:
        dx = kart["center"][0] - ego["center"][0]
        dy = kart["center"][1] - ego["center"][1]

        lr = _left_right_answer(dx)
        fb = _front_behind_answer(dy)
        rel = _relative_position(dx, dy)

        if lr == "left":
            left_count += 1
        elif lr == "right":
            right_count += 1

        if fb == "front":
            front_count += 1
        elif fb == "behind":
            behind_count += 1

        qa_pairs.append(
            {
                "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                "answer": lr,
            }
        )
        qa_pairs.append(
            {
                "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                "answer": fb,
            }
        )
        qa_pairs.append(
            {
                "question": f"Where is {kart['kart_name']} relative to the ego car?",
                "answer": rel,
            }
        )

    qa_pairs.append({"question": "How many karts are to the left of the ego car?", "answer": str(left_count)})
    qa_pairs.append({"question": "How many karts are to the right of the ego car?", "answer": str(right_count)})
    qa_pairs.append({"question": "How many karts are in front of the ego car?", "answer": str(front_count)})
    qa_pairs.append({"question": "How many karts are behind the ego car?", "answer": str(behind_count)})

    return qa_pairs


def _iter_view_indices(info_path: Path) -> list[int]:
    base_name = info_path.stem.replace("_info", "")
    images = sorted(info_path.parent.glob(f"{base_name}_*_im.jpg"))
    view_indices = []
    for image_file in images:
        _, view_index = extract_frame_info(str(image_file))
        view_indices.append(view_index)
    return sorted(set(view_indices))


def write_qa_pairs(split: str = "train", data_dir: str | None = None, img_width: int = 150, img_height: int = 100):
    if data_dir is None:
        data_root = Path(__file__).resolve().parent.parent / "data"
    else:
        data_root = Path(data_dir)

    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    info_files = sorted(split_dir.glob("*_info.json"))
    total_pairs = 0

    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        qa_pairs = []

        for view_index in _iter_view_indices(info_file):
            image_rel = f"{split}/{base_name}_{view_index:02d}_im.jpg"
            generated = generate_qa_pairs(str(info_file), view_index, img_width, img_height)
            for qa in generated:
                qa_pairs.append(
                    {
                        "image_file": image_rel,
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )

        out_file = split_dir / f"{base_name}_qa_pairs.json"
        with open(out_file, "w") as f:
            json.dump(qa_pairs, f, indent=2)

        total_pairs += len(qa_pairs)
        print(f"Wrote {len(qa_pairs)} QA pairs to {out_file}")

    print(f"Done. Generated {total_pairs} QA pairs for split '{split}'.")


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    qa_pairs = generate_qa_pairs(info_file, view_index)

    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0
   python generate_qa.py write --split train
"""


def main():
    fire.Fire({"check": check_qa_pairs, "write": write_qa_pairs})


if __name__ == "__main__":
    main()