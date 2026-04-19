# generate_captions.py

import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info,
)


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not kart_objects:
        return []

    ego = next((kart for kart in kart_objects if kart["is_center_kart"]), kart_objects[0])
    others = [kart for kart in kart_objects if not kart["is_center_kart"]]
    track_name = extract_track_info(info_path)

    captions = [
        f"{ego['kart_name']} is the ego car.",
        f"There are {len(kart_objects)} karts in the scenario.",
        f"The track is {track_name}.",
    ]

    for kart in others:
        dx = kart["center"][0] - ego["center"][0]
        dy = kart["center"][1] - ego["center"][1]

        if abs(dx) <= 2:
            lr = "center"
        else:
            lr = "left" if dx < 0 else "right"

        if abs(dy) <= 2:
            fb = "level with"
        else:
            fb = "in front of" if dy < 0 else "behind"

        if lr == "center":
            relation = fb
        elif fb == "level with":
            relation = f"to the {lr} of"
        else:
            relation = f"{fb} and to the {lr} of"

        captions.append(f"{kart['kart_name']} is {relation} the ego car.")

    return captions


def _iter_view_indices(info_path: Path) -> list[int]:
    base_name = info_path.stem.replace("_info", "")
    images = sorted(info_path.parent.glob(f"{base_name}_*_im.jpg"))
    view_indices = []
    for image_file in images:
        _, view_index = extract_frame_info(str(image_file))
        view_indices.append(view_index)
    return sorted(set(view_indices))


def write_captions(split: str = "train", data_dir: str | None = None, img_width: int = 150, img_height: int = 100):
    if data_dir is None:
        data_root = Path(__file__).resolve().parent.parent / "data"
    else:
        data_root = Path(data_dir)

    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    info_files = sorted(split_dir.glob("*_info.json"))
    total_captions = 0

    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        caption_pairs = []
        for view_index in _iter_view_indices(info_file):
            image_rel = f"{split}/{base_name}_{view_index:02d}_im.jpg"
            for caption in generate_caption(str(info_file), view_index, img_width, img_height):
                caption_pairs.append({"image_file": image_rel, "caption": caption})

        out_file = split_dir / f"{base_name}_captions.json"
        with open(out_file, "w") as f:
            json.dump(caption_pairs, f, indent=2)
        total_captions += len(caption_pairs)
        print(f"Wrote {len(caption_pairs)} captions to {out_file}")

    print(f"Done. Generated {total_captions} captions for split '{split}'.")


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0
   python generate_captions.py write --split train
"""


def main():
    fire.Fire({"check": check_caption, "write": write_captions})


if __name__ == "__main__":
    main()