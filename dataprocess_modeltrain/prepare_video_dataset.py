import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}
DEFAULT_CROP = (210, 270, 750, 810)  # x1, y1, x2, y2


def iter_video_files(dataset_dir: Path) -> list[Path]:
    videos = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return videos


def parse_crop(values: list[int]) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("crop must contain exactly 4 integers: x1 y1 x2 y2")
    x1, y1, x2, y2 = values
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("invalid crop range, expected x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def process_video(
    video_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    crop: tuple[int, int, int, int],
    image_ext: str,
    frame_step: int,
) -> int:
    class_name = video_path.parent.relative_to(dataset_dir).parts[0]
    class_output_dir = output_dir / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    x1, y1, x2, y2 = crop
    saved_count = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        height, width = frame.shape[:2]
        if x2 > width or y2 > height:
            capture.release()
            raise RuntimeError(
                f"crop {crop} exceeds frame size {(width, height)} in {video_path}"
            )

        if frame_index % frame_step == 0:
            cropped = frame[y1:y2, x1:x2]
            output_name = f"{video_path.stem}_frame_{frame_index:06d}{image_ext}"
            output_path = class_output_dir / output_name

            if not cv2.imwrite(str(output_path), cropped):
                capture.release()
                raise RuntimeError(f"failed to save frame: {output_path}")

            saved_count += 1
        frame_index += 1

    capture.release()
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert videos in datasets/ to cropped image dataset by class folder"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/mnt/hdd16t0/dataset/rps_dataset/datasets",
        help="Input dataset root directory containing class subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/hdd16t0/dataset/rps_dataset/processed_dataset",
        help="Output dataset root directory for cropped frames",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        default=DEFAULT_CROP,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop region in pixel coordinates: x1 y1 x2 y2",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".png",
        choices=[".png", ".jpg", ".jpeg", ".bmp"],
        help="Image extension for saved frames",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=3,
        help="Save one frame every N frames. Use 1 to save every frame.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    crop = parse_crop(list(args.crop))
    if args.frame_step <= 0:
        raise RuntimeError(f"frame_step must be >= 1, got: {args.frame_step}")

    if not dataset_dir.exists():
        raise RuntimeError(f"dataset directory not found: {dataset_dir}")

    videos = iter_video_files(dataset_dir)
    if not videos:
        raise RuntimeError(f"no video files found under: {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    print(f"Found {len(videos)} videos under {dataset_dir}")
    print(f"Crop region: {crop}")
    print(f"Output dir : {output_dir}")
    print(f"Frame step : {args.frame_step}")

    for index, video_path in enumerate(videos, start=1):
        saved_count = process_video(
            video_path=video_path,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            crop=crop,
            image_ext=args.image_ext,
            frame_step=args.frame_step,
        )
        total_frames += saved_count
        class_name = video_path.parent.relative_to(dataset_dir).parts[0]
        print(
            f"[{index}/{len(videos)}] {video_path.name} -> {class_name}, saved {saved_count} frames"
        )

    print(f"Finished. Total saved frames: {total_frames}")


if __name__ == "__main__":
    main()
