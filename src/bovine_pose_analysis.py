from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageEnhance


KEYPOINT_NAMES = [
    "left_eye",
    "right_eye",
    "nose",
    "left_ear",
    "right_ear",
    "left_front_elbow",
    "right_front_elbow",
    "left_back_elbow",
    "right_back_elbow",
    "left_front_knee",
    "right_front_knee",
    "left_back_knee",
    "right_back_knee",
    "left_front_paw",
    "right_front_paw",
    "left_back_paw",
    "right_back_paw",
    "throat",
    "withers",
    "tailbase",
]


@dataclass
class LoadedDataset:
    raw: dict
    categories: dict[int, str]
    image_names: dict[int, str]
    image_paths: dict[str, Path]
    annotations: pd.DataFrame
    keypoints_long: pd.DataFrame
    per_class_summary: pd.DataFrame
    cow_all: pd.DataFrame
    cow_local: pd.DataFrame


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(json_path: Path, images_root: Path) -> LoadedDataset:
    with json_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    categories = {item["id"]: item["name"] for item in raw["categories"]}
    image_names = {int(key): value for key, value in raw["images"].items()}
    image_paths = {path.name: path for path in images_root.rglob("*") if path.is_file()}

    records: list[dict] = []
    keypoint_rows: list[dict] = []
    for annotation_id, ann in enumerate(raw["annotations"], start=1):
        image_name = image_names[ann["image_id"]]
        bbox = ann["bbox"]
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_area = bbox_width * bbox_height
        category_name = categories[ann["category_id"]]
        is_local = image_name in image_paths

        records.append(
            {
                "annotation_id": annotation_id,
                "image_id": ann["image_id"],
                "image_name": image_name,
                "image_path": str(image_paths[image_name]) if is_local else None,
                "category_id": ann["category_id"],
                "category_name": category_name,
                "num_keypoints": ann["num_keypoints"],
                "bbox_xmin": bbox[0],
                "bbox_ymin": bbox[1],
                "bbox_xmax": bbox[2],
                "bbox_ymax": bbox[3],
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_area": bbox_area,
                "is_local_image": is_local,
                "keypoints": ann["keypoints"],
            }
        )

        for kp_name, (x_coord, y_coord, visible) in zip(KEYPOINT_NAMES, ann["keypoints"]):
            keypoint_rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_name": image_name,
                    "category_name": category_name,
                    "keypoint_name": kp_name,
                    "x": x_coord,
                    "y": y_coord,
                    "visible": visible,
                    "is_local_image": is_local,
                }
            )

    annotations = pd.DataFrame.from_records(records)
    keypoints_long = pd.DataFrame.from_records(keypoint_rows)

    per_class_summary = (
        annotations.groupby("category_name")
        .agg(
            total_instances=("annotation_id", "count"),
            unique_images=("image_name", "nunique"),
            local_images=("image_name", lambda series: series[annotations.loc[series.index, "is_local_image"]].nunique()),
            mean_bbox_area=("bbox_area", "mean"),
            mean_num_keypoints=("num_keypoints", "mean"),
        )
        .reset_index()
        .sort_values("total_instances", ascending=False)
    )

    cow_all = annotations.loc[annotations["category_name"] == "cow"].copy()
    cow_local = cow_all.loc[cow_all["is_local_image"]].copy()

    return LoadedDataset(
        raw=raw,
        categories=categories,
        image_names=image_names,
        image_paths=image_paths,
        annotations=annotations,
        keypoints_long=keypoints_long,
        per_class_summary=per_class_summary,
        cow_all=cow_all,
        cow_local=cow_local,
    )


def save_filtered_cow_json(dataset: LoadedDataset, output_path: Path, local_only: bool = True) -> Path:
    source_annotations = dataset.cow_local if local_only else dataset.cow_all
    selected_names = set(source_annotations["image_name"])
    selected_ids = {image_id for image_id, name in dataset.image_names.items() if name in selected_names}

    filtered = {
        "info": dataset.raw.get("info", {}),
        "categories": [item for item in dataset.raw["categories"] if item["name"] == "cow"],
        "images": {str(image_id): name for image_id, name in dataset.image_names.items() if image_id in selected_ids},
        "annotations": [
            ann
            for ann in dataset.raw["annotations"]
            if ann["category_id"] == 5 and ann["image_id"] in selected_ids
        ],
    }

    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _apply_cow_preprocessing(image: Image.Image, bbox: Iterable[float], output_size: tuple[int, int] = (256, 256)) -> tuple[Image.Image, Image.Image, Image.Image]:
    xmin, ymin, xmax, ymax = bbox
    cropped = image.crop((xmin, ymin, xmax, ymax))
    resized = cropped.resize(output_size)
    enhanced = ImageEnhance.Contrast(resized).enhance(1.15)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.25)
    return cropped, resized, enhanced


def draw_pose_overlay(image: Image.Image, bbox: Iterable[float], keypoints: list[list[float]], skeleton: list[list[int]]) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle((xmin, ymin, xmax, ymax), outline="#d62728", width=3)

    visible_points: dict[int, tuple[float, float]] = {}
    for index, (x_coord, y_coord, visible) in enumerate(keypoints):
        if visible:
            visible_points[index] = (x_coord, y_coord)
            radius = 4
            draw.ellipse((x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius), fill="#ffdd57", outline="#111111")

    for start_idx, end_idx in skeleton:
        start_idx -= 1
        end_idx -= 1
        if start_idx in visible_points and end_idx in visible_points:
            draw.line((visible_points[start_idx], visible_points[end_idx]), fill="#00bcd4", width=3)

    return overlay


def create_processing_figure(dataset: LoadedDataset, output_path: Path) -> pd.Series:
    sample = dataset.cow_local.sort_values(["num_keypoints", "bbox_area"], ascending=[False, False]).iloc[0]
    image = Image.open(sample["image_path"]).convert("RGB")
    bbox = [sample["bbox_xmin"], sample["bbox_ymin"], sample["bbox_xmax"], sample["bbox_ymax"]]
    cropped, resized, enhanced = _apply_cow_preprocessing(image, bbox)
    overlay = draw_pose_overlay(image, bbox, sample["keypoints"], dataset.raw["categories"][-1]["skeleton"])

    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    stages = [
        (image, "1. Imagem original"),
        (cropped, "2. Recorte pela bbox"),
        (enhanced, "3. Normalizacao visual"),
        (overlay, "4. Pose sobreposta"),
    ]
    for axis, (stage_image, title) in zip(axes, stages):
        axis.imshow(stage_image)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(f"Pipeline ilustrativo para bovinos: {sample['image_name']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return sample


def generate_figures(dataset: LoadedDataset, figures_dir: Path) -> dict[str, Path]:
    ensure_dir(figures_dir)
    sns.set_theme(style="whitegrid", palette="deep")
    outputs: dict[str, Path] = {}

    summary = dataset.per_class_summary.copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="category_name", y="total_instances", ax=ax)
    ax.set_title("Distribuicao de instancias anotadas por classe")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Instancias")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    path = figures_dir / "01_instancias_por_classe.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs["instances_by_class"] = path

    coverage = summary.melt(
        id_vars="category_name",
        value_vars=["unique_images", "local_images"],
        var_name="metric",
        value_name="value",
    )
    coverage["metric"] = coverage["metric"].map(
        {"unique_images": "Imagens anotadas", "local_images": "Imagens locais"}
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=coverage, x="category_name", y="value", hue="metric", ax=ax)
    ax.set_title("Cobertura de imagens anotadas vs. imagens disponiveis localmente")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de imagens")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    path = figures_dir / "02_cobertura_imagens.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs["image_coverage"] = path

    cow_pose_rows: list[dict] = []
    for _, row in dataset.cow_all.iterrows():
        bbox_width = row["bbox_width"]
        bbox_height = row["bbox_height"]
        if bbox_width <= 0 or bbox_height <= 0:
            continue
        for keypoint_name, (x_coord, y_coord, visible) in zip(KEYPOINT_NAMES, row["keypoints"]):
            if not visible:
                continue
            cow_pose_rows.append(
                {
                    "keypoint_name": keypoint_name,
                    "x_norm": (x_coord - row["bbox_xmin"]) / bbox_width,
                    "y_norm": (y_coord - row["bbox_ymin"]) / bbox_height,
                }
            )
    pose_df = pd.DataFrame(cow_pose_rows)
    pose_summary = pose_df.groupby("keypoint_name")[["x_norm", "y_norm"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pose_summary["x_norm"], pose_summary["y_norm"], s=60, color="#1f77b4")
    for _, row in pose_summary.iterrows():
        ax.text(row["x_norm"] + 0.01, row["y_norm"] + 0.01, row["keypoint_name"], fontsize=8)
    skeleton = dataset.raw["categories"][-1]["skeleton"]
    point_lookup = {row["keypoint_name"]: (row["x_norm"], row["y_norm"]) for _, row in pose_summary.iterrows()}
    for start_idx, end_idx in skeleton:
        start_name = KEYPOINT_NAMES[start_idx - 1]
        end_name = KEYPOINT_NAMES[end_idx - 1]
        start_point = point_lookup[start_name]
        end_point = point_lookup[end_name]
        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            color="#00bcd4",
            linewidth=2,
        )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_title("Pose media normalizada dos bovinos")
    ax.set_xlabel("Posicao X normalizada")
    ax.set_ylabel("Posicao Y normalizada")
    path = figures_dir / "03_keypoints_bovinos.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs["cow_keypoints"] = path

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(dataset.cow_all["bbox_area"], bins=25, kde=True, ax=ax)
    ax.set_title("Distribuicao da area das bounding boxes de bovinos")
    ax.set_xlabel("Area da bounding box (pixels)")
    ax.set_ylabel("Frequencia")
    path = figures_dir / "04_bbox_area_bovinos.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs["cow_bbox_area"] = path

    outputs["processing_pipeline"] = figures_dir / "05_pipeline_bovino.png"
    create_processing_figure(dataset, outputs["processing_pipeline"])
    return outputs


def generate_tables(dataset: LoadedDataset, tables_dir: Path) -> dict[str, Path]:
    ensure_dir(tables_dir)
    outputs: dict[str, Path] = {}

    class_table_path = tables_dir / "dataset_summary.csv"
    dataset.per_class_summary.round(2).to_csv(class_table_path, index=False)
    outputs["dataset_summary"] = class_table_path

    cow_summary = pd.DataFrame(
        [
            {
                "metric": "instancias_anotadas_cow",
                "value": int(len(dataset.cow_all)),
            },
            {
                "metric": "imagens_unicas_cow",
                "value": int(dataset.cow_all["image_name"].nunique()),
            },
            {
                "metric": "imagens_locais_cow",
                "value": int(dataset.cow_local["image_name"].nunique()),
            },
            {
                "metric": "proporcao_cow_no_dataset",
                "value": round(len(dataset.cow_all) / len(dataset.annotations), 4),
            },
            {
                "metric": "media_area_bbox_cow",
                "value": round(float(dataset.cow_all["bbox_area"].mean()), 2),
            },
            {
                "metric": "media_keypoints_visiveis_cow",
                "value": round(float(dataset.cow_all["num_keypoints"].mean()), 2),
            },
        ]
    )
    cow_summary_path = tables_dir / "cow_summary.csv"
    cow_summary.to_csv(cow_summary_path, index=False)
    outputs["cow_summary"] = cow_summary_path

    local_cow_metadata = dataset.cow_local[
        [
            "annotation_id",
            "image_name",
            "image_path",
            "num_keypoints",
            "bbox_width",
            "bbox_height",
            "bbox_area",
        ]
    ].sort_values("image_name")
    local_cow_path = tables_dir / "cow_local_metadata.csv"
    local_cow_metadata.to_csv(local_cow_path, index=False)
    outputs["cow_local_metadata"] = local_cow_path
    return outputs
