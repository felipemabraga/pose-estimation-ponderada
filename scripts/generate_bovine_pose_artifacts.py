from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.bovine_pose_analysis import (
        generate_figures,
        generate_tables,
        load_dataset,
        save_filtered_cow_json,
    )

    data_dir = repo_root / "data"
    results_dir = repo_root / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    processed_dir = results_dir / "processed"

    dataset = load_dataset(
        json_path=data_dir / "keypoints.json",
        images_root=data_dir,
    )
    figure_paths = generate_figures(dataset, figures_dir)
    table_paths = generate_tables(dataset, tables_dir)
    filtered_json = save_filtered_cow_json(
        dataset,
        processed_dir / "cow_keypoints_local.json",
        local_only=True,
    )

    manifest = {
        "figures": {name: str(path.relative_to(repo_root)) for name, path in figure_paths.items()},
        "tables": {name: str(path.relative_to(repo_root)) for name, path in table_paths.items()},
        "processed": {"cow_keypoints_local": str(filtered_json.relative_to(repo_root))},
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
