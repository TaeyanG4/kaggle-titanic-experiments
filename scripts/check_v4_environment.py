"""Inventory the local environment for the Titanic v4 Diverse Model Zoo.

This script does not fit any model and does not install packages. It records
which model families are ready now and which optional dependencies are missing.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
EXPORT_DIR = BASE_DIR / "exports" / "v4"


MODEL_REGISTRY = [
    (1, "TabPFN_v3", "Foundation", "tabpfn", "optional-supported", "GPU recommended; explicit v3 checkpoint"),
    (2, "TabPFN_v2_5", "Foundation", "tabpfn", "optional-supported", "Older TabPFN checkpoint for diversity"),
    (3, "TabPFN_v2", "Foundation", "tabpfn", "planned", "Version adapter deferred"),
    (4, "TabICLv2", "Foundation", "tabicl", "optional-supported", "In-context classifier; checkpoint download on first use"),
    (5, "TabDPT", "Foundation", "", "planned", "Research implementation; add after Stage-1 screen"),
    (6, "TabFM", "Foundation", "", "planned", "Research implementation; add after Stage-1 screen"),
    (7, "TabM", "Modern tabular DL", "tabm", "planned", "Needs a dedicated PyTorch training loop"),
    (8, "RealMLP", "Modern tabular DL", "pytabkit", "planned", "Can be added through pytabkit"),
    (9, "ModernNCA", "Modern tabular DL", "", "planned", "Nearest-neighbour-style neural representation"),
    (10, "TabR", "Modern tabular DL", "pytabkit", "planned", "Retrieval model; faiss dependency may be needed"),
    (11, "FTTransformer", "Modern tabular DL", "rtdl_revisiting_models", "planned", "Transformer baseline"),
    (12, "SAINT", "Modern tabular DL", "", "planned", "Transformer-style row/column attention"),
    (13, "TabTransformer", "Modern tabular DL", "pytorch_tabular", "planned", "Categorical transformer"),
    (14, "MLP_PLR", "Modern tabular DL", "rtdl_revisiting_models", "planned", "MLP with richer numerical embeddings"),
    (15, "ResNet", "Modern tabular DL", "rtdl_revisiting_models", "planned", "Tabular residual MLP"),
    (16, "MLP", "Modern tabular DL", "torch", "planned", "Simple neural baseline"),
    (17, "CatBoost", "Tree", "catboost", "reference-existing", "Already available from v1/v2 fold-safe audit"),
    (18, "LightGBM", "Tree", "lightgbm", "reference-existing", "Already available from v1/v2 fold-safe audit"),
    (19, "XGBoost", "Tree", "xgboost", "reference-existing", "Already available from v1/v2 fold-safe audit"),
    (20, "HistGradientBoosting", "Tree", "sklearn", "stage1-supported", "New low-friction tree-family comparison"),
    (21, "GradientBoosting", "Tree", "sklearn", "reference-existing", "Already available from v1/v2 fold-safe audit"),
    (22, "RandomForest", "Tree", "sklearn", "reference-existing", "Already available from v1/v2 fold-safe audit"),
    (23, "ExtraTrees", "Tree", "sklearn", "reference-existing", "Already available from v1/v2 fold-safe audit"),
    (24, "AdaBoost", "Tree", "sklearn", "stage1-supported", "Different boosting bias"),
    (25, "LogisticRegression", "Classical", "sklearn", "stage1-supported", "Linear decision boundary; high diversity value"),
    (26, "RBF_SVM", "Classical", "sklearn", "stage1-supported", "Kernel boundary; scaled features"),
    (27, "KNN", "Classical", "sklearn", "stage1-supported", "Local-neighbour baseline"),
    (28, "LDA", "Classical", "sklearn", "stage1-supported", "Shared-covariance discriminant model"),
    (29, "QDA", "Classical", "sklearn", "stage1-supported", "Class-specific covariance"),
    (30, "GaussianNB", "Classical", "sklearn", "stage1-supported", "Strongly different independence assumptions"),
    (31, "EBM", "Interpretable", "interpret", "optional-supported", "Explainable boosting model"),
    (32, "RuleFit", "Interpretable", "imodels", "planned", "Rule extraction plus linear weighting"),
    (33, "FIGS", "Interpretable", "imodels", "planned", "Small jointly optimized tree set"),
    (34, "NODE", "Tree/DL hybrid", "", "planned", "Differentiable oblivious tree ensemble"),
    (35, "GRANDE", "Tree/DL hybrid", "", "planned", "Gradient-trained decision-tree ensemble"),
    (36, "Trompt", "Modern tabular DL", "", "planned", "Prompt-inspired tabular model"),
    (37, "ExcelFormer", "Modern tabular DL", "", "planned", "Transformer-style feature interaction"),
    (38, "AMFormer", "Modern tabular DL", "", "planned", "Arithmetic interaction attention"),
    (39, "T2G_Former", "Modern tabular DL", "", "planned", "Feature-relation graph transformer"),
    (40, "Mambular", "Modern tabular DL", "", "planned", "State-space tabular model"),
    (41, "HyperFast", "Foundation/Meta", "", "planned", "Hypernetwork-generated task model"),
    (42, "MotherNet", "Foundation/Meta", "", "planned", "Meta-trained hypernetwork"),
    (43, "RFM", "Kernel/Feature", "", "planned", "Recursive feature learning"),
    (44, "xRFM", "Kernel/Feature", "pytabkit", "planned", "Scalable RFM variant"),
    (45, "AutoGluon", "AutoML", "autogluon", "planned", "Separate AutoML benchmark lane"),
]


def module_available(name: str) -> bool | None:
    if not name:
        return None
    return importlib.util.find_spec(name) is not None


def package_version(name: str) -> str | None:
    if not name or not module_available(name):
        return None
    package_name = {
        "sklearn": "scikit-learn",
    }.get(name, name)
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for number, model, family, package, implementation, note in MODEL_REGISTRY:
        rows.append(
            {
                "number": number,
                "model": model,
                "family": family,
                "package_probe": package or "n/a",
                "package_available": module_available(package),
                "package_version": package_version(package),
                "implementation_status": implementation,
                "note": note,
            }
        )
    registry = pd.DataFrame(rows)
    registry.to_csv(EXPORT_DIR / "model_registry.csv", index=False)

    torch_info = {
        "available": module_available("torch"),
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
    }
    if torch_info["available"]:
        import torch

        torch_info.update(
            {
                "version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_info,
        "ready_stage1_models": registry.loc[
            registry["implementation_status"].isin(["stage1-supported", "reference-existing"])
            & registry["package_available"].eq(True),
            "model",
        ].tolist(),
        "ready_optional_models": registry.loc[
            (registry["implementation_status"] == "optional-supported")
            & registry["package_available"].eq(True),
            "model",
        ].tolist(),
        "missing_optional_packages": sorted(
            registry.loc[
                (registry["implementation_status"] == "optional-supported")
                & registry["package_available"].eq(False),
                "package_probe",
            ].unique().tolist()
        ),
    }
    with (EXPORT_DIR / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)

    print("=== Titanic v4 Model Zoo environment ===")
    print(f"Python: {env['python']}")
    print(
        "GPU:",
        torch_info["gpu_name"] if torch_info["cuda_available"] else "not available through PyTorch",
        f"(CUDA {torch_info['cuda_version']})" if torch_info["cuda_available"] else "",
    )
    print()
    print("Ready Stage-1 / existing references:")
    print(", ".join(env["ready_stage1_models"]))
    print()
    print("Ready optional models:")
    print(", ".join(env["ready_optional_models"]) or "(none yet)")
    print("Missing optional packages:", ", ".join(env["missing_optional_packages"]) or "(none)")
    print(f"Registry: {EXPORT_DIR / 'model_registry.csv'}")
    print(f"Environment: {EXPORT_DIR / 'environment.json'}")


if __name__ == "__main__":
    main()
