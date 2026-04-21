from importlib import import_module
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dreamsearch.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REQUIRED_ARTIFACTS = [
    ARTIFACTS_DIR / "plot_df_3d.parquet",
    ARTIFACTS_DIR / "clusterer.joblib",
    ARTIFACTS_DIR / "reducer_10d.joblib",
    ARTIFACTS_DIR / "reducer_2d.joblib",
    ARTIFACTS_DIR / "reducer_3d.joblib",
    ARTIFACTS_DIR / "sentence_transformer" / "model.safetensors",
]

pipeline: Any | None = None
pipeline_error: str | None = None


def _get_pipeline_class() -> type:
    try:
        return import_module("model_pipeline").DreamPipeline
    except ModuleNotFoundError:
        return import_module("backend.model_pipeline").DreamPipeline


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(60).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _artifacts_ready() -> bool:
    for artifact_path in REQUIRED_ARTIFACTS:
        if not artifact_path.exists() or _is_lfs_pointer(artifact_path):
            return False
    return True


def _ensure_pipeline_loaded() -> None:
    global pipeline
    global pipeline_error

    if pipeline is not None or pipeline_error is not None:
        return

    try:
        dream_pipeline = _get_pipeline_class()(data_path=DATA_PATH, artifacts_dir=ARTIFACTS_DIR)
        if _artifacts_ready():
            dream_pipeline.load_artifacts()
        else:
            dream_pipeline.build_from_csv()
            dream_pipeline.save_artifacts()
        pipeline = dream_pipeline
    except Exception as exc:
        pipeline_error = f"{type(exc).__name__}: {exc}"


def _require_pipeline() -> Any:
    _ensure_pipeline_loaded()
    if pipeline is None:
        detail = "Pipeline is unavailable. Install backend requirements and verify artifacts."
        if pipeline_error:
            detail = f"{detail} ({pipeline_error})"
        raise HTTPException(status_code=503, detail=detail)
    return pipeline


@app.on_event("startup")
def startup() -> None:
    _ensure_pipeline_loaded()


class DreamRequest(BaseModel):
    dreamText: str


@app.get("/")
def root():
    _ensure_pipeline_loaded()
    return {
        "message": "Dream API is running",
        "pipeline_ready": pipeline is not None,
        "artifacts_ready": _artifacts_ready(),
        "pipeline_error": pipeline_error,
    }


@app.post("/predict")
def predict(req: DreamRequest):
    return _require_pipeline().predict(req.dreamText)


@app.get("/plot-data-3d")
def plot_data_3d():
    return _require_pipeline().get_plot_data_3d()