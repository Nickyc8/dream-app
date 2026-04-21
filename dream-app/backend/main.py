from fastapi import FastAPI
from pydantic import BaseModel
from model_pipeline import DreamPipeline
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dreamsearch.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REQUIRED_ARTIFACT = ARTIFACTS_DIR / "plot_df_3d.parquet"

pipeline = DreamPipeline(data_path=DATA_PATH, artifacts_dir=ARTIFACTS_DIR)

if REQUIRED_ARTIFACT.exists():
    pipeline.load_artifacts()
else:
    pipeline.build_from_csv()
    pipeline.save_artifacts()

class DreamRequest(BaseModel):
    dreamText: str

@app.get("/")
def root():
    return {"message": "Dream API is running"}

@app.post("/predict")
def predict(req: DreamRequest):
    return pipeline.predict(req.dreamText)

@app.get("/plot-data-3d")
def plot_data_3d():
    return pipeline.get_plot_data_3d()