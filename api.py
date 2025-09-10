from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = FastAPI(title="BikeShare Prediction API")

# Step 1: Load the latest model dynamically from MLflow
client = MlflowClient()
experiment = client.get_experiment_by_name("Default")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)
latest_run_id = runs[0].info.run_id

# Use MLflow to download the actual local path to the model folder
model_local_path = client.download_artifacts(run_id=latest_run_id, path="model")

# Load model using resolved path
model = mlflow.sklearn.load_model(model_local_path)

# Step 2: Define API routes
@app.get("/")
def read_root():
    return {"message": "Welcome to the BikeShare Prediction API"}

@app.post("/predict/")
async def predict_rentals(request: Request):
    try:
        # Step 3: Parse JSON input
        data = await request.json()
        df = pd.DataFrame([data])  # Expect a single row of features

        # Step 4: Predict
        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run with: uvicorn api:app --reload
