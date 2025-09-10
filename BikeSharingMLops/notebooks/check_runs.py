# check_runs.py
import mlflow
from mlflow.tracking import MlflowClient
import os


tracking_path = os.path.join(os.getcwd(), "BikeSharingMLops", "notebooks", "mlruns")
mlflow.set_tracking_uri(f"file:{tracking_path}")


client = MlflowClient()
experiment = client.get_experiment_by_name("Default")


runs = client.search_runs([experiment.experiment_id])


if not runs:
    print("❌ No runs found.")
else:
    print("✅ Found the following run IDs:")
    for run in runs:
        print("-", run.info.run_id)
