"""
Registers the best-performing model from MLflow tracking and promotes it
to the 'production' alias in the MLflow Model Registry.
"""
import yaml
import mlflow
from mlflow.tracking import MlflowClient

with open("params.yaml") as f:
    params = yaml.safe_load(f)

EXPERIMENT_NAME = "accident-severity-models"
MODEL_NAME = "accident-severity-lightgbm"
PROMOTION_METRIC = params.get("registry", {}).get("promotion_metric", "weighted_f1")
PROMOTION_THRESHOLD = params.get("registry", {}).get("promotion_threshold", 0.75)


def main():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment.experiment_id, order_by=[f"metrics.{PROMOTION_METRIC} DESC"]
    )

    best_run = runs[0]
    best_score = best_run.data.metrics.get(PROMOTION_METRIC)
    best_name = best_run.data.tags.get("mlflow.runName")

    print(f"Best run: {best_name} | {PROMOTION_METRIC}={best_score:.4f}")

    if best_score < PROMOTION_THRESHOLD:
        print(f"Score below threshold ({PROMOTION_THRESHOLD}) — not promoting.")
        return

    model_uri = f"runs:/{best_run.info.run_id}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)

    client.set_registered_model_alias(
        name=MODEL_NAME, alias="production", version=registered.version
    )
    print(f"{MODEL_NAME} v{registered.version} promoted to 'production' alias.")


if __name__ == "__main__":
    main()