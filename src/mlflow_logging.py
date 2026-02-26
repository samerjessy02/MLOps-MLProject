import mlflow
import mlflow.data
import mlflow.models
import mlflow.sklearn
from pathlib import Path
from src.evaluation import eval_metrics, plot_confusion_matrix
import logging


def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "http://localhost:5000") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id


def log_model_with_mlflow(model, X_test, y_test, label_classes, model_name: str, exp_id: str, output_dir: Path):
    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:
        logging.info(f"Logging {model_name} to MLflow...")
        mlflow.set_tag("model", model_name)

        pred = model.predict(X_test)
        accuracy, f1 = eval_metrics(y_test, pred)

        plot_confusion_matrix(y_test, pred, label_classes, model_name, output_dir)

        mlflow.log_params(model.best_params_)
        mlflow.log_metrics({
            "Mean CV Score": model.best_score_,
            "Accuracy": accuracy,
            "Macro F1-Score": f1,
        })

        mlflow.log_artifact(str(output_dir / f"confusion_matrix_{model_name}.png"))

        pd_dataset = mlflow.data.from_pandas(X_test, name="Testing Dataset")
        mlflow.log_input(pd_dataset, context="Testing")

        signature = mlflow.models.infer_signature(X_test, y_test)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_test.iloc[[0]])
