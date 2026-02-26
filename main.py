from src.model_training import (
    train_logistic_regression,
    train_decision_tree,
    train_svm,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
)
from src.mlflow_logging import log_model_with_mlflow, setup_mlflow_experiment
from pathlib import Path
import logging
from colorama import Fore, Style


GESTURE_CLASSES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=f"{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - {Fore.BLUE}%(levelname)s{Style.RESET_ALL} - %(message)s"
    )


def main():
    setup_logging()
    logging.info("Starting Hand Gesture Recognition Experiment...")

    experiment_id = setup_mlflow_experiment("hand-gesture-recognition")

    BASE_DIR = Path(__file__).resolve().parent
    output_dir = BASE_DIR / "plots"

    from src.data_preprocessing import load_and_preprocess_data
    data_path = BASE_DIR / "hand_landmarks_data.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    lr_model = train_logistic_regression(X_train, y_train)
    log_model_with_mlflow(lr_model, X_test, y_test, GESTURE_CLASSES, "LogisticRegression", experiment_id, output_dir)

    dt_model = train_decision_tree(X_train, y_train)
    log_model_with_mlflow(dt_model, X_test, y_test, GESTURE_CLASSES, "DecisionTree", experiment_id, output_dir)

    svm_model = train_svm(X_train, y_train)
    log_model_with_mlflow(svm_model, X_test, y_test, GESTURE_CLASSES, "SVM", experiment_id, output_dir)

    rf_model = train_random_forest(X_train, y_train)
    log_model_with_mlflow(rf_model, X_test, y_test, GESTURE_CLASSES, "RandomForest", experiment_id, output_dir)

    gb_model = train_gradient_boosting(X_train, y_train)
    log_model_with_mlflow(gb_model, X_test, y_test, GESTURE_CLASSES, "GradientBoosting", experiment_id, output_dir)

    xgb_model = train_xgboost(X_train, y_train)
    log_model_with_mlflow(xgb_model, X_test, y_test, GESTURE_CLASSES, "XGBoost", experiment_id, output_dir)

    logging.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
