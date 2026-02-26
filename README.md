# Hand Gesture Recognition — MLflow Experiment

## Overview

This project trains and compares multiple machine learning classifiers on a hand gesture recognition dataset containing **25,675 samples** across **18 gesture classes**, using 3D hand landmark features extracted with MediaPipe (63 features per sample).

MLflow was used throughout the experiment to track parameters, metrics, models, and artifacts for every run.

---

## Experiment Details

| | |
|---|---|
| **Experiment Name** | `hand-gesture-recognition` |
| **Tracking** | MLflow 3.9.0 |
| **Dataset** | `hand_landmarks_data.csv` |
| **Features** | 63 wrist-normalized MediaPipe landmark coordinates |
| **Classes** | 18 hand gestures |
| **Evaluation Metric** | Macro F1-Score & Accuracy |

---

## Model Comparison

| Model | Mean CV Score | Accuracy | Macro F1-Score |
|---|---|---|---|
| **Random Forest** ★ | **0.98** | **0.98** | **0.98** |
| SVM | 0.98 | 0.98 | 0.98 |
| Decision Tree | 0.95 | 0.96 | 0.96 |
| Logistic Regression | 0.89 | 0.89 | 0.89 |

---

## Best Model: Random Forest

**Random Forest** was selected as the final model and registered in the MLflow Model Registry under the name `hand-gesture-recognition-random-forest`.

### Why Random Forest?

Although SVM achieved the same test accuracy (0.98) and Macro F1-Score (0.98), **Random Forest was chosen** for the following reasons:

- **Highest Mean CV Score (0.98)** — consistent performance across all cross-validation folds, indicating better generalization compared to SVM
- **Faster inference** — Random Forest predicts significantly faster than SVM, which is important for real-time hand gesture recognition via webcam
- **More interpretable** — feature importances can be extracted to understand which landmarks contribute most to each gesture class
- **No kernel sensitivity** — SVM performance is sensitive to the choice of kernel and scaling; Random Forest is more robust out of the box

---

## MLflow Tracking Backend

This project uses a **SQLite database** as the MLflow tracking backend instead of the default file-based `mlruns/` folder. This means experiment data is stored in two places:

- **`mlflow.db`** — SQLite database that stores all run metadata: parameters, metrics, and tags
- **`mlartifacts/`** — folder that stores all artifacts: models, confusion matrix plots, and dataset inputs
---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the MLflow tracking server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

### 3. Run the experiment
```bash
python main.py
```

### 4. Open the MLflow UI
Navigate to [http://localhost:5000](http://localhost:5000) in your browser to explore runs, compare metrics, and access the model registry.

---

## Project Structure

```
├── main.py                        # Entry point — trains and logs all models
├── requirements.txt
├── mlflow.db                      # SQLite database storing all run metadata
├── mlartifacts/                   # MLflow artifacts: models, plots, datasets
├── dataset/
│   └── hand_landmarks_data.csv
├── plots/                         # Confusion matrix plots (auto-generated)
├── screenshots/                   # MLflow UI screenshots
└── src/
    ├── data_preprocessing.py      # Data loading, normalization, train/test split
    ├── evaluation.py              # Metrics calculation and confusion matrix plotting
    ├── model_training.py          # GridSearchCV training for all 6 models
    └── mlflow_logging.py          # MLflow experiment setup and run logging
```

---

## Model Registry

The best model is registered in the MLflow Model Registry:

- **Registry Name:** `hand-gesture-recognition-random-forest`
- **Stage:** Production
- **Run:** `RandomForest`
- **Experiment:** `hand-gesture-recognition`