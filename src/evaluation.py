from sklearn import metrics
import matplotlib.pyplot as plt
from pathlib import Path
import logging


def eval_metrics(actual, pred):
    logging.info("Calculating evaluation metrics...")
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, average="macro", zero_division=0)
    return accuracy, f1


def plot_confusion_matrix(actual, pred, labels, model_name: str, output_dir: Path):
    logging.info("Plotting confusion matrix...")
    cm = metrics.confusion_matrix(actual, pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png")
    plt.close()
