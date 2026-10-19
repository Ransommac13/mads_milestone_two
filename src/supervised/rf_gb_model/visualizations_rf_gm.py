import numpy as np 
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 


def plot_feature_set_comparison(comparison, path):

    """
    Validation R2
    """
    table = comparison.pivot(index="feature_set", columns="model", values="val R2 (log)")
    table = table.sort_values(table.columns[0])

    ax = table.plot(kind="barh", figsize=(8,5))
    ax.set_xlabel("Validation R2 (log sales)")
    ax.set_ylabel("")
    ax.set_title("Which feature predicts sales best?")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(y_log, pred_log, title, path):
    # predicted vs actual log sales 

    plt.figure(figsize=(6,6))
    plt.hexbin(y_log, pred_log, gridsize=40, cmap="Blues", mincnt=1)

    low, high = min(y_log.min(), pred_log.min()), max(y_log.max(), pred_log.max())
    plt.plot([low, high], [low, high], color="red",
             linestyle="--", label="perfect prediction")

    plt.xlabel("Actual Log (avg daily sales)")
    plt.ylabel("Predicted log (avg daily sales)")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_residuals(y_log, pred_log, title, path):

    residuals = y_log - pred_log
    fig, axes = plt.subplots(1,2, figsize=(11,4))

    axes[0].scatter(pred_log, residuals, s=6, alpha=0.3)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted log(avg daily sales)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs predicted")

    # how errors are spread
    axes[1].hist(residuals, bins=50)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Retailers")
    axes[1].set_title("Residuals Distribution")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_error_by_group(df, y_log, pred_log, cloumn, title, path, top=10):

    #abs error of each retailer 

    errors = (y_log-pred_log).abs().groupby(df[cloumn].values)
    summary = errors.agg(["mean", "size"])
    summary = summary.sort_values("size", ascending =False).head(top).sort_values("mean")

    plt.figure(figsize=(8,5))
    plt.barh(summary.index.astype(str), summary["mean"])
    plt.xlabel("Mean Absolute error (log sales)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_importance(importance, title, path, top=15):
    """
    Permutation importance
    """
    importance = importance.head(top).sort_values()

    plt.figure(figsize=(8,5))
    plt.barh(importance.index, importance.values)
    plt.xlabel("Increase in validation MAE when shuffled")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()