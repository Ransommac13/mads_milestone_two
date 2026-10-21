import numpy as np 
import pandas as pd
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


def plot_predicted_vs_actual(y_log, predictions, title, path):
    # predicted vs actual log sales for random forest and gradient boosting

    fig, axes = plt.subplots(1, len(predictions), figsize=(11, 5.5), sharex=True, sharey=True)

    # same axis range for every panel
    low = min(y_log.min(), *[pred.min() for pred in predictions.values()])
    high = max(y_log.max(), *[pred.max() for pred in predictions.values()])

    for ax, (model_name, pred_log) in zip(axes, predictions.items()):
        ax.hexbin(y_log, pred_log, gridsize=40, cmap="Blues", mincnt=1)
        ax.plot([low, high], [low, high], color="red", linestyle="--", label="perfect prediction")
        ax.set_title(model_name)
        ax.set_xlabel("Actual Log (avg daily sales)")

    axes[0].set_ylabel("Predicted log (avg daily sales)")
    axes[0].legend(loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_log, predictions, title, path):

    fig, axes = plt.subplots(len(predictions), 2, figsize=(11, 4*len(predictions)), squeeze=False)

    for row, (mode_name, pred_log) in zip(axes, predictions.items()):
        
        residuals = y_log - pred_log

        row[0].scatter(pred_log, residuals, s=6, alpha=0.3)
        row[0].axhline(0, color="red", linestyle="--")
        row[0].set_xlabel("Predicted log(avg daily sales)")
        row[0].set_ylabel(f"{mode_name}\nResidual")
        row[0].set_title("Residuals vs predicted")

        # how errors are spread
        row[1].hist(residuals, bins=50)
        row[1].set_xlabel("Residual")
        row[1].set_ylabel("Retailers")
        row[1].set_title("Residuals Distribution")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_error_by_group(df, y_log, predictions, column, title, path, top=10):

    #abs error of each retailer
    biggest = df[column].value_counts().head(top).index

    table = pd.DataFrame({model_name:(y_log - pred_log).abs().groupby(df[column].values).mean()
                          for model_name, pred_log in predictions.items()}).loc[biggest]
    table = table.sort_values(table.columns[0])
    table.index = table.index.astype(str)
    table.plot.barh(figsize=(9,6))
    

    # errors = (y_log-pred_log).abs().groupby(df[column].values)
    # summary = errors.agg(["mean", "size"])
    # summary = summary.sort_values("size", ascending =False).head(top).sort_values("mean")

    plt.ylabel("")
    plt.xlabel("Mean Absolute error (log sales)")
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
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