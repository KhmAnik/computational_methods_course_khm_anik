import matplotlib.pyplot as plt


def plot_observed_vs_predicted(y_true, y_pred, save_path="observed_vs_predicted.png"):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Observed")
    plt.plot(y_pred, label="Predicted")

    plt.xlabel("Time step")
    plt.ylabel("Normalized Streamflow")
    plt.title("Observed vs Predicted Streamflow")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_parity(y_true, y_pred, save_path="parity_plot.png"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Observed Streamflow")
    plt.ylabel("Predicted Streamflow")
    plt.title("Predicted vs Observed Streamflow")
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()