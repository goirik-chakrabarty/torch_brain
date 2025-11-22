import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    data_dir="model_predictions", output_dir="plots", duration_to_plot=30.0
):
    """
    Reads .npz files from data_dir and saves comparison plots to output_dir.

    Args:
        data_dir (str): Directory containing the .npz files.
        output_dir (str): Directory to save the generated plots.
        duration_to_plot (float): Seconds of data to plot (to keep visualizations readable).
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all .npz files
    files = glob.glob(os.path.join(data_dir, "*.npz"))

    if not files:
        print(
            f"No .npz files found in '{data_dir}'. Did you run the evaluation script?"
        )
        return

    print(f"Found {len(files)} files. Generating plots...")

    for output_file in files:
        # Load data
        try:
            data = np.load(output_file)
            timestamps = data["timestamps"]
            preds = data["preds"]
            targets = data["targets"]
        except Exception as e:
            print(f"Error loading {output_file}: {e}")
            continue

        # Ensure 2D shape for consistency (Time, Channels)
        if preds.ndim == 1:
            preds = preds[:, None]
            targets = targets[:, None]

        num_channels = preds.shape[1]
        filename = os.path.basename(output_file).replace(".npz", "")

        # Select a time window to plot (e.g., first 30 seconds)
        # We sort just in case, though they should be sorted
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        preds = preds[sort_idx]
        targets = targets[sort_idx]

        start_time = timestamps[0]
        end_time = start_time + duration_to_plot
        mask = (timestamps >= start_time) & (timestamps <= end_time)

        # If the session is shorter than duration_to_plot, take all of it
        if not np.any(mask):
            mask = np.ones_like(timestamps, dtype=bool)

        t_subset = timestamps[mask]
        p_subset = preds[mask]
        g_subset = targets[mask]

        # Create Plot
        fig, axes = plt.subplots(
            num_channels, 1, figsize=(15, 4 * num_channels), sharex=True
        )
        if num_channels == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(
                t_subset,
                g_subset[:, i],
                label="Ground Truth",
                color="black",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.plot(
                t_subset,
                p_subset[:, i],
                label="Prediction",
                color="#ff7f0e",
                linewidth=1.5,
                alpha=0.9,
            )

            ax.set_ylabel(f"Value (Channel {i})", fontsize=12)
            ax.set_title(f"{filename} - Channel {i}", fontsize=14)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            # Optional: Add correlation score for this specific window in the title
            if len(g_subset) > 1:
                corr = np.corrcoef(g_subset[:, i], p_subset[:, i])[0, 1]
                ax.set_title(f"{filename} - Channel {i} (Window Corr: {corr:.3f})")

        axes[-1].set_xlabel("Time (s)", fontsize=12)
        plt.tight_layout()

        # Save comparison
        save_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    # You can adjust the duration_to_plot to see more or less data
    plot_results(
        data_dir="model_predictions", output_dir="plots", duration_to_plot=60.0
    )
