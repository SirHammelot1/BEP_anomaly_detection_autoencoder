import matplotlib.pyplot as plt
import numpy as np

def plot_autoencoder_reconstructions(
    autoencoder,
    data,
    ppm,
    num_examples=9,
    rows=3,
    cols=3
):
    encoded_data = autoencoder.encoder(data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()

    ppm = np.asarray(ppm)
    ppm_flipped = ppm[::-1]

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = axes.flatten()

    for i in range(min(num_examples, len(data))):
        ppm_mask = (ppm_flipped >= 0) & (ppm_flipped <= 6)

        ppm_plot = ppm_flipped[ppm_mask]
        input_plot = data[i][ppm_mask]
        decoded_plot = decoded_data[i][ppm_mask]

        axes[i].plot(ppm_plot, input_plot, "b", linewidth=1)
        axes[i].plot(ppm_plot, decoded_plot, "r", linewidth=1)

        axes[i].fill_between(
            ppm_plot,
            decoded_plot,
            input_plot,
            color="lightcoral",
            alpha=0.5
        )

        axes[i].set_title(f"Example: {i}")
        axes[i].set_xlabel("Chemical Shift (ppm)")
        axes[i].set_xlim(6, 0)  # standard MRS display
        axes[i].grid(True)

    for j in range(num_examples, len(axes)):
        axes[j].axis("off")

    fig.legend(
        ["Input", "Reconstruction", "Error"],
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()
