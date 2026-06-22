from fsl_mrs.core import nifti_mrs
import matplotlib.pyplot as plt
import numpy as np

def plot_example_spectrum_fslmrs(file_path, ref_ppm=2.02):
    # Load NIfTI-MRS using FSL-MRS
    nifti_obj = nifti_mrs.NIFTI_MRS(file_path)

    # Convert to MRS object
    mrs = nifti_obj.mrs()

    # FSL-MRS MRS object contains the FID
    fid = mrs.FID

    # Convert FID to spectrum
    spectrum = np.fft.fftshift(np.fft.fft(fid))
    spectrum_real = np.real(spectrum)
    spectrum_real_flipped = spectrum_real[::-1]

    # Get ppm axis from FSL-MRS object
    ppm = mrs.getAxes(axis='ppmshift')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ppm_flipped = ppm[::-1]

    ppm_mask = (ppm_flipped >= 0) & (ppm_flipped <= 6)

    ppm_plot = ppm_flipped[ppm_mask]
    input_plot = spectrum_real_flipped[ppm_mask]

    axes[0].plot(np.arange(len(spectrum_real_flipped)), spectrum_real_flipped)
    axes[0].set_title("Input into autoencoder")
    axes[0].set_xlabel("input number into autoencoder")
    axes[0].set_ylabel("Normalized Real Signal")
    axes[0].grid(True)

    axes[1].plot(ppm_plot, input_plot)
    axes[1].invert_xaxis()
    axes[1].set_title("Input into autoencoder + ppm scale")
    axes[1].set_xlabel("ppm scale")
    axes[1].set_ylabel("Real Signal")
    axes[1].grid(True)
    axes[1].axvline(ref_ppm, linestyle="--", alpha=0.6, label=f"NAA reference ~{ref_ppm} ppm")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return ppm

