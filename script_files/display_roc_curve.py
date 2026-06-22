from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from fsl_mrs.core import nifti_mrs
from sklearn.metrics import fbeta_score

def plot_roc_autoencoder(
    autoencoder,
    test_data,
    test_labels,
    example_spec,
    use_autoencoder = True,
    use_water_peak_model = False,
    use_std_model = False,
    use_fwhm_model = False,
    threshold_method="youden",
    plot=True,
    beta = 2,
    anomaly_score_direction = None

):
    if use_autoencoder == True:
        # Reconstruction loss
        test_reconstructions = autoencoder.predict(test_data)
        test_loss = tf.keras.losses.mae(test_data, test_reconstructions).numpy()
        hard_coded_direction = "higher"

    elif use_water_peak_model == True:
        nifti_obj = nifti_mrs.NIFTI_MRS(example_spec)
        mrs = nifti_obj.mrs()
        ppm = np.asarray(mrs.getAxes(axis='ppmshift'))

        water_mask = (ppm > 4.5) & (ppm < 4.9)

        test_data_np = test_data.numpy() if hasattr(test_data, "numpy") else test_data

        water_scores = np.max(np.abs(test_data_np[:, water_mask]), axis=1)

        test_loss = water_scores
        hard_coded_direction = "higher"

    elif use_std_model == True:
        nifti_obj = nifti_mrs.NIFTI_MRS(example_spec)
        mrs = nifti_obj.mrs()
        ppm = np.asarray(mrs.getAxes(axis='ppmshift'))

        noise_mask = (ppm > 11) & (ppm < 12)

        test_data_np = test_data.numpy() if hasattr(test_data, "numpy") else test_data

        quiet_region = test_data_np[:, noise_mask]
        std_scores = np.std(quiet_region, axis=1)

        # Since if snr = high, then good, but if snr = low anomaly; so if higher test_loss it means anomalous 
        test_loss = std_scores
        hard_coded_direction = "higher"
    
    elif use_fwhm_model == True:
        nifti_obj = nifti_mrs.NIFTI_MRS(example_spec)
        mrs = nifti_obj.mrs()
        ppm = np.asarray(mrs.getAxes(axis='ppmshift'))

        water_fwhm_scores = np.array([
            calculate_water_fwhm(ppm, spectrum)
            for spectrum in test_data
        ])

        fwhm_scores = np.nan_to_num(
            water_fwhm_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        test_loss = fwhm_scores
        hard_coded_direction = "higher"

    else: 
        nifti_obj = nifti_mrs.NIFTI_MRS(example_spec)
        mrs = nifti_obj.mrs()
        ppm = np.asarray(mrs.getAxes(axis='ppmshift'))

        signal_mask = (ppm > 1.9) & (ppm < 2.1)
        noise_mask = (ppm > 11) & (ppm < 12)

        test_data_np = test_data.numpy() if hasattr(test_data, "numpy") else test_data

        quiet_region = test_data_np[:, noise_mask]
        std_scores = np.std(quiet_region, axis=1)
        signal_region = test_data_np[:, signal_mask]
        signal_scores = np.max(np.abs(signal_region), axis=1)
        snr_scores = signal_scores / (std_scores + 1e-9)

        # Since if snr = high, then good, but if snr = low anomaly; so if higher test_loss it means anomalous 
        test_loss = snr_scores
        hard_coded_direction = "lower"

    # True labels are loaded in here:
    test_labels = np.asarray(test_labels)
    y_true = (~test_labels.astype(bool)).astype(int)

    normal_mean = np.nanmean(test_loss[y_true == 0])
    anomaly_mean = np.nanmean(test_loss[y_true == 1])

    print("\nNormal mean score:", normal_mean)
    print("Anomaly mean score:", anomaly_mean)

    # ROC assumes higher score = more anomalous.
    if anomaly_score_direction is None:
        anomaly_score_direction = hard_coded_direction

    if anomaly_score_direction == "lower":
        print("Direction: lower raw score = more anomalous, flipping scores for ROC")
        roc_scores = -test_loss
        flipped = True
    else:
        print("Direction: higher score = more anomalous")
        roc_scores = test_loss
        flipped = False

    # ROC + AUC
    fpr, tpr, thresholds = roc_curve(y_true, roc_scores)
    auc = roc_auc_score(y_true, roc_scores)

    # Choose threshold
    if threshold_method == "youden":
        scores = tpr - fpr
        best_idx = np.argmax(scores)

    elif threshold_method == "closest_top_left":
        scores = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        best_idx = np.argmin(scores)

    elif threshold_method == "fbeta_score":
        scores = []

        for i in thresholds:
            y_pred = roc_scores >= i
            y_pred = (y_pred).astype(int)

            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            scores.append(score)

        best_idx = np.argmax(scores)

    else:
        raise ValueError(
            "threshold_method must be 'youden' or 'closest_top_left' or 'fbeta_score'"
        )

    threshold = thresholds[best_idx]
    raw_threshold = -threshold if flipped else threshold

    print("\nThreshold method:", threshold_method)
    print("Optimal ROC threshold:", threshold)
    print("Optimal raw threshold:", raw_threshold)
    print("AUC:", auc)

    print("Min ROC score:", np.min(roc_scores))
    print("Max ROC score:", np.max(roc_scores))
    print("Min raw score:", np.min(test_loss))
    print("Max raw score:", np.max(test_loss))

    # y_pred: 0 = normal, 1 = anomaly
    y_pred = (roc_scores >= threshold).astype(int)
    pred_labels = y_pred == 0  # True = normal, False = anomaly

    print("Predicted anomalies:", np.sum(y_pred == 1))
    print("Predicted normals:", np.sum(y_pred == 0))

    if (
        use_autoencoder == False
        and use_water_peak_model == False
        and use_std_model == False
        and use_fwhm_model == False
    ):
        print("Normal mean NAA peak:", np.nanmean(signal_scores[y_true == 0]))
        print("Anomaly mean NAA peak:", np.nanmean(signal_scores[y_true == 1]))
        print("Normal mean noise std:", np.nanmean(std_scores[y_true == 0]))
        print("Anomaly mean noise std:", np.nanmean(std_scores[y_true == 1]))
        print("Normal mean SNR:", np.nanmean(test_loss[y_true == 0]))
        print("Anomaly mean SNR:", np.nanmean(test_loss[y_true == 1]))



    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.scatter(
            fpr[best_idx],
            tpr[best_idx],
            label=f"Threshold = {threshold:.4f}"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "test_loss": test_loss,
        "roc_scores": roc_scores,
        "y_true": y_true,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc,
        "best_threshold": threshold,
        "best_raw_threshold": raw_threshold,
        "best_idx": best_idx,
        "y_pred": y_pred,              # 0 = normal, 1 = anomaly
        "pred_labels": pred_labels,    # True = normal, False = anomaly
        "direction": flipped,
        "anomaly_score_direction": anomaly_score_direction
    }


def calculate_water_fwhm(ppm, spectrum, peak_min=4.5, peak_max=4.9):
    ppm = np.asarray(ppm)
    spectrum = np.asarray(spectrum)

    # Sort ppm so left/right searches are physically meaningful
    sort_idx = np.argsort(ppm)
    ppm = ppm[sort_idx]
    spectrum = spectrum[sort_idx]

    spec_abs = np.abs(spectrum)

    # Only use 4.5-4.9 ppm to locate the water peak maximum
    peak_mask = (ppm >= peak_min) & (ppm <= peak_max)

    if not np.any(peak_mask):
        return 0.0

    peak_region_indices = np.where(peak_mask)[0]
    peak_idx = peak_region_indices[np.argmax(spec_abs[peak_mask])]

    # Require a sufficiently large water peak
    if spec_abs[peak_idx] <= 1.0:
        return 0.0

    baseline = np.min(spec_abs)
    spec_corrected = spec_abs - baseline

    peak_height = spec_corrected[peak_idx]

    if peak_height <= 0:
        return 0.0

    half_max = peak_height / 2.0

    # Search the full spectrum for half-max intersections
    left_candidates = np.where(spec_corrected[:peak_idx] < half_max)[0]
    right_candidates = np.where(spec_corrected[peak_idx:] < half_max)[0]

    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return 0.0

    left_idx = left_candidates[-1]
    right_idx = peak_idx + right_candidates[0]

    left_ppm = np.interp(
        half_max,
        [spec_corrected[left_idx], spec_corrected[left_idx + 1]],
        [ppm[left_idx], ppm[left_idx + 1]]
    )

    right_ppm = np.interp(
        half_max,
        [spec_corrected[right_idx], spec_corrected[right_idx - 1]],
        [ppm[right_idx], ppm[right_idx - 1]]
    )

    return abs(right_ppm - left_ppm)
