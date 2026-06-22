# %%
from script_files.load_spectra import *
from script_files.train_test_balanced import *
from script_files.normalize import *
from script_files.display_spectrum import *
from script_files.display_reconstructions import *
from script_files.display_roc_curve import *
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

direction_to_this_main_file = str(Path(__file__).resolve().parent)

# Simulator
"""
Generate data using notebooks/generate_data.ipynb from
https://github.com/dennisvds/MRS-Digital-Phantom/tree/main
using the config.json files in /data.
"""

# Generate "normal" spectra
"""
Generate the "normal" spectra using the config.json file in data/normal_data

"""
#Load the normal data as Free Induction Decay (FID) objects and extract the voxel weighted concentrations of the metabolites.
normal_dir = direction_to_this_main_file + r"\data\normal_data\spectra"
normal_fids, normal_paths, normal_voxel_weighted_conc = load_fids_from_base_dir(normal_dir)


# Generate "anomalous" spectra
"""
Generate the "anomalous" spectra using the config.json file in data/anomaly_data

"""
#Load the anomalous data as Free Induction Decay (FID) objects and extract the voxel weighted concentrations of the metabolites.
anomaly_dir = direction_to_this_main_file + r"\data\anomaly_data\spectra"
anomaly_fids, anomaly_paths, anomaly_voxel_weighted_conc= load_fids_from_base_dir(anomaly_dir)

# General summary
print(f"Loaded {len(normal_fids)} normal FIDs")
print(f"Loaded {len(anomaly_fids)} anomaly FIDs")

# Convert the Free Induction Decay (FID) objects into the real part of the spectra and normalize with real = real / max_real
normal_spectra = fids_to_spectra(normal_fids)
anomaly_spectra = fids_to_spectra(anomaly_fids)

# Make labels for the respective data
normal_labels = np.ones(len(normal_spectra), dtype=bool)      # True = normal
anomaly_labels = np.zeros(len(anomaly_spectra), dtype=bool)   # False = anomaly

# Combine data sets
X = np.concatenate([normal_spectra, anomaly_spectra], axis=0)
y = np.concatenate([normal_labels, anomaly_labels], axis=0)
all_paths = normal_paths + anomaly_paths

# General summary
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Normal count:", np.sum(y))
print("Anomaly count:", np.sum(~y))

# Train test split with a balanced 50% normal / 50% anomalous data split for the test data
train_data, train_labels, train_paths, test_data, test_labels, test_paths, test_conc = train_test_balanced_function(        
    normal_spectra,
    normal_labels,
    normal_paths,
    anomaly_spectra,
    anomaly_labels,
    anomaly_paths,
    normal_voxel_weighted_conc,
    anomaly_voxel_weighted_conc
    )

# General summary
print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
print("Test normal count:", np.sum(test_labels))
print("Test anomaly count:", np.sum(~test_labels))

mean = np.mean(train_data)
std = np.std(train_data)

print(mean, std)
# Normalize with z-score so the center inputs are around zero.
train_data, test_data = normalize_data(train_data, test_data)

mean = np.mean(train_data)
std = np.std(train_data)

print(mean, std)

# Separate normal/anomalous AFTER normalization
normal_train_data = train_data
normal_test_data = tf.boolean_mask(test_data, test_labels)
anomalous_test_data = tf.boolean_mask(test_data, ~test_labels)

#%%

## To visualize what is going into the auto-encoder, and what normally is seen we visually it.

example_index = 0
ppm = plot_example_spectrum_fslmrs(normal_paths[example_index])
# %%
import tensorflow as tf
from keras import layers, Model

# Here the model for the auto-encoder is made.
# Activation of relu because ...
# Latent space of size 16 because ...
# Use of output activation = linear ... why?
# Use of dropout(0.2) because ...

class AnomalyDetector(Model):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            #layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            #layers.Dropout(0.2),
            layers.Dense(8, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(input_dim, activation="linear")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_dim = train_data.shape[1]
autoencoder = AnomalyDetector(input_dim)


# Optimizer of adam because ...
# Loss of mae because ... 

autoencoder.compile(optimizer="adam", loss="mae")

# Early stopping because ...

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

#%%

# Training on only the normal spectra
history = autoencoder.fit(
    normal_train_data,
    normal_train_data,
    epochs=100,
    batch_size=32,
    validation_data=(normal_test_data, normal_test_data),
    callbacks=[early_stop],
    shuffle=True
)

# Epoch = 100 chosen because ...
# Batch_size = 32 chosen because ...

# Showing the history of the training, this must reach a stable level (little change at last epoch)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Reconstruction loss (MAE)")

plt.legend()
plt.title("Autoencoder Training and Validation Loss")
plt.show()


# Plot reconstructed normal
plot_autoencoder_reconstructions(
    autoencoder,
    normal_test_data,
    ppm,
    num_examples=4,
    cols=2,
    rows=2
)

# Plot reconstructed anomalous
plot_autoencoder_reconstructions(
    autoencoder,
    anomalous_test_data,
    ppm,
    num_examples=4,
    cols=2,
    rows=2
)

#%%

# Calculate reconstruction loss on only the normal training data
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data).numpy()

plt.hist(train_loss, bins=50)
plt.xlabel("Normal train reconstruction loss")
plt.ylabel("Number of examples")
plt.title("Normal Training Loss Distribution")
plt.show()


# Plot the ROC curve using the selected threshold_method.

#The threshold_method ... has been chosen because ... 
roc_results = plot_roc_autoencoder(
    autoencoder,
    test_data,
    test_labels,
    example_spec=normal_paths[example_index],
    threshold_method="fbeta_score"
)

threshold = roc_results["best_threshold"]

normal_reconstructions = autoencoder.predict(normal_test_data)
normal_test_loss = tf.keras.losses.mae(normal_reconstructions, normal_test_data).numpy()

anomaly_reconstructions = autoencoder.predict(anomalous_test_data)
anomaly_test_loss = tf.keras.losses.mae(anomaly_reconstructions, anomalous_test_data).numpy()

# Plot a histogram that shows the reconstruction losses of the normal and anomalous data, with a threshold seperation line.
plt.hist(normal_test_loss, bins=50, alpha=0.6, label="Normal")
plt.hist(anomaly_test_loss, bins=50, alpha=0.6, label="Anomalous")
plt.axvline(threshold, linestyle="--", label="Threshold")
plt.xlabel("Reconstruction loss")
plt.ylabel("Number of examples")
plt.legend()
plt.title("Normal vs Anomalous Reconstruction Loss")
plt.show()

# %%
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score
)

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    print("Accuracy:", accuracy_score(labels, predictions))
    print("Precision:", precision_score(labels, predictions))
    print("Recall:", recall_score(labels, predictions))

preds = predict(autoencoder, test_data, threshold).numpy()

print_stats(preds, test_labels)

cm = confusion_matrix(test_labels, preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Anomaly", "Normal"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# %%

# SNR Model

roc_results_snr = plot_roc_autoencoder(
    autoencoder,
    test_data,
    test_labels,
    example_spec=normal_paths[example_index],
    use_autoencoder=False,
    threshold_method="fbeta_score",
    # Change this to "lower" only if your anomaly definition is low-SNR spectra. (To flip the ROC curve)
    anomaly_score_direction="higher"
)

snr_scores = roc_results_snr["test_loss"]
pred_labels = roc_results_snr["pred_labels"]
threshold = roc_results_snr["best_raw_threshold"]

print_stats(pred_labels, test_labels)

normal_snr = snr_scores[test_labels]
anomaly_snr = snr_scores[~test_labels]



print("Mean of the anomaly_snr:", np.mean(anomaly_snr))
print("Maximum of the anomaly_snr:", max(anomaly_snr))

plt.figure(figsize=(8, 5))
plt.hist(normal_snr, bins=50, alpha=0.6, label="Normal")
plt.hist(anomaly_snr, bins=50, alpha=0.6, label="Anomaly")
plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.4f}")
plt.xlabel("SNR")
plt.ylabel("Count")
plt.title("SNR Distribution")
plt.legend()
plt.grid(True)
plt.show()

cm = confusion_matrix(test_labels, pred_labels)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Anomaly", "Normal"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - SNR Model")
plt.show()

# %%

# Water peak model

roc_results_water = plot_roc_autoencoder(
    autoencoder,
    test_data,
    test_labels,
    example_spec=normal_paths[example_index],
    use_autoencoder = False,
    use_water_peak_model=True,
    threshold_method="fbeta_score"
)

water_scores = roc_results_water["test_loss"]
pred_labels = roc_results_water["pred_labels"]
threshold = roc_results_water["best_raw_threshold"]

print_stats(pred_labels, test_labels)

normal_water = water_scores[test_labels]
anomaly_water = water_scores[~test_labels]

plt.figure(figsize=(8, 5))
plt.hist(normal_water, bins=50, alpha=0.6, label="Normal")
plt.hist(anomaly_water, bins=50, alpha=0.6, label="Anomaly")
plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.4f}")
plt.xlabel("Water")
plt.ylabel("Count")
plt.title("water")
plt.legend()
plt.grid(True)
plt.show()

cm = confusion_matrix(test_labels, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Anomaly", "Normal"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Water Model")
plt.show()

# FWHM Model

roc_results_fwhm = plot_roc_autoencoder(
    autoencoder,
    test_data,
    test_labels,
    example_spec=normal_paths[example_index],
    use_autoencoder = False,
    use_std_model= False,
    use_fwhm_model=True,
    use_water_peak_model=False,
    threshold_method="fbeta_score"
)
 
fwhm_scores = roc_results_fwhm["test_loss"]
pred_labels = roc_results_fwhm["pred_labels"]
threshold = roc_results_fwhm["best_raw_threshold"]

print_stats(pred_labels, test_labels)

normal_fwhm = fwhm_scores[test_labels]
anomaly_fwhm = fwhm_scores[~test_labels]

plt.figure(figsize=(8, 5))
plt.hist(normal_fwhm, bins=50, alpha=0.6, label="Normal")
plt.hist(anomaly_fwhm, bins=50, alpha=0.6, label="Anomaly")
plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.4f}")
plt.xlabel("FWHM score")
plt.ylabel("Count")
plt.title("FWHM Distribution")
plt.legend()
plt.grid(True)
plt.show()

cm = confusion_matrix(test_labels, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Anomaly", "Normal"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - FWHM Model")
plt.show()


# %%
#Fitting:
from script_files.fitting_script import run_fsl_fit_evaluation
import winsound

basis = r"data/LCModel_Universal_Philips_UnEdited_PRESS_TE35.BASIS"

fitting_results = run_fsl_fit_evaluation(
    test_paths,
    test_conc,
    test_labels,
    basis,
    Fitargs=None,
    scaling="internal",
    use_existing_csv=True
)

winsound.PlaySound(
    r"C:\Windows\Media\notify.wav",
    winsound.SND_FILENAME
)

winsound.PlaySound(
    r"C:\Windows\Media\tada.wav",
    winsound.SND_FILENAME
)

#%%
from script_files.display_comparison_metabolites import plot_stat_vs_stat


#%%
import pandas as pd


final_results_of_fitting = fitting_results["final_results_of_fitting"]

if "autoencoder_loss" not in final_results_of_fitting.columns:

    test_reconstructions = autoencoder.predict(test_data)

    autoencoder_loss = tf.keras.losses.mae(
        test_reconstructions,
        test_data
    ).numpy()

    autoencoder_loss_final_data = pd.DataFrame({"path": test_paths, "autoencoder_loss": autoencoder_loss})

    final_results_of_fitting = final_results_of_fitting.merge(
        autoencoder_loss_final_data,
        on="path",
        how="left"
    )

    directiontothescript = Path(__file__).resolve().parent
    directiontothecsvfiles = directiontothescript / "data" / "csv_files"

    final_results_of_fitting.to_csv(directiontothecsvfiles / "final_results_of_fitting.csv", index=False)

    print("+ autoencoder_loss to final_results_of_fitting")

else:
    print("autoencoder_loss exists")

if "fwhm_score" not in final_results_of_fitting.columns:
    fwhm_final_data = pd.DataFrame({"path": test_paths, "fwhm_score": fwhm_scores})

    final_results_of_fitting = final_results_of_fitting.merge(
        fwhm_final_data,
        on="path",
        how="left"
    )

    directiontothescript = Path(__file__).resolve().parent
    directiontothecsvfiles = directiontothescript / "data" / "csv_files"

    final_results_of_fitting.to_csv(
        directiontothecsvfiles / "final_results_of_fitting.csv",
        index=False
    )


else:
    print("fwhm_score already exists. Skipping merge and save.")


results = plot_stat_vs_stat(
    csv_file="data/csv_files/final_results_of_fitting.csv",
    stat1="snr",
    stat2="crlb_percent"
)

results = plot_stat_vs_stat(
    csv_file="data/csv_files/final_results_of_fitting.csv",
    stat1="fwhm_score",
    stat2="crlb_percent"
)

results = plot_stat_vs_stat(
    csv_file="data/csv_files/final_results_of_fitting.csv",
    stat1="autoencoder_loss",
    stat2="crlb_percent"
)

# %%