# BEP_anomaly_detection_autoencoder

## Directory structure

```
BachelorEndProject_AnomalyDetection/
├── data/                                           # Main folder to store the data
│   ├── csv_files/                                  # Folder where the csv_files will load into when running main.py
│   ├── normal_data/                                # Data for the normal spectra
│       ├── spectra/                                # Data folder for the spectra
│       ├── config.json/                            # The config file used to create the spectra with MRS-Digital-Phantom
│   ├── anomaly_data/                               # Data for the anomalous spectra
│       ├── spectra/                                # Data folder for the spectra
│       ├── config.json/                            # The config file used to create the spectra with MRS-Digital-Phantom
├── script_files/                                   # Folder containing scripts to run main.py
│   ├── display_comparison_metabolites.py           # Displays the correlations and figures for correlation
│   ├── display_reconstructions.py                  # Displays the reconstructed spectra from the auto encoder
│   ├── display_roc_curve.py                        # Displays the ROC curves in the main file
│   ├── display_spectrum.py                         # Displays the spectrum with corresponding ppm values
│   ├── fitting_script.py                           # Fits the spectra using the FSL MRS package
│   ├── load_spectra.py                             # Loads in the spectra from the data folder
│   ├── normalize.py                                # Normalizes the spectra
│   ├── train_test_balanced.py                      # Does the train test split with 50/50 normal and anomalous data
├── main.py                                         # Main script that runs the scripts
├── requirements.txt                                # Python dependencies
└── README.md                                       # Project documentation (this file)
```

## Usage

Before anything, make sure requirements.txt is used to install the dependencies.

Using the config.json files, run the generate_data.ipynb from the MRS-Digital-main from https://github.com/dennisvds/MRS-Digital-Phantom.
D.M.J. van de Sande, A.T. Gudmundson, S. Murali-Manohar, C.W. Davies-Jenkins, D. Simicic, G. Simegn, İ. Özdemir, S. Amirrajab, J.P. Merkofer, H.J. Zöllner, G. Oeltzschner, R.A.E. Edden A Digital Phantom for MR Spectroscopy Data Simulation. 

Then put the spectra into the corresponding spectra/ folder and run the main.py file.
