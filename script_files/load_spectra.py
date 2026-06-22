import os
import nibabel as nib
import numpy as np
import json
from fsl_mrs.core import nifti_mrs

def load_fids_from_base_dir(base_dir):
    fids = []
    file_paths = []
    voxel_weigthed = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, "total.nii.gz")
            file_concentration = os.path.join(folder_path, "concentrations.json")
            if os.path.exists(file_path):
                img = nib.load(file_path)
                data = np.asanyarray(img.dataobj)
                fid = np.squeeze(data)
                fids.append(fid)
                file_paths.append(file_path)
            if os.path.exists(file_concentration):
                with open(file_concentration, "r") as f:
                    data = json.load(f)
                voxel_weigthed.append(data["voxel_weighted"])

    return fids, file_paths, voxel_weigthed


def fids_to_spectra(fids):
    spectra = []

    for fid in fids:
        spectrum = np.fft.fftshift(np.fft.fft(fid))
        real = np.real(spectrum)

        max_real = np.max(real)
        if max_real != 0:
            real = real / max_real

        spectra.append(real)

    return np.array(spectra, dtype="float32")