from fsl_mrs.utils import mrs_io
from fsl_mrs.utils import preproc as proc
import matplotlib.pyplot as plt
from fsl_mrs.utils import fitting, misc, plotting
import pandas as pd
import numpy as np
import os
from pathlib import Path

def run_fsl_fit_evaluation(
    test_paths,
    test_conc,
    test_labels,
    basis,
    Fitargs=None,
    scaling="internal",
    ppm_region_of_naa_peak_ppm = (1.9, 2.1),
    ppm_region_of_quiett_region_ppm = (9.0, 10.0),
    use_existing_csv = True
):
    directiontothescript = Path(__file__).resolve().parent
    directiontothecsvfiles = directiontothescript.parent / "data" / "csv_files"

    results_csv = directiontothecsvfiles / "final_results_of_fitting.csv"


    if use_existing_csv == True:
        print("Loading existing CSV files...")

        return {"final_results_of_fitting": pd.read_csv(results_csv)}


    #Fit with Newton algorithm
    if Fitargs is None:
        Fitargs = {
            "ppmlim": [0.2, 4.2],
            "method": "Newton",
            "baseline_order": 4,
            "model": "voigt"
        }

    def calculate_snr(mrs, ppm_region_of_naa_peak_ppm=(1.9, 2.1), ppm_region_of_quiett_region_ppm=(9.0, 10.0)):
        ppm_axis = mrs.getAxes()
        spectrum = np.real(mrs.get_spec())

        signal_naapeak = (ppm_axis >= ppm_region_of_naa_peak_ppm[0]) & (ppm_axis <= ppm_region_of_naa_peak_ppm[1])
        noise_in_quieet = (ppm_axis >= ppm_region_of_quiett_region_ppm[0]) & (ppm_axis <= ppm_region_of_quiett_region_ppm[1])

        signal = np.max(np.abs(spectrum[signal_naapeak]))
        noise = np.std(spectrum[noise_in_quieet])

        return signal / (noise + 1e-8)

    combined_list_of_results = []


    for i in range(len(test_paths)):

        spectrum_path = test_paths[i]
        print(f"Processing: {test_paths[i]} | Status: {i + 1} / {len(test_paths)}")

        supp_data = mrs_io.read_FID(spectrum_path)
        mrs = supp_data.mrs(basis_file = basis)
        mrs.processForFitting()

        # Separate macromolecule from the rest (it will have its own lineshape parameters)
        metab_groups = []
        
        res = fitting.fit_FSLModel(mrs,**Fitargs)

        combinationList = [
        ["NAA", "NAAG"],
        ["Glu", "Gln"],
        ["GPC", "PCh"],
        ["Cr", "PCr"],
        ["Glc", "Tau"]]

        res.combine(combinationList)

        from fsl_mrs.utils import quantify
        te = supp_data.hdr_ext['EchoTime']
        tr = supp_data.hdr_ext['RepetitionTime']
        q_info = quantify.QuantificationInfo(te,
                                            tr,
                                            mrs.names,
                                            mrs.centralFrequency / 1E6)
        q_info.set_fractions({'WM':0.45,'GM':0.45,'CSF':0.1})

        fitted = res.getConc(scaling=scaling, function=None).mean()

        true = pd.Series(test_conc[i], name="true")

        fitted.name = "fitted"
    
        snr = calculate_snr(mrs, ppm_region_of_naa_peak_ppm=ppm_region_of_naa_peak_ppm, ppm_region_of_quiett_region_ppm=ppm_region_of_quiett_region_ppm)

        crlb_percent = res.getUncertainties(type="percentage")
        crlb_percent = crlb_percent.mean(axis=1)
        crlb_percent.name = "crlb_percent"

        crlb = (crlb_percent / 100) * np.abs(fitted)
        crlb.name = "crlb"

        #Remove the metabolites in the fitted dataset and only keep the comparison of the true data
        comparison = pd.concat([true, fitted, crlb, crlb_percent], axis=1).dropna(subset=["true", "fitted"],how="any")

        if "Cr" in comparison.index and "PCr" in comparison.index:
            cr_pcr_fitted = comparison.loc["Cr", "fitted"] + comparison.loc["PCr", "fitted"]
            cr_pcr_true = comparison.loc["Cr", "true"] + comparison.loc["PCr", "true"]
        else:
            cr_pcr_fitted = np.nan
            cr_pcr_true = np.nan

        comparison["spectrum_number"] = i                                                                                                    # Index of the spectrum
        comparison["path_to_spectrum"] = spectrum_path                                                                                                  # Path to the corresponding spectrum
        comparison["metabolite"] = comparison.index                                                                                         # Name of the metabolite 
        comparison["label"] = test_labels[i]                                                                                                # Ground-truth label 
        comparison["snr"] = snr                                                                                                             # Signal-to-noise ratio of corresponding spectrum
        comparison["error"] = (comparison["fitted"] - comparison["true"])                                                                   # Error between the fitted and the true concentrations
        comparison["percent_error"]= np.where(comparison["true"] != 0, 100 * comparison["error"] / comparison["true"], np.nan)              # Percentage error between the fitted and the true concentrations
        comparison["absolute_error"] = np.abs(comparison["error"])                                                                          # No minus signs absolute error between the fitted and the true concentrations
        comparison["abs_percent_error"] = np.abs(comparison["percent_error"])                                                               # No minus signs percentage error between the fitted and the true concentrations
        comparison["dataset"] = np.where(test_labels[i], "normal", "anomaly")                                                               # Dataset label with anomaly/normal
        comparison["fitted_over_cr_pcr"] = np.where(cr_pcr_fitted != 0, comparison["fitted"] / cr_pcr_fitted, np.nan)                       # Fitted concentration after normalized with the fitted cr_pcr
        comparison["true_over_cr_pcr"] = np.where(cr_pcr_true != 0, comparison["true"] / cr_pcr_true, np.nan)                               # True concentration after normalized with the true cr_pcr
        comparison["crlb_over_cr_pcr"] = np.where(cr_pcr_fitted != 0, comparison["crlb"] / cr_pcr_fitted, np.nan)                           # CRLB normalized by fitted cr_pcr
        comparison["error_over_cr_pcr"] = (comparison["fitted_over_cr_pcr"] - comparison["true_over_cr_pcr"])                               # Error between the fitted and the true concentrations after normalization with cr_pcr
        comparison["absolute_error_over_cr_pcr"] = np.abs(comparison["error_over_cr_pcr"])                                                  # No minus signs absolute error between the fitted and the true concentrations after normalization with cr_pcr
        combined_list_of_results.append(comparison.reset_index(drop=True))

    pandas_csv_File_results = pd.concat(combined_list_of_results, ignore_index=True)
    pandas_csv_File_results = pandas_csv_File_results.replace([np.inf, -np.inf], np.nan)

    print(pandas_csv_File_results)

    # Saving final results to csv
    directiontothescript = Path(__file__).resolve().parent
    directiontothecsvfiles = directiontothescript.parent / "data" / "csv_files"
    directiontothecsvfiles.mkdir(parents=True, exist_ok=True)
    pandas_csv_File_results.to_csv(directiontothecsvfiles / "final_results_of_fitting.csv", index=False)

    print("Done")

    return {
        "final_results_of_fitting": pandas_csv_File_results
    }