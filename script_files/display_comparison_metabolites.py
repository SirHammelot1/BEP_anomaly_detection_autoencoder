import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr


def plot_stat_vs_stat(
    csv_file,
    stat1,
    stat2,
    column_metabolites_in_fitting_results="metabolite"
):
    final_results_of_fitting_csv = pd.read_csv(csv_file)
    final_results_of_fitting_csv = final_results_of_fitting_csv.replace([np.inf, -np.inf], np.nan)

    metabolites = sorted(final_results_of_fitting_csv[column_metabolites_in_fitting_results].dropna().unique())

    correlation_rows = []

    for metabolite in metabolites:

        final_results_of_only_this_metbolite = final_results_of_fitting_csv[final_results_of_fitting_csv[column_metabolites_in_fitting_results] == metabolite]

        metab_final_results_of_fitting_csv = final_results_of_only_this_metbolite.dropna(subset=[stat1, stat2])

        if len(metab_final_results_of_fitting_csv) >= 2:
            pearson_corr, pearson_p = pearsonr(
                metab_final_results_of_fitting_csv[stat1],
                metab_final_results_of_fitting_csv[stat2]
            )

            spearman_corr, spearman_p = spearmanr(
                metab_final_results_of_fitting_csv[stat1],
                metab_final_results_of_fitting_csv[stat2]
            )
        else:
            pearson_corr = np.nan
            pearson_p = np.nan
            spearman_corr = np.nan
            spearman_p = np.nan

        correlation_rows.append({
            "metabolite": metabolite,
            "correlation (pearson)": pearson_corr,
            "p-value (pearson)": pearson_p,
            "correlation (spearman)": spearman_corr,
            "p-value (spearman)": spearman_p,
            "n": len(metab_final_results_of_fitting_csv)
})

    correlation_final_results_of_fitting_csv = pd.DataFrame(correlation_rows)

    print(f"\nCorrelation between {stat1} and {stat2} per metabolite:")
    print(correlation_final_results_of_fitting_csv.to_string(index=False))

    return {
        "data": final_results_of_fitting_csv,
        "correlation_final_results_of_fitting_csv": correlation_final_results_of_fitting_csv
    }