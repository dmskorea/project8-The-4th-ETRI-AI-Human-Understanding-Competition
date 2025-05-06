from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def visualize_correlation(title, corr, result_dir):
    """
    Visualize the correlation matrix and save it as an image.
    """

    row_len = len(corr.index) + 3
    col_len = len(corr.columns) - 3
    plt.figure(figsize=(col_len, row_len), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, 
                cbar_kws={"shrink": .8}, annot_kws={"size": 14})
    plt.title(title, fontdict={"fontsize": 30})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(result_dir / (title + ".png"))
    plt.close()


if __name__ == "__main__":

    dataset_version = "v4"

    result_dir = Path(f"./eda/{dataset_version}/")
    result_dir.mkdir(parents=True, exist_ok=True)

    KEY_COLS = ["subject_id", "lifelog_date"]
    TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3"]

    X_train = pd.read_parquet(f"./results/{dataset_version}/X_train.parquet")
    Y_train = pd.read_parquet(f"./results/{dataset_version}/Y_train.parquet")

    merge_df = pd.merge(X_train, Y_train, on=KEY_COLS, how="inner")

    merge_df = merge_df.drop(columns=KEY_COLS + ["sleep_date"])
    numeric_df = merge_df.select_dtypes(include=["int", "float"])
    
    corr = numeric_df.corr(method="pearson")
    kendall_corr = numeric_df.corr(method="kendall")
    spearman_corr = numeric_df.corr(method="spearman")

    def inner(corr, method, result_dir):
        target_corr = corr.loc[TARGET_COLS, TARGET_COLS]
        visualize_correlation(f"target_corr [{method}]", target_corr, result_dir)
        feature_corr = corr.loc[TARGET_COLS, :].drop(columns=TARGET_COLS)
        visualize_correlation(f"feature_corr [{method}]", feature_corr, result_dir)
    
    inner(corr, "pearson", result_dir)
    inner(kendall_corr, "kendall", result_dir)
    inner(spearman_corr, "spearman", result_dir)
