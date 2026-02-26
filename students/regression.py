"""
Linear regression functions for predicting cholesterol using ElasticNet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback for minimal environments
    sns = None


def train_elasticnet_grid(X_train, y_train, l1_ratios, alphas):
    """
    Train ElasticNet models over a grid of hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training feature matrix
    y_train : np.ndarray or pd.Series
        Training target vector
    l1_ratios : list or np.ndarray
        L1 ratio values to test (0 = L2 only, 1 = L1 only)
    alphas : list or np.ndarray
        Regularization strength values to test
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['l1_ratio', 'alpha', 'r2_score', 'model']
        Contains R² scores for each parameter combination on training data
    """
    results = []

    for l1_ratio in l1_ratios:
        for alpha in alphas:
            model = ElasticNet(
                l1_ratio=l1_ratio,
                alpha=alpha,
                max_iter=5000,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            train_r2 = r2_score(y_train, y_pred_train)

            results.append(
                {
                    "l1_ratio": l1_ratio,
                    "alpha": alpha,
                    "r2_score": float(train_r2),
                    "model": model,
                }
            )

    return pd.DataFrame(results)


def create_r2_heatmap(results_df, l1_ratios, alphas, output_path=None):
    """
    Create a heatmap of R² scores across l1_ratio and alpha parameters.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from train_elasticnet_grid
    l1_ratios : list or np.ndarray
        L1 ratio values used in grid
    alphas : list or np.ndarray
        Alpha values used in grid
    output_path : str, optional
        Path to save figure. If None, returns figure object
        
    Returns
    -------
    matplotlib.figure.Figure
        The heatmap figure
    """
    heatmap_data = results_df.pivot_table(
        index="alpha",
        columns="l1_ratio",
        values="r2_score",
        aggfunc="mean",
    )
    heatmap_data = heatmap_data.reindex(index=alphas, columns=l1_ratios)

    fig, ax = plt.subplots(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": "R² Score"},
            ax=ax,
        )
    else:
        matrix = heatmap_data.to_numpy(dtype=float)
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, label="R² Score")
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticklabels(heatmap_data.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white")

    ax.set_xlabel("L1 Ratio")
    ax.set_ylabel("Alpha")
    ax.set_title("ElasticNet Training R² Across Hyperparameters")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig


def get_best_elasticnet_model(X_train, y_train, X_test, y_test, 
                               l1_ratios=None, alphas=None):
    """
    Find and train the best ElasticNet model on test data.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training target
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_test : np.ndarray or pd.Series
        Test target
    l1_ratios : list, optional
        L1 ratio values to test. Default: [0.1, 0.3, 0.5, 0.7, 0.9]
    alphas : list, optional
        Alpha values to test. Default: [0.001, 0.01, 0.1, 1.0, 10.0]
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': fitted ElasticNet model
        - 'best_l1_ratio': best l1 ratio
        - 'best_alpha': best alpha
        - 'train_r2': R² on training data
        - 'test_r2': R² on test data
        - 'results_df': full results DataFrame
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    results_df = train_elasticnet_grid(
        X_train=X_train,
        y_train=y_train,
        l1_ratios=l1_ratios,
        alphas=alphas,
    )

    test_scores = []
    for model in results_df["model"]:
        y_pred_test = model.predict(X_test)
        test_scores.append(float(r2_score(y_test, y_pred_test)))

    results_df = results_df.copy()
    results_df["test_r2"] = test_scores

    best_idx = results_df["test_r2"].idxmax()
    best_row = results_df.loc[best_idx]
    best_model = best_row["model"]

    return {
        "model": best_model,
        "best_l1_ratio": float(best_row["l1_ratio"]),
        "best_alpha": float(best_row["alpha"]),
        "train_r2": float(best_row["r2_score"]),
        "test_r2": float(best_row["test_r2"]),
        "results_df": results_df,
    }
