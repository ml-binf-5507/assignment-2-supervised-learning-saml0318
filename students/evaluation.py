"""
Model evaluation functions: metrics and ROC/PR curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, r2_score
)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score for regression.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True target values
    y_pred : np.ndarray or pd.Series
        Predicted target values
        
    Returns
    -------
    float
        R² score (between -inf and 1, higher is better)
    """
    return float(r2_score(y_true, y_pred))


def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred : np.ndarray or pd.Series
        Predicted binary labels
        
    Returns
    -------
    dict
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def calculate_auroc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the ROC Curve (AUROC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUROC score (between 0 and 1)
    """
    return float(roc_auc_score(y_true, y_pred_proba))


def calculate_auprc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUPRC score (between 0 and 1)
    """
    return float(average_precision_score(y_true, y_pred_proba))


def generate_auroc_curve(y_true, y_pred_proba, model_name="Model", 
                        output_path=None, ax=None):
    """
    Generate and plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUROC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig if created_ax else (fig, ax)


def generate_auprc_curve(y_true, y_pred_proba, model_name="Model",
                        output_path=None, ax=None):
    """
    Generate and plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    baseline = np.mean(y_true)

    ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AUPRC={auprc:.3f})")
    ax.axhline(
        y=baseline,
        linestyle="--",
        color="gray",
        linewidth=1,
        label=f"Baseline={baseline:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig if created_ax else (fig, ax)


def plot_comparison_curves(y_true, y_pred_proba_log, y_pred_proba_knn,
                          output_path=None):
    """
    Plot ROC and PR curves for both logistic regression and k-NN side by side.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba_log : np.ndarray or pd.Series
        Predicted probabilities from logistic regression
    y_pred_proba_knn : np.ndarray or pd.Series
        Predicted probabilities from k-NN
    output_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2 subplots (ROC and PR curves)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    generate_auroc_curve(
        y_true=y_true,
        y_pred_proba=y_pred_proba_log,
        model_name="Logistic Regression",
        ax=axes[0],
    )
    generate_auroc_curve(
        y_true=y_true,
        y_pred_proba=y_pred_proba_knn,
        model_name="k-NN",
        ax=axes[0],
    )
    axes[0].set_title("ROC Comparison")

    generate_auprc_curve(
        y_true=y_true,
        y_pred_proba=y_pred_proba_log,
        model_name="Logistic Regression",
        ax=axes[1],
    )
    generate_auprc_curve(
        y_true=y_true,
        y_pred_proba=y_pred_proba_knn,
        model_name="k-NN",
        ax=axes[1],
    )
    axes[1].set_title("PR Comparison")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig