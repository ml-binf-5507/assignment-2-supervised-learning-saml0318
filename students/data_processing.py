"""
Data loading and preprocessing functions for heart disease dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(filepath):
    """
    Load the heart disease dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the heart disease CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw dataset with all features and targets
        
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    ValueError
        If the CSV is empty or malformed
        
    Examples
    --------
    >>> df = load_heart_disease_data('data/heart_disease_uci.csv')
    >>> df.shape
    (270, 15)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath, na_values=["?", "NA", "N/A", ""])
    except pd.errors.EmptyDataError as exc:
        raise ValueError("CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("CSV file is malformed and could not be parsed.") from exc

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df


def preprocess_data(df):
    """
    Handle missing values, encode categorical variables, and clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset
    """
    df_clean = df.copy()
    df_clean = df_clean.replace("?", np.nan)

    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            coerced = pd.to_numeric(df_clean[col], errors="coerce")
            if coerced.notna().sum() > 0:
                df_clean[col] = coerced

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    if len(numeric_cols) > 0:
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
            df_clean[numeric_cols].median()
        )

    for col in categorical_cols:
        mode = df_clean[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df_clean[col] = df_clean[col].fillna(fill_value)

    if len(categorical_cols) > 0:
        df_clean = pd.get_dummies(df_clean, columns=list(categorical_cols), drop_first=False)

    bool_cols = df_clean.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df_clean[bool_cols] = df_clean[bool_cols].astype(int)

    return df_clean


def prepare_regression_data(df, target='chol'):
    """
    Prepare data for linear regression (predicting serum cholesterol).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'chol')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector
    """
    target_candidates = [target, "chol", "cholesterol"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"Target column not found. Checked: {target_candidates}")

    df_reg = df.copy()
    y = pd.to_numeric(df_reg[target_col], errors="coerce")
    valid_mask = y.notna()

    X = df_reg.loc[valid_mask].drop(columns=[target_col], errors="ignore")
    y = y.loc[valid_mask]

    if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
        X = pd.get_dummies(X, drop_first=False)

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def prepare_classification_data(df, target='num'):
    """
    Prepare data for classification (predicting heart disease presence).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'num')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector (binary)
    """
    target_candidates = [target, "num", "target", "heart_disease"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"Target column not found. Checked: {target_candidates}")

    df_clf = df.copy()

    y_numeric = pd.to_numeric(df_clf[target_col], errors="coerce")
    if y_numeric.isna().all():
        y_categorical = pd.Series(
            pd.Categorical(df_clf[target_col]).codes, index=df_clf.index
        )
        y = (y_categorical > 0).astype(int)
    else:
        y = (y_numeric > 0).astype(int)

    valid_mask = y.notna()

    chol_like_cols = [c for c in ["chol", "cholesterol"] if c in df_clf.columns]
    drop_cols = [target_col] + chol_like_cols
    X = df_clf.loc[valid_mask].drop(columns=drop_cols, errors="ignore")
    y = y.loc[valid_mask]

    if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
        X = pd.get_dummies(X, drop_first=False)

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        where scaler is the fitted StandardScaler
    """
    y_array = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
    unique_classes = np.unique(y_array[~pd.isna(y_array)])
    stratify = y if len(unique_classes) == 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
