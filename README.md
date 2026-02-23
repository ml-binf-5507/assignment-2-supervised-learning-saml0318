[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/7SCRFDcv)
# Assignment 2: Machine Learning for Heart Disease Prediction

## Overview

In this assignment, you will implement machine learning models to:

1. **Linear Regression**: Predict serum cholesterol (`chol`) using ElasticNet with hyperparameter tuning
2. **Logistic Regression**: Classify heart disease presence with best parameter selection
3. **k-Nearest Neighbors**: Classify heart disease presence with best parameter selection

## Dataset

The [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) contains medical measurements for patients with and without heart disease.

**Download the dataset:**
1. **Option A (Blackboard)**: Download `heart_disease_uci.csv` from the course Blackboard page
2. **Option B (UCI Repository)**: Download from [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

**Setup:**
- Create a `data/` folder in the project root (if it doesn't exist)
- Place `heart_diease_uci.csv` in the `data/` directory
- Your folder structure should look like:
  ```
  assn2-supervised-learning/
  ├── data/
  │   └── heart_disease_uci.csv
  ├── students/
  ├── tests/
  └── README.md
  ```

## Tasks

### Task 1: Linear Regression (30 points)

Predict serum cholesterol (`chol`) using ElasticNet regression with grid search over:
- **l1_ratio**: [0.3, 0.5, 0.7]
- **alpha**: [0.01, 0.1, 1.0]

*This creates 3×3 = 9 models to evaluate*

**Deliverables:**
- Heatmap of R² scores showing performance across parameter combinations
- Best l1_ratio and alpha values
- R² score on test set
- Brief interpretation of results

### Task 2: Logistic Regression (30 points)

Classify heart disease (binary) using Logistic Regression with hyperparameter tuning over:
- **C**: [0.001, 0.01, 0.1, 1, 10, 100]
- **penalty**: ['l2']
- **solver**: ['lbfgs']

**Deliverables:**
- Best C value and other parameters
- AUROC curve with AUC score
- AUPRC curve with AP score
- Interpretation of model performance

### Task 3: k-Nearest Neighbors (30 points)

Classify heart disease (binary) using k-NN with hyperparameter tuning over:
- **n_neighbors (k)**: [3, 5, 7, 9, 11]
- **weights**: ['uniform', 'distance']
- **metric**: ['euclidean']

*This creates 5×2×1 = 10 models to evaluate*

**Deliverables:**
- Best k value and other parameters
- AUROC curve with AUC score
- AUPRC curve with AP score
- Interpretation of model performance

### Task 4: Model Comparison (10 points)

Compare the logistic regression and k-NN models:
- Create side-by-side plots of AUROC and AUPRC curves
- Discuss which model performs better and why
- Comment on any differences in their strengths/weaknesses

## Getting Started

1. **Clone the repository** from GitHub Classroom
2. **Place dataset** in `data/heart.csv` (see Dataset section above)
3. **Read the resources** listed at the bottom of this file
4. **Implement functions** in the `students/` modules
5. **Test locally** with `pytest tests/ -v` (see Testing section below)
6. **Push your code** when ready (autograding runs on GitHub)

## Implementation Structure

### Module Organization

```
students/
├── data_processing.py      # Data loading and preprocessing
├── regression.py           # ElasticNet functions
├── classification.py       # Logistic Regression and k-NN functions
└── evaluation.py           # Metrics and curve plotting
```

### Key Functions to Implement

**Data Processing:**
- `load_heart_disease_data()` - Load CSV file with proper data types
- `preprocess_data()` - Handle missing values, encode categoricals
- `prepare_regression_data()` - Prepare serum cholesterol (`chol`) prediction data
- `prepare_classification_data()` - Prepare heart disease classification data
- `split_and_scale()` - Train/test split and standardization

**Regression:**
- `train_elasticnet_grid()` - Grid search ElasticNet models
- `create_r2_heatmap()` - Visualize R² across parameters
- `get_best_elasticnet_model()` - Select best model

**Classification:**
- `train_logistic_regression_grid()` - Grid search logistic regression
- `train_knn_grid()` - Grid search k-NN
- `get_best_logistic_regression()` - Select best logistic regression
- `get_best_knn()` - Select best k-NN model

**Evaluation:**
- `calculate_r2_score()` - Regression R² metric
- `calculate_classification_metrics()` - Accuracy, precision, recall, F1
- `calculate_auroc_score()` - AUROC metric
- `calculate_auprc_score()` - AUPRC metric
- `generate_auroc_curve()` - Plot ROC curve
- `generate_auprc_curve()` - Plot PR curve

## Getting Started

1. **Clone the repository** from GitHub Classroom
2. **Download the heart disease dataset** and place in `data/`
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Validate your setup**:
   ```bash
   python validate_submission.py
   ```
5. **Run tests**:
   ```bash
   pytest tests/ -v
   ```
6. **Implement functions** in the `students/` module
7. **Create a Jupyter notebook** (optional) to demonstrate your analysis

## Submission

1. Implement all functions in the `students/` module
2. Pass all autograded tests
3. Provide clear code comments and documentation
4. Push your changes to GitHub

## Autograding

Your submission will be automatically tested for:

- ✓ **Data processing**: Correct data loading, preprocessing, and feature engineering
- ✓ **ElasticNet grid search**: Grid search over l1_ratio and alpha, heatmap generation, R² validation (> -0.5)
- ✓ **Logistic regression**: Hyperparameter tuning, AUC performance (> 0.6 on test set)
- ✓ **k-NN tuning**: Grid search over k values, AUC performance (> 0.6 on test set)
- ✓ **Evaluation metrics**: ROC curves, PR curves, proper labels and ranges
- ✓ **Code quality and correctness**: Functions are implemented, not stubbed out

**Important**: Tests validate not just that functions exist, but that they produce **meaningful results**. For example:
- Models must achieve AUC > 0.6 (better than random guessing)
- Grid search must vary parameters and find best model
- Curves must be properly labeled and formatted
- Metrics must be in valid ranges


## Important Notes

- **Use scikit-learn** for all model implementations
- **Feature scaling**: Scale features for k-NN (they're distance-based)
- **Test set evaluation**: Use test set ONLY for final evaluation, not hyperparameter selection
- **Cross-validation**: GridSearchCV uses cross-validation on training data
- **Random seed**: Use `random_state=42` for reproducibility
- **Binary classification**: Ensure `num` target is binary (0/1) after binarization

## Testing Your Code

### Local Testing

Before pushing, test your code locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_submission.py::TestDataProcessing -v

# Run specific test
pytest tests/test_submission.py::TestRegression::test_elasticnet_grid_returns_dataframe -v
```

**Expected output**: Tests should pass (✓) or skip if function not implemented (⊘)

### Autograding

When you push to GitHub, autograding runs automatically:
- Tests run on a fresh clone of your repository
- Your `data/heart.csv` must be in the correct location
- Autograding feedback appears in GitHub Classroom

## Resources

- **[PANDAS_CHEATSHEET.md](PANDAS_CHEATSHEET.md)** - Common pandas operations
- **[SKLEARN_REFERENCE.md](SKLEARN_REFERENCE.md)** - scikit-learn quick reference
- **[DATA_GUIDE.md](DATA_GUIDE.md)** - Dataset column descriptions
- **[notebooks/example_workflow.ipynb](notebooks/example_workflow.ipynb)** - Complete working example
- **[AUTOGRADING_STRATEGY.md](AUTOGRADING_STRATEGY.md)** - How tests validate your code

## Tips

1. Start with data loading and preprocessing
2. Implement regression task first (simpler)
3. Use GridSearchCV from scikit-learn
4. Test individual functions with `pytest` as you implement them
5. Visualizations are important - make sure plots are clear and labeled
6. Read sklearn documentation for ROC/PR curve generation

## Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Data Preparation | 10 | Correct data loading, preprocessing, and feature engineering |
| ElasticNet Regression | 30 | Grid search, heatmap, best model selection |
| Logistic Regression | 30 | Grid search, AUROC/AUPRC curves, best parameters |
| k-NN Classification | 30 | Grid search, AUROC/AUPRC curves, best k |
| Code Quality | 10 | Comments, organization, clarity |
| **Total** | **100** | |

## Resources

- [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ROC Curves](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
- [Precision-Recall Curves](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

## Questions?

If you have questions about the assignment, please open an issue on the repository or contact the instructor.
