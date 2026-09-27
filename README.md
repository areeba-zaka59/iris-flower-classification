# Iris Flower Classification

Multi-class classification of iris flowers into *setosa*, *versicolor*, and *virginica* using four physical measurements: sepal length, sepal width, petal length, and petal width.

This project goes beyond the typical single-model approach (load data, train KNN, report accuracy) by including proper exploratory analysis, a fair comparison of five models, PCA-based visualization, hyperparameter tuning, and a decision boundary plot to explain the final model's behavior.

## Dataset

The classic UCI Iris dataset — 150 samples, 4 numeric features, 3 balanced classes of 50 samples each, no missing values. Loaded via `sklearn.datasets.load_iris`, which provides the exact same values as the commonly used Kaggle "Iris Species" dataset.

## Approach

1. **Exploratory Data Analysis** — pairplots, a correlation heatmap, and boxplots to identify which features actually separate the species.
2. **PCA Visualization** — reduce four features to two principal components to visually confirm class separability.
3. **Model Comparison** — five models (KNN, Logistic Regression, SVM, Decision Tree, Random Forest) compared fairly using 5-fold stratified cross-validation.
4. **Hyperparameter Tuning** — `GridSearchCV` over the best-performing model (SVM).
5. **Final Evaluation** — confusion matrix, classification report, and a decision boundary plot on the two most informative features.

## Key Results

| Step | Result |
|---|---|
| EDA | No missing values, perfectly balanced classes; petal features separate species far better than sepal features |
| PCA | First 2 components explain **95.81%** of total variance |
| Best baseline model (5-fold CV) | SVM (RBF kernel) — **96.67%** mean accuracy |
| Tuned model (GridSearchCV) | Linear-kernel SVM, C=0.1 — **97.50%** CV accuracy |
| Final test accuracy | **93.33%** (28/30), errors only between versicolor/virginica — the one genuinely overlapping pair |

## Screenshots

### 1. Feature Pairplot
Pairwise relationships between all four features, colored by species. Petal measurements separate the species far more cleanly than sepal measurements.

![Pairplot](images/plot_01_pairplot.png)

### 2. Feature Correlation Heatmap
Petal length and petal width are extremely strongly correlated (0.96).

![Correlation Heatmap](images/plot_02_correlation_heatmap.png)

### 3. Feature Distributions by Species
Boxplots confirming petal length/width show almost no overlap between setosa and the other two species.

![Boxplots](images/plot_03_boxplots.png)

### 4. PCA Projection
Two principal components explain 95.81% of variance; setosa is fully isolated, versicolor/virginica overlap slightly.

![PCA](images/plot_04_pca.png)

### 5. Model Comparison
5-fold cross-validation accuracy across five different classifiers.

![Model Comparison](images/plot_05_model_comparison.png)

### 6. Confusion Matrix
Final tuned model on the held-out test set — 93.33% accuracy, errors isolated to the two overlapping species.

![Confusion Matrix](images/plot_06_confusion_matrix.png)

### 7. Decision Boundary
Linear SVM decision boundary using petal length and petal width, the two strongest features.

![Decision Boundary](images/plot_07_decision_boundary.png)

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (model training, cross-validation, GridSearchCV, PCA)
- matplotlib, seaborn (visualization)

## Files

- `Iris_Classification.ipynb` — full notebook with all code, outputs, and plots
- `Iris.csv` — the dataset used
- `images/` — all generated plots

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Iris_Classification.ipynb
```

Run all cells top to bottom — each step builds on variables created in the previous one.

## Author

**Areeba Zaka**
Machine Learning Intern
areebazaka59@gmail.com