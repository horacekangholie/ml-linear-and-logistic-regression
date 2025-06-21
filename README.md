# 📘 Linear Regression

## 📌 Introduction

**Linear Regression** is a **supervised learning algorithm** used to model the relationship between a **dependent variable** and one or more **independent variables** by fitting a linear equation to observed data. It is one of the simplest and most widely used regression techniques in machine learning and statistics.

### 🔑 Key Concepts

-   **Model formula**:

    $y=ax+b$

-   **Loss functions**:

    -   **MAE (Mean Absolute Error)**

        $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
    
    -   **MSE (Mean Squared Error)**
        
        $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

-   **Model training methods**:

    -   **Closed-form solution** (Normal Equation):
        
        $\theta = (X^T X)^{-1} X^T y$
    
    -   **Gradient Descent**:
        
        $\theta_j := \theta_j - \eta \frac{\partial}{\partial \theta_j} \text{Loss}$

## ⚙️ What the Model Does

**Goal**: Model the relationship between a continuous target variable $y$ and one or more input features $x$.    

## 🚧 Limitations

| Limitation                             | Description |
|----------------------------------------|-------------|
| Assumes Linearity                      | Fails when the relationship is nonlinear. |
| Sensitive to Outliers                  | Outliers can significantly affect the fitted model. |
| Multicollinearity                      | Highly correlated features can distort predictions. |
| No Automatic Feature Selection         | All features contribute unless manually selected or regularized. |

---

## 🔧 Common Parameters, Attributes, and Methods

### Parameters (in `sklearn.linear_model.LinearRegression`)

| Parameter         | Description |
|------------------|-------------|
| `fit_intercept`   | Whether to calculate the intercept for the model. |
| `normalize`       | If `True`, the regressors are normalized before regression. (deprecated in latest versions) |
| `n_jobs`          | Number of jobs to use for computation. |

### Attributes (after fitting)

| Attribute          | Description |
|-------------------|-------------|
| `coef_`            | Estimated coefficients for the input features. |
| `intercept_`       | Intercept (bias) term of the model. |
| `rank_`            | Rank of the coefficient matrix. |
| `singular_`        | Singular values of the feature matrix. |

### Methods

| Method             | Description |
|-------------------|-------------|
| `fit(X, y)`        | Fits the model to the data. |
| `predict(X)`       | Predicts target values using the fitted model. |
| `score(X, y)`      | Returns the \( R^2 \) coefficient of determination of the prediction. |

---

## 📏 Evaluation Metrics

### 1. Mean Squared Error (MSE)

$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$

- Penalizes larger errors more than smaller ones.
- Lower is better.

### 2. Root Mean Squared Error (RMSE)

$\text{RMSE} = \sqrt{\text{MSE}}$

- Easier to interpret as it’s in the same unit as the target variable.

### 3. Mean Absolute Error (MAE)

$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$

- Measures average magnitude of errors.
- More robust to outliers than MSE.

### 4. R-squared ( $R^2$ )

$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

- Measures proportion of variance explained by the model.
- Closer to 1 means better fit.

---

## 🧪 Example - Predicting Excitation Current in Synchronous Machines

The **Synchronous Machine Dataset** from the UCI Machine Learning Repository contains measurements related to the operation of a synchronous machine (a type of AC electric motor).

### 🔍 Dataset Description:

-   **Instances**: 557 samples

-   **Features**:

    -   `I_y`: Load current

    -   `PF`: Power factor

    -   `e_PF`: Power factor error

    -   `d_If`: Change in excitation current

-   **Target**:

    -   `I_f`: Excitation current of the synchronous machine

### 🎯 Purpose:

The dataset aims to **model and predict the excitation current** (`I_f`) based on other measurable electrical parameters.\
This supports performance optimization and energy efficiency in industrial motor control.

[Demo code](/notebooks/linear_regression.ipynb)



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



# 📘 Logistic Regression

## 📌 Introduction

**Logistic Regression** is a **supervised learning algorithm** used for **binary classification problems**. It models the **probability** that a given input belongs to a particular class using the **logistic (sigmoid) function**. Despite its name, it is actually a classification algorithm, not a regression one.

### 🔑 Key Concepts

-   **Use Case**: Classify binary or multi-class categorical outcomes.

-   **Logistic (Sigmoid) Function**:
    
    $\sigma(z) = \frac{1}{1 + e^{-z}}$
    
    where, $\quad z = w_1x_1 + \dots + w_nx_n + b$
    

-   **Loss Function (Cross Entropy)**:
    
    $\text{Loss} = -\left[ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right]$

-   **Multiclass Strategy**:

    -   `OvR` (One-vs-Rest)

    -   `MvM` (Many-vs-Many)


## ⚙️ What the Model Does

Logistic Regression outputs a probability between 0 and 1. By applying a **threshold** (usually 0.5), it assigns a class label.

Variants:
- **Binary logistic regression**: 2 classes.
- **Multinomial logistic regression**: more than 2 classes (multiclass).
- **One-vs-rest strategy**: default approach in scikit-learn for multiclass.

## 🚧 Limitations

| Limitation                            | Description |
|---------------------------------------|-------------|
| Assumes Linear Decision Boundary      | Only effective if classes are linearly separable. |
| Not Robust to Multicollinearity       | Highly correlated features can destabilize predictions. |
| Sensitive to Outliers                 | Can affect the decision boundary. |
| Cannot Capture Complex Relationships  | Nonlinear boundaries require more advanced models or feature engineering. |

---

## 🔧 Common Parameters, Attributes, and Methods

### Parameters (in `sklearn.linear_model.LogisticRegression`)

| Parameter         | Description |
|------------------|-------------|
| `penalty`         | Regularization type (`'l2'`, `'l1'`, `'elasticnet'`, `'none'`). |
| `C`               | Inverse of regularization strength (smaller = stronger regularization). |
| `solver`          | Optimization algorithm (`'liblinear'`, `'saga'`, `'newton-cg'`, etc.). |
| `max_iter`        | Maximum iterations for solver convergence. |
| `multi_class`     | `'auto'`, `'ovr'` (one-vs-rest), or `'multinomial'`. |
| `random_state`    | For reproducibility. |

### Attributes (after fitting)

| Attribute          | Description |
|-------------------|-------------|
| `coef_`            | Coefficients for features. |
| `intercept_`       | Intercept term(s). |
| `classes_`         | Class labels. |
| `n_iter_`          | Number of iterations taken by the solver to converge. |

### Methods

| Method             | Description |
|-------------------|-------------|
| `fit(X, y)`        | Fit the model to the training data. |
| `predict(X)`       | Predict class labels. |
| `predict_proba(X)` | Predict class probabilities. |
| `score(X, y)`      | Return accuracy score on the given test data. |
| `decision_function(X)` | Distance of samples to decision boundary. |

---

## 📏 Evaluation Metrics

### 1. Accuracy

$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$

- Good baseline for balanced datasets.

### 2. Precision, Recall, F1 Score
| Metric     | Formula | Interpretation |
|------------|---------|----------------|
| Precision  | $\frac{TP}{TP + FP}$ | How many predicted positives were actually positive. |
| Recall     | $\frac{TP}{TP + FN}$ | How many actual positives were correctly predicted. |
| F1 Score   | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision and recall. |

### 3. ROC-AUC
- Measures model’s ability to distinguish between classes.
- AUC near 1 means excellent classifier.

---

## 🧪 Example - Iris Flower Classification

The **Iris flower dataset** is a classic dataset used for **multi-class classification**, making it ideal for demonstrating **logistic regression**.

### 🔍 Dataset Description:

-   **Samples**: 150

-   **Classes**: 3 species of Iris flowers:

    -   *Setosa*

    -   *Versicolor*

    -   *Virginica*

-   **Features** (all numerical):

    -   Sepal length

    -   Sepal width

    -   Petal length

    -   Petal width

### 🎯 Use in Logistic Regression:

The goal is to use the 4 features to **predict the species** of a flower.\
Since there are 3 classes, logistic regression applies a **multinomial (or one-vs-rest)** strategy to model and classify the flower species.

[Demo code](/notebooks/logistic_regression.ipynb)



