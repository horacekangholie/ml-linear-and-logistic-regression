# 📘 Linear & Logistic Regression
---

## 📑 Table of Contents
1. [Linear Regression](#linear-regression)
2. [Logistic Regression](#logistic-regression)
3. [Model Parameters, Attributes, and Methods](#model-parameters-attributes-and-methods)
4. [Conclusion](#conclusion)


## Linear Regression

Linear regression is a type of statistical model used to construct a linear relationship between one or more independent variables and a dependent variable.\
When there is only one independent variable, it is called **simple linear regression**.\
When there are multiple independent variables, it is called **multiple linear regression**.

### Key Concepts
- Used for modeling the relationship between a dependent variable ($y$) and one or more independent variables ($x$).
- Model formula:
  $$y = ax + b$$

  Where:
    $y$ is the dependent variable,

    $x$ is the independent variable,

    $a$ is the slope (representing the effect of changes in $x$ on $y$),

    $b$ is the intercept (the value of $y$ when $x$ is zero).

### Loss Functions

In linear regression, our goal is to find the most suitable parameters aa and bb for the model.
This involves the use of a loss function, which measures the difference between the predicted value and the actual observed value.\
This difference is called the **residual**.\
Our objective is to minimize this residual.\
Commonly used loss functions include **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**.

- **Mean Absolute Error (MAE)**:
MAE is the average of the absolute differences between the predicted values and the actual observed values.
For each sample ii, the MAE is calculated using the following formula:
  $$
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  $$
- **Mean Squared Error (MSE)**:
MSE is the average of the squared differences between the actual observed values and the predicted values.
Compared to MAE, MSE gives more weight to larger errors. The formula for calculating MSE is as follows:
  $$
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$

### Model Solving Methods
- **Closed-form** (Least Squares):
When the number of features is relatively small, the closed-form solution is generally more suitable.
  $$
  \theta = (X^T X)^{-1} X^T y
  $$
- **Gradient Descent**:
When the number of features is large, gradient descent becomes a more flexible and widely applicable solution.
  $$
  \theta_j := \theta_j - \eta \frac{\partial \text{Loss}}{\partial \theta_j}
  $$



## Logistic Regression

### Key Concepts
- Used for classification tasks.
- **Sigmoid function**:
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$
- **Model equation**:
  $$
  P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}
  $$
- **Multi-class support** via:
  - One-vs-Rest (OvR)
  - Many-vs-Many (MvM)

### Loss Function
- **Cross Entropy**:
  $$
  \text{Loss} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
  $$


## Model Parameters, Attributes, and Methods

| Category     | Name             | Description |
|--------------|------------------|-------------|
| Parameter    | `penalty`        | Regularization type (`l1`, `l2`) |
|              | `C`              | Inverse of regularization strength |
|              | `solver`         | Optimization algorithm |
|              | `multi_class`    | Classification strategy (`ovr`, `multinomial`) |
|              | `max_iter`       | Maximum number of iterations |
|              | `class_weight`   | Handling imbalanced classes |
|              | `random_state`   | Random seed for reproducibility |
| Attribute    | `coef_`          | Model coefficients |
|              | `intercept_`     | Model bias |
| Method       | `fit(X, y)`      | Train the model |
|              | `predict(X)`     | Predict class labels |
|              | `predict_proba(X)`| Predict class probabilities |
|              | `score(X, y)`    | Accuracy score |


## Conclusion

- **Linear Regression** is suitable for regression tasks and optimized via closed-form or gradient descent.
- **Logistic Regression** is effective for binary/multiclass classification using sigmoid and cross-entropy loss.
- The chapter emphasized model training, feature scaling, evaluation (MSE, accuracy, confusion matrix), and Python implementation.
