# 📘 Linear & Logistic Regression

This chapter introduces two key supervised learning models: **Linear Regression** and **Logistic Regression**, covering both theoretical foundations and practical implementation using Python.

---

## 📑 Table of Contents
1. [Linear Regression](#linear-regression)
    - [Key Concepts](#key-concepts)
    - [Loss Functions](#loss-functions)
    - [Model Solving Methods](#model-solving-methods)
    - [Demo: Predicting Motor Current](#demo-predicting-motor-current)
2. [Logistic Regression](#logistic-regression)
    - [Key Concepts](#key-concepts-1)
    - [Loss Function](#loss-function)
    - [Demo: Iris Flower Classification](#demo-iris-flower-classification)
3. [Model Parameters, Attributes, and Methods](#model-parameters-attributes-and-methods)
4. [Conclusion](#conclusion)

---

## Linear Regression

### Key Concepts
- Used for modeling the relationship between a dependent variable \( y \) and one or more independent variables \( x \).
- **Model formula**:
  \[
  y = ax + b
  \]

### Loss Functions
- **Mean Absolute Error (MAE)**:
  $$
  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]
  $$
- **Mean Squared Error (MSE)**:
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]

### Model Solving Methods
- **Closed-form** (Least Squares):
  \[
  \theta = (X^T X)^{-1} X^T y
  \]
- **Gradient Descent**:
  \[
  \theta_j := \theta_j - \eta \frac{\partial \text{Loss}}{\partial \theta_j}
  \]

### Demo: Predicting Motor Current
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.read_csv("auto_mpg.csv")
X = df[['I_y', 'PF', 'e_PF', 'd_if']].values
y = df['I_f'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## Logistic Regression

### Key Concepts
- Used for classification tasks.
- **Sigmoid function**:
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
- **Model equation**:
  \[
  P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}
  \]
- **Multi-class support** via:
  - One-vs-Rest (OvR)
  - Many-vs-Many (MvM)

### Loss Function
- **Cross Entropy**:
  \[
  \text{Loss} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
  \]

### Demo: Iris Flower Classification
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(multi_class='auto', solver='liblinear', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))
```

---

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

---

## Conclusion

- **Linear Regression** is suitable for regression tasks and optimized via closed-form or gradient descent.
- **Logistic Regression** is effective for binary/multiclass classification using sigmoid and cross-entropy loss.
- The chapter emphasized model training, feature scaling, evaluation (MSE, accuracy, confusion matrix), and Python implementation.
