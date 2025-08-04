# Predicting Car Prices using an Artificial Neural Network compared to Linear Regression

---

## Part 1: CarPricesANN – Artificial Neural Network

**Description:**  
`CarPricesANN` is an ANN built to predict car prices based on features such as brand, model, year, engine size, etc.

**Model Architecture:**
- Input layer: **48 features**
- Hidden Layer 1: **64 neurons**, activation: **ReLU**
- Hidden Layer 2: **32 neurons**, activation: **ReLU**
- Output Layer: **1 neuron** (predicted price)

**Preprocessing:**  
- Normalized input data to prevent large variations in Mean Squared Error (MSE).
- Without normalization, the MSE deviated by tens of thousands of dollars.

**Training Results:**  
- Epoch 25 / 100 → Average loss: **0.007929**
- Epoch 50 / 100 → Average loss: **0.000771**
- Epoch 75 / 100 → Average loss: **0.000397**
- Epoch 100 / 100 → Average loss: **0.000265**

**Performance:**
- **MSE (normalized):** `5.0985e-05`
- **MSE (unnormalized):** `493.96`  
*(Car prices range: \$2,000 – \$18,301 → model is very accurate.)*

---

## Part 2: CarPricesReg – Linear Regression (Baseline)

**Description:**  
`CarPricesReg` uses Scikit-learn's **Linear Regression** for the same task. The dataset was cleaned to remove outliers (e.g., very old or damaged cars).

**Feature Correlation with Price:**
Year 0.646377
Mileage -0.308777
Owner_Count 0.017468
Engine_Size 0.301200


**Performance:**
- **R² Score:** `0.7763`
- **MSE:** `1095.82`

---

## Comparison

| Model          | Normalized? | MSE      | R² Score  |
|---------------|------------|----------|-----------|
| **ANN**       | Yes        | 0.00005 | N/A       |
| **Linear Reg**| No         | 1095.82 | 0.7763    |

**Observation:**  
The ANN outperformed linear regression significantly, although at the cost of higher implementation complexity and training time.
