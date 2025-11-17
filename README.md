# Combined Cycle Power Plant (CCPP) Output Prediction

## Project Goal
This project aimed to predict the Net Hourly Electrical Energy Output (PE) of a Combined Cycle Power Plant using its thermodynamic state variables. The primary objective was to evaluate the suitability of sequential deep learning models (LSTM/GRU) versus static deep learning models (DNN) for this specific type of energy modeling problem.

## Data and Features
The dataset consists of **hourly averages** collected over six years (2006-2011).

| Feature | Description | Type |
| :--- | :--- | :--- |
| **PE** (Target) | Net Hourly Electrical Energy Output (MW) | Numerical |
| **AT** | Ambient Temperature | Numerical |
| **V** | Exhaust Vacuum | Numerical |
| **AP** | Ambient Pressure | Numerical |
| **RH** | Relative Humidity | Numerical |

## Methodology and Findings

The project followed a standard deep learning workflow, with a critical pivot based on initial results:

### 1. Sequential Modeling Attempt (LSTM/GRU)
* **Preprocessing:** Data was chronologically split (80/20), scaled using `MinMaxScaler`, and transformed into 3D sequences (timesteps, features) using the **sliding window** technique. Cyclical time features (`hour_sin`, `hour_cos`) were engineered.
* **Result:** Both the LSTM and GRU models failed to converge effectively. The predictions **collapsed to the mean** of the test set, indicating a complete inability to capture the underlying patterns, despite tuning hyperparameters (units, layers, look-back, learning rate).

### 2. Successful Pivot to Static Regression (DNN)

The failure of the recurrent models suggested that the problem was **non-sequential**. The project successfully pivoted to a Deep Neural Network (DNN) Regression approach.

* **Preprocessing Change:** The sequence generation step was **removed**. The data was fed into the model as a 2D array of independent samples (features, not timesteps).
* **Model:** A three-layer Feed-Forward DNN was constructed to model the instantaneous, non-linear relationship between the thermodynamic inputs and $\text{PE}$.
* **Result:** The DNN model successfully converged, proving that the relationship between the CCPP's input variables and its power output is primarily a **static, non-linear regression problem**, not a sequential time series problem.

## Final Model Performance

The final metrics achieved by the Deep Neural Network (DNN) model on the test set were highly accurate, confirming its suitability for this type of thermodynamic modeling.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | **4.25 MW** | Average error magnitude in the original units. |
| **Mean Absolute Error (MAE)** | **3.34 MW** | Average absolute deviation from the actual value. |
| **R-squared ($\text{R}^2$)** | **0.9379** | The model explains approximately 93.8% of the variance in the power output. |

***

## Dependencies

The project relies on the following key Python libraries:

* `pandas` (Data manipulation)
* `numpy` (Numerical operations)
* `matplotlib` / `seaborn` (Visualization)
* `scikit-learn` (Scaling and metrics)
* `tensorflow` / `keras` (Deep learning model construction)

***

## How to Run the Notebook

1.  Ensure you have the CCPP dataset (e.g., `power_plant_data.csv`) available.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
    ```
3.  Open and run the `Notebook.ipynb` file in a Jupyter environment.
