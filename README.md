**QuantPredict**

 is a machine learning project that uses Support Vector Regression (SVR) to model and predict stock prices based on historical time-series data. The goal of this project is to compare different SVR kernels and observe how they perform on non-linear financial data.

**Overview**

The project loads historical stock price data from a CSV file and performs the following steps:

- Converts date strings into datetime objects

- Transforms datetime values into numerical timestamps

- Normalizes timestamps using MinMaxScaler

- Trains three SVR models (Linear, Polynomial, and RBF)

- Visualizes model predictions alongside historical prices

- The x-axis represents normalized time (0 to 1), and the y-axis represents stock price.

**Models Used**

Three Support Vector Regression models were trained:

- Linear Kernel: Captures the overall upward trend but underfits curvature and volatility.

- Polynomial Kernel: Improves on the linear model by modeling curvature but still struggles with sharper structural changes.

- RBF Kernel:Best adapts to non-linear patterns and captures acceleration in later time periods. This model provided the strongest overall fit.

**Key Observations**

The stock data shows:

- A long-term upward trend

- Short-term volatility

- Mid-period dips

- Accelerated growth near the end

Because the data is non-linear, kernel selection significantly affects performance. The RBF kernel performed best due to its flexibility.

**Importance of Feature Scaling**

Raw timestamps are extremely large (Unix timestamps in the billions). Without normalization, SVR training becomes unstable and slow.

Using MinMaxScaler:

- Improved model stability

- Prevented convergence issues

- Allowed successful model training

**Limitations**

- No train/test split

- No quantitative metrics (MSE or R²)

- No additional financial indicators

- Not intended as a production trading model

- This project focuses on demonstrating kernel behavior and machine learning workflow.

**How to Run**

1) Clone the repository:

 git clone https://github.com/vishipp/QuantPredict.git

cd QuantPredict

2) Install dependencies:

pip install numpy scikit-learn matplotlib

3) Run the program:

python3 main.py

A graph window will open showing model comparisons, and predicted prices will print in the terminal.

**Author**
Violet Shipp
Computer Science Student | Data Analysis & Machine Learning Enthusiast
