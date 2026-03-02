# ==========================================
# IMPORT LIBRARIES
# ==========================================

import csv
import numpy as np
from datetime import datetime
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# ==========================================
# DATA STORAGE
# ==========================================

dates = []   # Will store numeric timestamps
prices = []  # Will store stock prices (floats)


# ==========================================
# READ DATA FROM CSV
# ==========================================

def get_data(file):
    print("Loading file...")

    try:
        with open(file, 'r') as csvfile:
            csvFileReader = csv.reader(csvfile)
            next(csvFileReader)

            for row in csvFileReader:
                try:
                    print("Row:", row)

                    dt = datetime.strptime(row[0], "%m/%d/%Y %H:%M:%S")
                    dates.append(dt.timestamp())
                    prices.append(float(row[1]))

                except Exception as row_error:
                    print("Row Skipped — Error:", row_error)

        print("File loaded successfully")

    except Exception as e:
        print("File error:", e)


# ==========================================
# TRAIN MODEL + PREDICT
# ==========================================
def predict_prices(dates, prices, x):

    print("DEBUG: Starting prediction function")

    from sklearn.preprocessing import MinMaxScaler
    dates = np.array(dates).reshape(-1, 1)
    prices = np.array(prices)

    # Normalize timestamps to range 0 → 1
    scaler = MinMaxScaler()
    dates = scaler.fit_transform(dates)
    
    print("DEBUG: Creating models")

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    print("DEBUG: Training models")

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    print("DEBUG: Models trained")

    plt.scatter(dates, prices, color='black')

    plt.plot(dates, svr_rbf.predict(dates), color='red')
    plt.plot(dates, svr_lin.predict(dates), color='green')
    plt.plot(dates, svr_poly.predict(dates), color='blue')

    print("DEBUG: Showing plot")

    plt.show()

    print("DEBUG: Plot closed")

    x = np.array([[x]])

    return (
        svr_rbf.predict(x)[0],
        svr_lin.predict(x)[0],
        svr_poly.predict(x)[0]
    )

# ==========================================
# RUN PROGRAM
# ==========================================

# Load dataset
get_data("Apple Stock Prices 2024-2025 - Sheet1.csv")

# Example: Predict for a specific date

# If you want to predict a date manually:
future_date = datetime.strptime("1/30/2025 16:00:00",
                                "%m/%d/%Y %H:%M:%S")

future_timestamp = future_date.timestamp()

predicted_price = predict_prices(dates, prices, future_timestamp)

print("Predicted Prices (RBF, Linear, Polynomial):")
print(predicted_price)
print("Calling prediction function...")
predicted_price = predict_prices(dates, prices, future_timestamp)

print("Prediction Finished")
print("Predicted Prices:", predicted_price)