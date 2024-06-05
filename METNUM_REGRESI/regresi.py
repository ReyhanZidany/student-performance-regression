import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Load dataset
file_path = r'c:\Users\ACER\Documents\archive\Student_Performance.csv'  # Gunakan raw string
data = pd.read_csv(file_path)

# Extract relevant columns
TB = data['Hours Studied'].values.reshape(-1, 1)
NL = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
NT = data['Performance Index'].values

# Problem 1: Durasi waktu belajar (TB) terhadap nilai ujian (NT)

# Metode 1: Model Linear
model_linear_TB = LinearRegression()
model_linear_TB.fit(TB, NT)
NT_pred_linear_TB = model_linear_TB.predict(TB)
rms_error_linear_TB = math.sqrt(mean_squared_error(NT, NT_pred_linear_TB))

# Metode 2: Model Pangkat Sederhana (transformasi logaritmik)
TB_log = np.log(TB)
model_power_TB = LinearRegression()
model_power_TB.fit(TB_log, NT)
NT_pred_power_TB = model_power_TB.predict(TB_log)
rms_error_power_TB = math.sqrt(mean_squared_error(NT, NT_pred_power_TB))

# Problem 2: Jumlah latihan soal (NL) terhadap nilai ujian (NT)

# Metode 1: Model Linear
model_linear_NL = LinearRegression()
model_linear_NL.fit(NL, NT)
NT_pred_linear_NL = model_linear_NL.predict(NL)
rms_error_linear_NL = math.sqrt(mean_squared_error(NT, NT_pred_linear_NL))

# Metode 2: Model Pangkat Sederhana (transformasi logaritmik)
NL_log = np.log(NL + 1e-9)  # Menambahkan nilai kecil untuk menghindari log(0)
model_power_NL = LinearRegression()
model_power_NL.fit(NL_log, NT)
NT_pred_power_NL = model_power_NL.predict(NL_log)
rms_error_power_NL = math.sqrt(mean_squared_error(NT, NT_pred_power_NL))

# Plotting
plt.figure(figsize=(14, 12))

# Plot for Model Linear (TB vs NT)
plt.subplot(2, 2, 1)
plt.scatter(TB, NT, color='blue', label='Data')
plt.plot(TB, NT_pred_linear_TB, color='red', label='Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Model Linear (Hours Studied vs Performance Index)')
plt.legend()

# Plot for Model Pangkat Sederhana (TB vs NT)
plt.subplot(2, 2, 2)
plt.scatter(TB, NT, color='blue', label='Data')
plt.plot(TB, NT_pred_power_TB, color='green', label='Power Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Model Pangkat Sederhana (Hours Studied vs Performance Index)')
plt.legend()

# Plot for Model Linear (NL vs NT)
plt.subplot(2, 2, 3)
plt.scatter(NL, NT, color='blue', label='Data')
plt.plot(NL, NT_pred_linear_NL, color='red', label='Linear Regression')
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')
plt.title('Model Linear (Sample Question Papers Practiced vs Performance Index)')
plt.legend()

# Plot for Model Pangkat Sederhana (NL vs NT)
plt.subplot(2, 2, 4)
plt.scatter(NL, NT, color='blue', label='Data')
plt.plot(NL, NT_pred_power_NL, color='green', label='Power Regression')
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')
plt.title('Model Pangkat Sederhana (Sample Question Papers Practiced vs Performance Index)')
plt.legend()

plt.tight_layout()
plt.show()

# Print RMS errors
print(f'RMS Error for Model Linear (Hours Studied): {rms_error_linear_TB}')
print(f'RMS Error for Model Pangkat Sederhana (Hours Studied): {rms_error_power_TB}')
print(f'RMS Error for Model Linear (Sample Question Papers Practiced): {rms_error_linear_NL}')
print(f'RMS Error for Model Pangkat Sederhana (Sample Question Papers Practiced): {rms_error_power_NL}')
