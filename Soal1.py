from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
Y = np.array([200, 400, 600, 800, 1000])

# Model
model = LinearRegression()
model.fit(X, Y)

# Prediksi untuk 35 pelanggan
x_pred = np.array([[35]])
y_pred = model.predict(x_pred)

# Output
print("Koefisien (b):", model.coef_[0])
print("Intercept (a):", model.intercept_)
print("Prediksi pendapatan untuk 35 pelanggan:", y_pred[0])

# Plot data dan garis regresi
plt.scatter(X, Y, color='blue', label='Data Asli')
plt.plot(X, model.predict(X), color='red', label='Regresi Linear')
plt.scatter(35, y_pred, color='green', label='Prediksi (35 pelanggan)')
plt.title('Regresi Linear: Pelanggan vs Pendapatan')
plt.xlabel('Jumlah Pelanggan')
plt.ylabel('Pendapatan (ribu rupiah)')
plt.legend()
plt.grid(True)
plt.show()
