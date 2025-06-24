from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
y = np.array([100000, 200000, 400000, 550000, 800000])

# Model
model = LinearRegression()
model.fit(x, y)

# Prediksi untuk 25 pelanggan
x_pred = np.array([[25]])
y_pred = model.predict(x_pred)

# Output
print("Koefisien (a):", model.coef_[0])
print("Intersep (b):", model.intercept_)
print("Prediksi pengeluaran untuk 25 pelanggan:", y_pred[0])

# Plot data dan garis regresi
plt.scatter(x, y, color='blue', label='Data asli')
plt.plot(x, model.predict(x), color='red', label='Regresi linear')
plt.scatter(x_pred, y_pred, color='green', label='Prediksi (25 pelanggan)')
plt.title('Regresi Linear: pelanggan vs pengeluaran')
plt.xlabel('Jumlah pelanggan')
plt.ylabel('Pengeluaran (ribu rupiah)')
plt.legend()
plt.grid(True)
plt.show()
