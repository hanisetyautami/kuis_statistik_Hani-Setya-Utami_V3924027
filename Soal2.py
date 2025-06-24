from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([1, 2, 3, 4]).reshape(-1, 1)
Y = np.array([60, 65, 70, 75])

# Membuat model regresi linear
model = LinearRegression()
model.fit(X, Y)

# Prediksi nilai berdasarkan waktu belajar
Y_pred = model.predict(X)

# Prediksi nilai jika belajar 5 jam
prediksi_5_jam = model.predict([[5]])

# Output koefisien dan hasil prediksi
print("Rata-rata peningkatan nilai per jam belajar:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediksi nilai jika belajar 5 jam:", prediksi_5_jam[0])

# Visualisasi
plt.scatter(X, Y, color='blue', label='Data Asli')
plt.plot(X, Y_pred, color='red', label='Garis Regresi')
plt.scatter(5, prediksi_5_jam, color='green', label='Prediksi (5 jam)')
plt.title('Regresi Linear: Waktu Belajar vs Nilai Ujian')
plt.xlabel('Waktu Belajar (jam)')
plt.ylabel('Nilai Ujian')
plt.legend()
plt.grid(True)
plt.show()
