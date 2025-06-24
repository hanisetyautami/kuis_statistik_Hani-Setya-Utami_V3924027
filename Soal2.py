from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([50, 60, 65, 80, 85, 90, 100])

model = LinearRegression()
model.fit(x, y)

# prediksi nilai berdasarkan ukuran balige
x_pred = np.array([[8]])
prediksi = model.predict(x_pred)

# Output
print("Koefisien (nilai prediksi per balige):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediksi nilai jika jumlah balige = 8:", prediksi[0])

# Visualisasi
plt.scatter(x, y, color='blue', label='data asli')
plt.plot(x, model.predict(x), color='red', label='regresi linear')
plt.scatter(x_pred, prediksi, color='green', label='prediksi (8 balige)')
plt.xlabel('Jumlah balige')
plt.ylabel('Nilai (skor)')
plt.title('Regresi Linear: jumlah balige vs nilai')
plt.legend()
plt.grid(True)
plt.show()
