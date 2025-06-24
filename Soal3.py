from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([1, 3, 5, 7, 9]).reshape(-1, 1)
Y = np.array([150, 120, 90, 60, 30])

# Buat model regresi
model = LinearRegression()
model.fit(X, Y)

# Prediksi nilai Y
Y_pred = model.predict(X)

# Prediksi harga untuk usia 4 tahun
prediksi_4_tahun = model.predict([[4]])

# Output
print("Persamaan regresi: Y = {:.2f} + {:.2f}X".format(model.intercept_, model.coef_[0]))
print("Prediksi harga jual untuk usia 4 tahun:", prediksi_4_tahun[0], "juta rupiah")

# Plot
plt.scatter(X, Y, color='blue', label='Data Asli')
plt.plot(X, Y_pred, color='red', label='Garis Regresi')
plt.scatter(4, prediksi_4_tahun, color='green', label='Prediksi (4 tahun)')
plt.title('Usia Kendaraan vs Harga Jual')
plt.xlabel('Usia (tahun)')
plt.ylabel('Harga Jual (juta rupiah)')
plt.legend()
plt.grid(True)
plt.show()
