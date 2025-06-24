from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data
X1 = np.array([50, 60, 70, 80, 90])
X2 = np.array([2, 3, 4, 5, 6])
Y = np.array([400, 480, 560, 640, 720])

# Gabungkan X1 dan X2 sebagai input fitur
X = np.column_stack((X1, X2))

# Model regresi
model = LinearRegression()
model.fit(X, Y)

# Koefisien dan Intersep
a = model.intercept_
b1, b2 = model.coef_

# Prediksi untuk X1 = 75 dan X2 = 4
x_pred = np.array([[75, 4]])
y_pred = model.predict(x_pred)

# Output hasil
print(f"Persamaan regresi: Y = {a:.2f} + {b1:.2f}X1 + {b2:.2f}X2")
print("Prediksi penjualan (X1=75, X2=4):", y_pred[0])

# Plot 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, Y, color='blue', label='Data Asli')

# Permukaan regresi
X1_surf, X2_surf = np.meshgrid(np.linspace(50, 90, 10), np.linspace(2, 6, 10))
Y_surf = a + b1 * X1_surf + b2 * X2_surf
ax.plot_surface(X1_surf, X2_surf, Y_surf, alpha=0.5, color='red')

# Titik prediksi
ax.scatter(75, 4, y_pred, color='green', s=100, label='Prediksi')

# Label dan judul
ax.set_xlabel("Pengunjung (X1)")
ax.set_ylabel("Menu Baru (X2)")
ax.set_zlabel("Penjualan (Y)")
plt.title("Regresi Linear Berganda: Penjualan")
ax.legend()

# Simpan grafik
plt.savefig("regresi_berganda_penjualan.png")

# Tampilkan
plt.show()
