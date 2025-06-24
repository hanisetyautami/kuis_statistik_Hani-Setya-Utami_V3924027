from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([5, 10, 15, 20]).reshape(-1, 1)
Y = np.array([25000, 50000, 75000, 100000])

# Model regresi
model = LinearRegression()
model.fit(X, Y)

# Prediksi nilai Y
Y_pred = model.predict(X)

# Prediksi untuk 18 jam
prediksi_18 = model.predict([[18]])

# Output
print("Persamaan regresi: Y = {:.0f} + {:.0f}X".format(model.intercept_, model.coef_[0]))
print("Prediksi biaya pulsa untuk 18 jam:", prediksi_18[0])

# Plot
plt.scatter(X, Y, color='blue', label='Data Asli')
plt.plot(X, Y_pred, color='red', label='Garis Regresi')
plt.scatter(18, prediksi_18, color='green', label='Prediksi (18 jam)')
plt.title('Jam Internet vs Biaya Pulsa')
plt.xlabel('Jam Internet')
plt.ylabel('Biaya Pulsa (Rp)')
plt.legend()
plt.grid(True)
plt.show()
