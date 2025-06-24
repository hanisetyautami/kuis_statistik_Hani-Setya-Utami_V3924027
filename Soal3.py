from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# data
x = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([20,40,50,65,80])

model = LinearRegression()
model.fit(x,y)

# prediksi nilai
x_pred = np.array([[4]])
prediksi = model.predict(x_pred)

print(f"Persamaan regresi: Y = {round(model.intercept_,2)} + {round(model.coef_[0],2)}X")
print(f"Prediksi harga jual jika ukuran = 6 adalah: {round(prediksi[0],2)} juta rupiah")

plt.scatter(x,y, color='blue', label='data asli')
plt.plot(x, model.predict(x), color='red', label='Regresi linear')
plt.scatter(x_pred, prediksi, color='green', label='Prediksi (4 Tahun)')
plt.xlabel('Ukuran rumah')
plt.ylabel('Harga jual (juta rupiah)')
plt.title('Regresi Linear')
plt.legend()
plt.grid(True)
plt.show()
