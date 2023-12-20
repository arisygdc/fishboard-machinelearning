import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# activation = 'identity', 'logistic', 'tanh', 'relu'
def train(x, y, activation):
    # Bagi data menjadi data pelatihan dan data tes
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Inisialisasi model regresi dengan sigmoid sebagai fungsi aktivasi
    regressor = MLPRegressor(hidden_layer_sizes=(10, 10), activation=activation, solver='lbfgs', random_state=42, max_iter=1000)

    # Latih model regresi
    regressor.fit(X_train, y_train)

    # Lakukan prediksi pada data tes
    y_pred = regressor.predict(X_test)

    # Hitung MSE (Mean Squared Error) untuk mengukur performa model
    mse = mean_squared_error(y_test, y_pred)

    print("Hasil Prediksi pada Data Tes:")
    for i in range(len(y_test)):
        print(f"Data Tes-{i}: Prediksi={y_pred[i]:.6f}, Harga Sebenarnya={y_test[i]:.6f}")

    print(f"Mean Squared Error: {mse:.6f}")


    # Menghitung R-squared (R^2)
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2:.6f}")

    # Menghitung Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.6f}")

    # Ubah R-squared menjadi persentase
    accuracy_percentage = r2 * 100

    print(f"Akurasi dalam Persen: {accuracy_percentage:.2f}%")