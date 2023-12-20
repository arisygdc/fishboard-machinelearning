from flask import Flask, jsonify, request
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

app = Flask(__name__)

data = [
    {"Tanggal": 1637514000, "Berat": 83, "Harga": 17500},
    {"Tanggal": 1637600400, "Berat": 830, "Harga": 16000},
    {"Tanggal": 1637773200, "Berat": 41.5, "Harga": 17500},
    {"Tanggal": 1638723600, "Berat": 124.5, "Harga": 16500},
    {"Tanggal": 1641315600, "Berat": 16.6, "Harga": 17500},
]

X = np.array([[row["Tanggal"], row["Berat"]] for row in data])
y = np.array([row["Harga"] for row in data])

weights = np.array([row["Berat"] for row in data]).reshape(-1, 1)

# scaler = MinMaxScaler()
# normalized_weights = scaler.fit_transform(weights)

# Replace the original "Berat" values in X with normalized weights
# X[:, 1] = normalized_weights[:, 0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLP Regressor
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), activation='identity', max_iter=500, random_state=42)

# Initialize SVR dengan kernel RBF
svr_rbf = SVR(kernel='rbf')

# Train the model NN
mlp_regressor.fit(X_train, y_train)

# Train model SVR
svr_rbf.fit(X_train, y_train)

@app.route("/", methods=["GET"])
def neural_network():
    # Predict using the trained model
    y_pred_nn = mlp_regressor.predict(X_test)
    y_pred_svr = svr_rbf.predict(X_test)

    # Evaluate the model
    r2_nn = r2_score(y_test, y_pred_nn)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)

    r2_svr = r2_score(y_test, y_pred_svr)
    mae_svr = mean_absolute_error(y_test, y_pred_svr)

    result_data = {
        "Evaluation Neural Network": {"r2_score": r2_nn, "mean_absolute_error": mae_nn},
        "Evaluation SVR": {"r2_score": r2_svr, "mean_absolute_error": mae_svr}
    }

    return jsonify(result_data)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil input dari pengguna
        data_json = request.get_json()
        tanggal = float(data_json["tanggal"])
        berat = float(data_json["berat"])

        # Buat array input untuk prediksi
        input_data = np.array([[tanggal, berat]])

        # Normalisasi berat
        # berat_normalized = scaler.transform(input_data[:, 1].reshape(-1, 1))
        # input_data[:, 1] = berat_normalized[:, 0]

        # Prediksi harga Neural Network
        harga_prediksi_NN = mlp_regressor.predict(input_data)

        harga_prediksi_SVR = svr_rbf.predict(input_data)

        # Format hasil sebagai JSON
        result_data = {
            "tanggal": tanggal,
            "berat": berat,
            "harga_prediksi Neural Network": float(harga_prediksi_NN[0]),
            "harga_prediksi SVR": float(harga_prediksi_SVR[0])
        }

        return jsonify(result_data)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
