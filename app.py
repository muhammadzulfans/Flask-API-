from flask import Flask, request, jsonify
import joblib
import os
import traceback

app = Flask(__name__)

# --- LOAD MASTER MODEL DARI FILE .PKL ---
MODEL_PATH = "master_model_arima.pkl"

if os.path.exists(MODEL_PATH):
    print("Memuat master model ARIMA ke memori server...")
    master_models = joblib.load(MODEL_PATH)
    print(f"Berhasil memuat model untuk {len(master_models)} produk!")
    print("Contoh 5 key yang tersedia di model:", list(master_models.keys())[:5])
else:
    master_models = {}
    print("⚠️ Peringatan: File master_model_arima.pkl tidak ditemukan!")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Server Flask ARIMA Prediksi Penjualan Aktif!",
        "total_model_loaded": len(master_models)
    })


# --- ENDPOINT PREDIKSI PENJUALAN UNTUK BACKEND EXPRESS ---
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    cabang = data.get("cabang")
    kode_cat = str(data.get("kode_cat"))

    key_unik = f"{cabang}_{kode_cat}"

    if key_unik not in master_models:
        return jsonify({
            "error": f"Model untuk cabang {cabang} dengan kode {kode_cat} tidak ditemukan."
        }), 404

    try:
        model = master_models[key_unik]

        # Ramal untuk 1 periode ke depan (Prediksi Penjualan)
        prediksi_array = model.predict(n_periods=1)
        angka_prediksi = int(round(float(prediksi_array.iloc[0])))

        if angka_prediksi < 0:
            angka_prediksi = 0

        return jsonify({
            "status": "success",
            "cabang": cabang,
            "kode_cat": kode_cat,
            "prediksi_penjualan": angka_prediksi  # Diubah dari rekomendasi_stok
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)