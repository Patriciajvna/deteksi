from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# path model dengan model yang telah dilatih sebelumnya
model = load_model('static/trained_model2.h5')

# Fungsi untuk memprediksi gambar baru
def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(100, 100))  # Ubah ukuran gambar sesuai kebutuhan model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalisasi gambar sesuai praproses yang digunakan pada pelatihan model

    prediction = model.predict(img)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Simpan gambar sementara untuk memprediksi
            temp_image_path = 'static/temp.jpg'
            file.save(temp_image_path)

            # Lakukan prediksi menggunakan model h5
            prediction_result = predict_image(temp_image_path, model)
            prediction_result = "Tidak Rusak" if prediction_result[0][0] < 0.5 else "Rusak"

            # Hapus gambar sementara setelah diprediksi
            os.remove(temp_image_path)

            # Kirim hasil prediksi ke klien
            return jsonify({'result': prediction_result})
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)