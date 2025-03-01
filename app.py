import os
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from cryptography.fernet import Fernet

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Load AI Model
model_path = "./ai_crypt_model (2).h5"  # Ensure the model is uploaded here
model = tf.keras.models.load_model(model_path, compile=False)

# Load Fernet Key from Environment Variable
KEY = os.getenv("FERNET_KEY")
if not KEY:
    raise ValueError("FERNET_KEY is not set in environment variables")
cipher_suite = Fernet(KEY.encode())

# Function to resample audio if needed
def resample_audio(audio_data, sr, target_sr=44100):
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
    return audio_data, target_sr

@app.route("/")
def home():
    return "AI-Crypt Flask API is running on Render!"

@app.route("/encrypt", methods=["POST"])
def encrypt_audio():
    if "audio" not in request.files or "noise" not in request.files:
        return jsonify({"error": "Both audio and noise files are required"}), 400

    audio_file = request.files["audio"]
    noise_file = request.files["noise"]

    # Read the files
    audio_data, sr1 = sf.read(audio_file)
    noise_data, sr2 = sf.read(noise_file)

    audio_data, sr1 = resample_audio(audio_data, sr1)
    noise_data, sr2 = resample_audio(noise_data, sr2)


    if sr1 != sr2:
        return jsonify({"error": "Sample rates of audio and noise must match"}), 400

    if len(noise_data) < len(audio_data):
    # Trim the audio to fit inside the noise length
      audio_data = audio_data[: len(noise_data)] 


      encrypted_audio = (noise_data[: len(audio_data)] + audio_data).astype(np.float32)
      encrypted_data = cipher_suite.encrypt(encrypted_audio.tobytes())


    return jsonify({"encrypted_audio": encrypted_data.hex(), "key": KEY.decode()}), 200

@app.route("/decrypt", methods=["POST"])
def decrypt_audio():
    try:
        data = request.json
        encrypted_audio = bytes.fromhex(data["encrypted_audio"])
        decryption_key = data["key"].encode()

        cipher = Fernet(decryption_key)
        decrypted_data = cipher.decrypt(encrypted_audio)

        # Convert bytes back to numpy array
        audio_array = np.frombuffer(decrypted_data, dtype=np.float32)
        audio_array = np.expand_dims(audio_array, axis=0)

        # AI Model Prediction (Modify based on your model’s purpose)
        prediction = model.predict(audio_array)

        return jsonify({"prediction": prediction.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Use Render's dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
