from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import io

app = Flask(__name__)

# Load AI Model
model = tf.keras.models.load_model("ai_crypt_model (2).h5", compile=False)

def preprocess_audio(file):
    audio, sr = librosa.load(file, sr=None)
    return audio, sr

def postprocess_audio(audio, sr):
    output = io.BytesIO()
    sf.write(output, audio, sr, format='WAV')
    output.seek(0)
    return output

@app.route("/")
def home():
    return "AI-Crypt Flask API is running!"

@app.route("/encrypt", methods=["POST"])
def encrypt():
    if "audio" not in request.files or "noise" not in request.files:
        return jsonify({"error": "Both audio and noise files are required"}), 400
    
    audio_file = request.files["audio"]
    noise_file = request.files["noise"]
    
    audio, sr = preprocess_audio(audio_file)
    noise, _ = preprocess_audio(noise_file)
    
    if len(audio) > len(noise):
        return jsonify({"error": "Noise file must be at least as long as the audio file"}), 400
    
    # Reshape input for model
    input_data = np.stack([audio[:len(noise)], noise], axis=0)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    # Encrypt using model
    encrypted_audio = model.predict(input_data)[0]
    
    output_file = postprocess_audio(encrypted_audio, sr)
    return send_file(output_file, as_attachment=True, download_name="encrypted.wav", mimetype="audio/wav")

@app.route("/decrypt", methods=["POST"])
def decrypt():
    if "file" not in request.files:
        return jsonify({"error": "Encrypted file required"}), 400
    
    encrypted_file = request.files["file"]
    encrypted_audio, sr = preprocess_audio(encrypted_file)
    
    # Reshape input for model
    input_data = np.expand_dims(encrypted_audio, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)  # Add channel dimension
    
    # Decrypt using model
    decrypted_audio = model.predict(input_data)[0]
    
    output_file = postprocess_audio(decrypted_audio, sr)
    return send_file(output_file, as_attachment=True, download_name="decrypted.wav", mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
