from flask import Flask
from flask_socketio import SocketIO, emit
from io import BytesIO
from pydub import AudioSegment
import whisper
import os

# Inicializar Flask y Socket.IO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Cargar el modelo de Whisper
model = whisper.load_model("small")  # Cambiar a "tiny" o "small" para mayor velocidad

# Ruta de prueba para verificar que el backend está corriendo
@app.route("/")
def index():
    return "Whisper Streaming Backend is running."

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """
    Maneja los fragmentos de audio enviados por el cliente.
    Convierte el audio de webm a wav y lo transcribe usando Whisper.
    """
    try:
        # Convertir los bytes recibidos (webm) en un archivo de audio
        audio = AudioSegment.from_file(BytesIO(data), format="webm")

        # Guardar temporalmente el archivo WAV
        temp_wav_path = "temp_chunk.wav"
        audio.export(temp_wav_path, format="wav")

        # Transcribir el archivo WAV con Whisper
        result = model.transcribe(temp_wav_path, fp16=False, language="es")

        # Emitir la transcripción al cliente
        emit("transcription", {"text": result["text"]}, broadcast=True)

        # Eliminar el archivo temporal
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    except Exception as e:
        print(f"Error al procesar el fragmento de audio: {e}")
        emit("error", {"message": str(e)})
if __name__ == "__main__":
    # Ejecutar el servidor Flask-SocketIO
    socketio.run(app, host="0.0.0.0", port=5000)
