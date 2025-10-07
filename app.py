"""
Aplicação Flask para detecção de flores em tempo real usando a câmera
Acesse pelo celular: http://SEU_IP:5000
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import json
import os

app = Flask(__name__)
CORS(app)

# Variáveis globais
model = None
class_names = None
img_size = None

def load_model_and_classes():
    """
    Carrega o modelo treinado e informações das classes
    """
    global model, class_names, img_size

    model_path = 'models/flower_classifier.h5'
    class_info_path = 'models/class_info.json'

    if not os.path.exists(model_path):
        print("⚠️ Modelo não encontrado!")
        print("Execute primeiro: python train_flower_model.py")
        return False

    try:
        # Carregar modelo
        print("Carregando modelo...")
        model = keras.models.load_model(model_path)
        print("✅ Modelo carregado com sucesso!")

        # Carregar informações das classes
        if os.path.exists(class_info_path):
            with open(class_info_path, 'r') as f:
                class_info = json.load(f)
                class_names = class_info['class_names']
                img_size = class_info['img_size']
        else:
            # Valores padrão
            class_names = ['setosa', 'versicolor', 'virginica']
            img_size = 224

        print(f"Classes: {class_names}")
        return True

    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

def preprocess_image(image):
    """
    Preprocessa a imagem para o modelo
    """
    # Redimensionar
    image = image.resize((img_size, img_size))

    # Converter para array numpy
    img_array = np.array(image)

    # Garantir que tem 3 canais (RGB)
    if img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Normalizar para [0, 1]
    img_array = img_array.astype(np.float32) / 255.0

    # Adicionar dimensão do batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.route('/')
def index():
    """
    Página principal com interface da câmera
    """
    return render_template('index.html', class_names=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para fazer predição da imagem
    """
    try:
        # Verificar se o modelo está carregado
        if model is None:
            return jsonify({
                'error': 'Modelo não carregado. Execute train_flower_model.py primeiro.'
            }), 500

        # Obter imagem do request
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'Nenhuma imagem fornecida'}), 400

        # Decodificar imagem base64
        image_data = data['image'].split(',')[1]  # Remover "data:image/jpeg;base64,"
        image_bytes = base64.b64decode(image_data)

        # Abrir imagem
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocessar
        processed_image = preprocess_image(image)

        # Fazer predição
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Preparar resultado
        result = {
            'class': class_names[predicted_class_idx],
            'confidence': confidence,
            'all_predictions': {
                class_names[i]: float(predictions[0][i])
                for i in range(len(class_names))
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de health check
    """
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': class_names
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🌸 Detector de Flores - Servidor Web")
    print("=" * 50)

    # Carregar modelo
    if not load_model_and_classes():
        print("\n⚠️ ATENÇÃO: Modelo não encontrado!")
        print("Execute primeiro: python train_flower_model.py\n")
        exit(1)

    # Obter IP local
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"\n✅ Servidor iniciado!")
    print(f"\n📱 Acesse pelo celular:")
    print(f"   http://{local_ip}:5000")
    print(f"\n💻 Ou localmente:")
    print(f"   http://localhost:5000")
    print(f"\nPressione CTRL+C para parar o servidor\n")

    # Iniciar servidor
    app.run(host='0.0.0.0', port=5000, debug=False)
