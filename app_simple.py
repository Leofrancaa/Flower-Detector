"""
Aplicação Flask SIMPLIFICADA para detecção de flores usando scikit-learn
Não requer TensorFlow! Funciona apenas com o que já está instalado.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import json
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Variáveis globais
model = None
class_names = ['setosa', 'versicolor', 'virginica']
img_size = 64  # Imagem menor para processar mais rápido

def train_simple_model():
    """
    Treina um modelo Random Forest simples usando dados do Iris
    """
    print("Treinando modelo simples...")

    # Carregar dataset Iris
    iris = load_iris()
    X, y = iris.data, iris.target

    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Salvar modelo
    os.makedirs('models', exist_ok=True)
    with open('models/simple_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Modelo treinado e salvo!")
    return model

def load_model_simple():
    """
    Carrega ou treina o modelo simples
    """
    global model

    model_path = 'models/simple_model.pkl'

    if os.path.exists(model_path):
        print("Carregando modelo...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Modelo carregado!")
    else:
        print("Modelo não encontrado. Treinando novo modelo...")
        model = train_simple_model()

    return True

def extract_features_from_image(image):
    """
    Extrai 4 features de uma imagem para simular o dataset Iris
    Features: média de cada canal RGB + brilho
    """
    # Redimensionar
    image = image.resize((img_size, img_size))

    # Converter para array numpy
    img_array = np.array(image)

    # Garantir RGB
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Extrair features (4 valores para coincidir com Iris dataset)
    # Simulando: sepal length, sepal width, petal length, petal width
    features = []

    # Feature 1: Média do canal vermelho (normalizado)
    red_mean = np.mean(img_array[:, :, 0]) / 255.0 * 7 + 4  # Escala similar ao Iris

    # Feature 2: Média do canal verde
    green_mean = np.mean(img_array[:, :, 1]) / 255.0 * 3 + 2

    # Feature 3: Média do canal azul
    blue_mean = np.mean(img_array[:, :, 2]) / 255.0 * 6 + 1

    # Feature 4: Desvio padrão do brilho
    brightness = np.mean(img_array, axis=2)
    brightness_std = np.std(brightness) / 255.0 * 2 + 0.1

    features = [red_mean, green_mean, blue_mean, brightness_std]

    return np.array(features).reshape(1, -1)

@app.route('/')
def index():
    """
    Página principal
    """
    return render_template('index.html', class_names=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para fazer predição
    """
    try:
        if model is None:
            return jsonify({
                'error': 'Modelo não carregado'
            }), 500

        # Obter imagem
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'Nenhuma imagem fornecida'}), 400

        # Decodificar imagem base64
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Abrir imagem
        image = Image.open(io.BytesIO(image_bytes))

        # Extrair features
        features = extract_features_from_image(image)

        # Fazer predição
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]

        predicted_class_idx = prediction[0]
        confidence = float(probabilities[predicted_class_idx])

        # Preparar resultado
        result = {
            'class': class_names[predicted_class_idx],
            'confidence': confidence,
            'all_predictions': {
                class_names[i]: float(probabilities[i])
                for i in range(len(class_names))
            },
            'features_extracted': features[0].tolist()
        }

        return jsonify(result)

    except Exception as e:
        print(f"Erro na predição: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check
    """
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': class_names,
        'model_type': 'RandomForest (scikit-learn)'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Detector de Flores SIMPLIFICADO - Sem TensorFlow")
    print("=" * 60)

    # Carregar/treinar modelo
    if not load_model_simple():
        print("\nErro ao carregar modelo!")
        exit(1)

    # Obter IP local
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = '127.0.0.1'

    print(f"\nServidor iniciado!")
    print(f"\nAcesse pelo celular:")
    print(f"   http://{local_ip}:5000")
    print(f"\nOu localmente:")
    print(f"   http://localhost:5000")
    print(f"\nNOTA: Este eh um modelo simplificado usando Random Forest")
    print(f"   As predicoes sao baseadas em cores da imagem, nao em deep learning.")
    print(f"\nPressione CTRL+C para parar\n")

    # Iniciar servidor
    app.run(host='0.0.0.0', port=5000, debug=False)
