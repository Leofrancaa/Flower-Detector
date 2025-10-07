"""
Script para treinar um modelo CNN de classificação de flores usando o dataset Iris
e um dataset visual de flores (usaremos transfer learning com MobileNetV2)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import os
import json

# Configurações
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = 'models/flower_classifier.h5'
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

def create_model(num_classes=3):
    """
    Cria um modelo CNN usando transfer learning com MobileNetV2
    """
    # Carregar modelo pré-treinado (sem a camada de classificação)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Congelar as camadas do modelo base
    base_model.trainable = False

    # Criar o modelo completo
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def create_simple_cnn_model(num_classes=3):
    """
    Cria um modelo CNN simples para demonstração
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Bloco 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Bloco 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Bloco 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Camadas densas
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def download_flower_dataset():
    """
    Baixa um dataset de flores real usando TensorFlow Datasets
    """
    print("Baixando dataset de flores...")

    # Usar o dataset tf_flowers (5 classes de flores)
    # Para simplificar, vamos usar apenas 3 classes correspondentes ao Iris
    import tensorflow_datasets as tfds

    try:
        # Dataset de flores da Google
        dataset, info = tfds.load(
            'tf_flowers',
            with_info=True,
            as_supervised=True,
            split=['train[:80%]', 'train[80%:]']
        )

        train_dataset, val_dataset = dataset

        # Classes: daisy, dandelion, roses, sunflowers, tulips
        # Vamos mapear para 3 classes
        class_names = ['daisy', 'roses', 'tulips']

        return train_dataset, val_dataset, class_names

    except Exception as e:
        print(f"Erro ao baixar dataset: {e}")
        print("Criando dataset sintético para demonstração...")
        return None, None, CLASS_NAMES

def preprocess_image(image, label):
    """
    Preprocessa imagens para o modelo
    """
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Normalizar para [0, 1]
    return image, label

def create_synthetic_dataset():
    """
    Cria um dataset sintético para demonstração (quando não conseguir baixar real)
    """
    print("Criando dataset sintético para demonstração...")

    # Criar imagens sintéticas coloridas para cada classe
    samples_per_class = 100
    X_train = []
    y_train = []

    for class_id in range(3):
        for _ in range(samples_per_class):
            # Criar imagem com padrão de cor baseado na classe
            img = np.random.rand(IMG_SIZE, IMG_SIZE, 3)

            # Adicionar padrão específico por classe
            if class_id == 0:  # setosa - tons de vermelho
                img[:, :, 0] = img[:, :, 0] * 0.7 + 0.3
            elif class_id == 1:  # versicolor - tons de azul
                img[:, :, 2] = img[:, :, 2] * 0.7 + 0.3
            else:  # virginica - tons de verde
                img[:, :, 1] = img[:, :, 1] * 0.7 + 0.3

            X_train.append(img)
            y_train.append(class_id)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)

    # Split em treino e validação
    split_idx = int(0.8 * len(X_train))
    indices = np.random.permutation(len(X_train))

    X_val = X_train[indices[split_idx:]]
    y_val = y_train[indices[split_idx:]]
    X_train = X_train[indices[:split_idx]]
    y_train = y_train[indices[:split_idx]]

    return X_train, y_train, X_val, y_val

def train_model():
    """
    Treina o modelo de classificação de flores
    """
    print("=== Treinamento do Modelo de Classificação de Flores ===\n")

    # Criar diretório para salvar o modelo
    os.makedirs('models', exist_ok=True)

    # Tentar baixar dataset real
    train_ds, val_ds, class_names = download_flower_dataset()

    if train_ds is None:
        # Usar dataset sintético
        X_train, y_train, X_val, y_val = create_synthetic_dataset()
        class_names = CLASS_NAMES

        # Criar model simples
        print("Criando modelo CNN simples...")
        model = create_simple_cnn_model(num_classes=len(class_names))

    else:
        # Preprocessar datasets
        train_ds = train_ds.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Criar modelo com transfer learning
        print("Criando modelo com Transfer Learning (MobileNetV2)...")
        model = create_model(num_classes=len(class_names))

    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Resumo do modelo
    print("\n=== Arquitetura do Modelo ===")
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]

    # Treinar modelo
    print("\n=== Iniciando Treinamento ===")

    if train_ds is None:
        # Treinar com dados sintéticos
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Treinar com dataset real
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

    # Salvar modelo
    print(f"\nSalvando modelo em {MODEL_PATH}...")
    model.save(MODEL_PATH)

    # Salvar nomes das classes
    class_info = {
        'class_names': class_names,
        'img_size': IMG_SIZE
    }
    with open('models/class_info.json', 'w') as f:
        json.dump(class_info, f)

    # Plotar histórico de treinamento
    plot_training_history(history)

    print("\n✅ Treinamento concluído com sucesso!")
    print(f"Modelo salvo em: {MODEL_PATH}")

    return model, history

def plot_training_history(history):
    """
    Plota gráficos do histórico de treinamento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Acurácia
    ax1.plot(history.history['accuracy'], label='Treino')
    ax1.plot(history.history['val_accuracy'], label='Validação')
    ax1.set_title('Acurácia do Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Treino')
    ax2.plot(history.history['val_loss'], label='Validação')
    ax2.set_title('Loss do Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    print("\nGráfico salvo em: models/training_history.png")
    plt.close()

if __name__ == "__main__":
    # Configurar seed para reprodutibilidade
    np.random.seed(42)
    tf.random.set_seed(42)

    # Treinar modelo
    model, history = train_model()

    print("\n=== Próximos Passos ===")
    print("1. Execute 'python app.py' para iniciar o servidor web")
    print("2. Acesse pelo navegador do celular: http://SEU_IP:5000")
    print("3. Use a câmera para detectar flores!")
