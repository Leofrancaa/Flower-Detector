# 🌸 Detector de Flores com Deep Learning

Projeto de Machine Learning para detectar flores usando a câmera do celular! Construído com TensorFlow, Flask e uma interface web moderna.

## 📋 O que este projeto faz?

- ✅ Treina um modelo de Deep Learning (CNN) para classificar flores
- ✅ Cria uma aplicação web que acessa a câmera
- ✅ Detecta flores em tempo real usando a câmera do celular
- ✅ Mostra a confiança da predição

## 🚀 Como usar

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

> **Nota:** A instalação do TensorFlow pode demorar alguns minutos.

### 2. Treinar o Modelo

```bash
python train_flower_model.py
```

Isso vai:
- Criar um modelo CNN de classificação
- Treinar o modelo (pode usar dataset sintético ou real)
- Salvar o modelo treinado em `models/flower_classifier.h5`
- Gerar gráficos de treinamento

**Tempo estimado:** 5-15 minutos dependendo do seu hardware.

### 3. Iniciar o Servidor Web

```bash
python app.py
```

O servidor vai mostrar algo assim:

```
==================================================
🌸 Detector de Flores - Servidor Web
==================================================

✅ Modelo carregado com sucesso!
Classes: ['setosa', 'versicolor', 'virginica']

✅ Servidor iniciado!

📱 Acesse pelo celular:
   http://192.168.1.100:5000

💻 Ou localmente:
   http://localhost:5000
```

### 4. Acessar pelo Celular

1. **Conecte o celular na mesma rede WiFi** que o computador
2. Abra o navegador do celular
3. Digite o endereço IP mostrado no terminal (ex: `http://192.168.1.100:5000`)
4. Clique em "Iniciar Câmera"
5. Permita o acesso à câmera
6. Aponte para uma flor e clique em "Detectar Flor"

## 📱 Interface

A aplicação tem uma interface moderna e responsiva:

- **Câmera ao vivo** - Visualização em tempo real
- **Botão de captura** - Clique para detectar
- **Resultado colorido** - Mostra a flor detectada com confiança
- **Todas probabilidades** - Vê as chances de cada classe

## 🧠 Arquitetura do Modelo

O projeto oferece duas opções:

### Opção 1: CNN Simples (Dataset Sintético)
- 3 blocos convolucionais
- Batch Normalization
- Dropout para regularização
- ~150K parâmetros

### Opção 2: Transfer Learning (Dataset Real)
- MobileNetV2 pré-treinado
- Fine-tuning nas camadas finais
- Mais preciso para flores reais
- ~2.2M parâmetros

## 📊 Classes de Flores

Por padrão, o modelo classifica 3 tipos de flores:

1. **Setosa** 🌺
2. **Versicolor** 🌸
3. **Virginica** 🌷

## 🛠️ Estrutura do Projeto

```
ML/
├── train_flower_model.py    # Script de treinamento
├── app.py                    # Servidor Flask
├── requirements.txt          # Dependências
├── iris_classification.ipynb # Notebook educacional
├── templates/
│   └── index.html           # Interface web
└── models/
    ├── flower_classifier.h5  # Modelo treinado
    ├── class_info.json       # Informações das classes
    └── training_history.png  # Gráficos de treinamento
```

## 🔧 Configurações Avançadas

### Ajustar Hiperparâmetros

Edite `train_flower_model.py`:

```python
IMG_SIZE = 224      # Tamanho da imagem
BATCH_SIZE = 32     # Tamanho do batch
EPOCHS = 20         # Número de épocas
```

### Usar Dataset Real

O script tenta baixar automaticamente o dataset `tf_flowers` do TensorFlow. Se falhar, usa dados sintéticos para demonstração.

Para forçar o uso de dataset real, instale:

```bash
pip install tensorflow-datasets
```

### Mudar Porta do Servidor

Edite `app.py`:

```python
app.run(host='0.0.0.0', port=5000, debug=False)
```

## 📝 Requisitos

- Python 3.8+
- TensorFlow 2.13+
- Flask 2.3+
- Câmera (celular ou webcam)
- Navegador moderno com suporte a `getUserMedia`

## 🌐 Compatibilidade

### Navegadores
- ✅ Chrome/Chromium (Android e iOS)
- ✅ Safari (iOS)
- ✅ Firefox (Android)
- ✅ Edge (mobile)

### Sistemas
- ✅ Windows
- ✅ Linux
- ✅ macOS

## 🐛 Troubleshooting

### Modelo não encontrado
```
⚠️ Modelo não encontrado!
Execute primeiro: python train_flower_model.py
```
**Solução:** Execute `python train_flower_model.py` antes de iniciar o servidor.

### Câmera não funciona
**Possíveis causas:**
- Navegador não tem permissão
- Conexão não é HTTPS (alguns navegadores exigem)
- Câmera em uso por outro app

**Solução:**
- Permita acesso à câmera nas configurações
- Use `http://localhost` localmente (não precisa HTTPS)

### Não consegue conectar pelo celular
**Solução:**
- Verifique se está na mesma rede WiFi
- Desative firewall temporariamente
- Use o IP correto mostrado no terminal

### Erro ao instalar TensorFlow
**Para Windows:**
```bash
pip install tensorflow --upgrade
```

**Para Mac M1/M2:**
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

## 🎓 Aprendizado

Este projeto é ótimo para aprender:

- ✅ Deep Learning com TensorFlow/Keras
- ✅ Redes Neurais Convolucionais (CNN)
- ✅ Transfer Learning
- ✅ Criação de APIs com Flask
- ✅ Interface web com HTML/CSS/JavaScript
- ✅ Acesso à câmera via WebRTC
- ✅ Deploy de modelos ML

## 🔄 Próximos Passos

1. **Coletar dados reais** - Tire fotos de flores e retreine
2. **Mais classes** - Adicione mais tipos de flores
3. **Data Augmentation** - Melhore a generalização
4. **Deploy na nuvem** - Use Heroku, Railway ou Render
5. **App mobile nativo** - Criar app Android/iOS

## 📚 Recursos

- [TensorFlow](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [WebRTC/getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)

## 🤝 Contribuindo

Sinta-se livre para:
- Reportar bugs
- Sugerir melhorias
- Adicionar features
- Compartilhar resultados

## 📄 Licença

Este projeto é livre para uso educacional e comercial.

---

**Desenvolvido com ❤️ para aprender Machine Learning na prática!**

🌸 Boa sorte detectando flores! 🌸
