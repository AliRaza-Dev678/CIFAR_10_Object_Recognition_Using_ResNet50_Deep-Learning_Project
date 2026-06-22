# 🖼️ CIFAR-10 Object Recognition Using ResNet50 — Deep Learning

A deep learning project that performs **10-class image classification** on the CIFAR-10 dataset using **ResNet50** with transfer learning. The model leverages a pretrained ResNet50 backbone (trained on ImageNet) fine-tuned for recognizing objects like airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

---

## 🏷️ CIFAR-10 Classes

| Label | Class |
|---|---|
| 0 | Airplane ✈️ |
| 1 | Automobile 🚗 |
| 2 | Bird 🐦 |
| 3 | Cat 🐱 |
| 4 | Deer 🦌 |
| 5 | Dog 🐶 |
| 6 | Frog 🐸 |
| 7 | Horse 🐴 |
| 8 | Ship 🚢 |
| 9 | Truck 🚛 |

---

## 🧠 Model Architecture

```
CIFAR-10 Image (32×32×3)
        │
        ▼
Resize to (224×224×3)          ← ResNet50 expects larger input
        │
        ▼
ResNet50 (pretrained ImageNet) ← Feature extraction backbone
  - 50 layers deep
  - Residual (skip) connections
  - Frozen base layers
        │
        ▼
Global Average Pooling
        │
        ▼
Dense + ReLU                   ← Custom classification head
        │
        ▼
Dropout                        ← Regularization
        │
        ▼
Dense (10 units) + Softmax     ← 10-class output
```

---

## 📁 Project Structure

```
CIFAR_10_Object_Recognition_Using_ResNet50/
│
├── CIFAR_10_Object_Recognition_Using_ResNet50.ipynb
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AliRaza-Dev678/CIFAR_10_Object_Recognition_Using_ResNet50_Deep-Learning_Project.git
cd CIFAR_10_Object_Recognition_Using_ResNet50_Deep-Learning_Project
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 3. Dataset

CIFAR-10 is loaded automatically via Keras — no manual download needed:

```python
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

### 4. Run the notebook

```bash
jupyter notebook CIFAR_10_Object_Recognition_Using_ResNet50.ipynb
```

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Pretrained Model | ResNet50 (ImageNet weights) |
| Dataset | CIFAR-10 (built into Keras) |
| Data Handling | NumPy |
| Visualization | Matplotlib |
| Evaluation | Scikit-learn |
| Environment | Jupyter Notebook |

---

## 📊 Features

- **Transfer learning** — ResNet50 pretrained on ImageNet, fine-tuned for CIFAR-10
- **Residual connections** — skip connections prevent vanishing gradients in deep networks
- **Image upsampling** — CIFAR-10 images resized from 32×32 to 224×224 to match ResNet50 input
- **Data normalization** — pixel values scaled to [0, 1]
- **Training & validation curves** — accuracy and loss plotted per epoch
- **Confusion matrix** — per-class performance visualization
- **Classification report** — precision, recall, and F1-score for all 10 classes

---

## 📈 Results

| Metric | Value |
|---|---|
| Training Accuracy | ~XX% |
| Validation Accuracy | ~XX% |
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Epochs | XX |

> Update the table above with your actual results after training.

---

## 🔍 Why ResNet50?

Standard CNNs struggle with very deep architectures due to vanishing gradients. ResNet50 solves this with **residual (skip) connections** that allow gradients to flow directly through shortcut paths, enabling training of much deeper and more accurate networks.

---

## 📋 Requirements

```
tensorflow
numpy
matplotlib
scikit-learn
jupyter
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [CIFAR-10 Dataset — Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ResNet Paper — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/api_docs)
