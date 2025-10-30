# COVID-19 X-ray Classification using Transfer Learning

Deep learning project for classifying chest X-ray images to detect COVID-19 using VGG16 transfer learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 📋 Overview

This project classifies chest X-ray images into two categories:
- **COVID-19 positive**
- **Normal (Healthy)**

**Key Results:**
- Test Accuracy: **97%**
- F1 Score: **0.97**
- Training samples: 980 (after augmentation)

## 📊 Dataset

**Source**: [Kaggle - Chest X-Ray COVID-19 & Pneumonia](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

- Original: 121 images per class
- After augmentation: 700 images per class
- Split: 70% train / 10% validation / 20% test

### Data Augmentation
Applied to address limited dataset size:
- Zoom, rotation, brightness adjustments
- Horizontal flipping
- Gaussian distortion

## 🏗️ Model Architecture

**Transfer Learning with VGG16:**
- Base: VGG16 (pre-trained on ImageNet, frozen layers)
- Custom classifier:
  - GlobalAveragePooling2D
  - Dense (128, ReLU)
  - Dropout (0.5)
  - Dense (1, Sigmoid)

**Configuration:**
- Input: 224×224×3
- Optimizer: Adam
- Loss: Binary Crossentropy
- Class weights: Balanced

## 📈 Results

### Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.98 | 0.96 | 0.97 |
| COVID-19 | 0.96 | 0.98 | 0.97 |

**Overall Accuracy: 97%**

### Confusion Matrix
|  | Predicted Normal | Predicted COVID-19 |
|--|------------------|-------------------|
| **Normal** | 134 | 6 |
| **COVID-19** | 3 | 137 |

## 🚀 Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Running the Project

**In Google Colab:**
1. Open the notebook in Colab
2. Mount Google Drive
3. Upload dataset to Drive
4. Run all cells

**Making Predictions:**
```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('covid19_cnn.h5')

# Load and preprocess image
img = image.load_img('xray.png', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
print("COVID-19" if prediction[0] > 0.5 else "Normal")
```

## 📁 Project Structure

```
covid19-xray-classification/
├── COVID_19_Classification_CNN.ipynb  # Main notebook
├── README.md
├── requirements.txt
├── models/                            # Saved models
├── results/                           # Visualizations
└── docs/                              # Documentation
```

## 🎯 Project Requirements

This project fulfills the following assignment requirements:
- ✅ Module and layer imports
- ✅ Dataset loading and visualization
- ✅ Data preprocessing and augmentation
- ✅ CNN model creation (Transfer Learning)
- ✅ Optimizer configuration and compilation
- ✅ Model training
- ✅ Evaluation (Accuracy, Loss, F1 Score, Confusion Matrix)

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **VGG16** - Pre-trained model
- **Augmentor** - Data augmentation
- **Scikit-learn** - Evaluation metrics
- **Matplotlib/Seaborn** - Visualizations

## 📝 Notes

- Training time: ~2.3 hours (10 epochs)
- Best validation accuracy: 96.43% (epoch 10)
- Model files: `.h5` and `.keras` formats

## 🙏 Acknowledgments

- Dataset: [Kaggle COVID-19 X-ray Dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
- VGG16 Architecture: Simonyan & Zisserman, 2014
- Course instructor for project guidance

## 👤 Author

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

⚠️ **Disclaimer**: This model is for educational purposes only and should not be used for medical diagnosis.
