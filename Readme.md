# Audio Deepfake Detection - Model Comparison

## 📌 Project Overview
This project aims to compare different deep learning models for detecting spoofed audio. The task is to classify audio files as either **real** or **spoofed**, using the **ASVspoof 2019 (Logical Access) dataset**. Throughout this project, we evaluate the performance of three models:

- **RawNet**
- **ResNet**
- **Wav2Vec 2.0**

By training all models on the same dataset and using a consistent preprocessing pipeline, we aim to fairly compare their effectiveness in detecting synthetic speech.

---

## 📂 Dataset - ASVspoof 2019 (Logical Access)
The **ASVspoof 2019 LA dataset** consists of real and AI-generated speech samples. The dataset is structured as follows:
- **Real Speech:** Genuine human voices
- **Spoofed Speech:** Speech generated using various text-to-speech (TTS) and voice conversion (VC) algorithms
- **Format:** `.flac` audio files
- **Metadata:** A CSV file containing filenames and corresponding labels (real/spoof)

We preprocess the dataset differently depending on the model, as detailed below.

---

## Data overview:
### Audio file features:
**MFCC (Mel-Frequency Cepstral Coefficients)** is a widely used feature extraction technique in audio signal processing, particularly for speech and audio classification tasks. It is designed to capture the short-term power spectrum of an audio signal in a way that is more aligned with human auditory perception. 

###  Audio example file - mel spectogram:
![img.png](images/img.png)

## 🔄 Data Preprocessing
Each model requires different preprocessing techniques to transform raw audio into a suitable input format.

### **1️⃣ RawNet** (End-to-End Deep Learning)
- Converts `.flac` files into **1D raw waveform vectors**.
- No need for handcrafted feature extraction.
- Directly inputs the waveform into the network.

### **2️⃣ ResNet** (CNN-Based)
- Extracts **MFCC (Mel Frequency Cepstral Coefficients)** as input features.
- Generates **Mel Spectrograms** to visualize audio patterns.
- Unlike RawNet, this model relies on spectral representations.

### **3️⃣ Wav2Vec 2.0** (Transformer-Based)
- Requires raw audio converted into a **1D vector**.
- Pretrained on a large corpus of speech data, making it useful for transfer learning.
- Uses a transformer-based approach to learn representations from waveforms.

#### **Example: Mel Spectrogram (Used for ResNet)**
![Mel Spectrogram Placeholder](INSERT_IMAGE_URL_HERE)

---

## 🎛️ Data Augmentation Techniques
To improve model robustness and generalization, we applied several **audio augmentation techniques**:

### **1️⃣ Time Stretching**
- Changes the speed of the audio without altering pitch.
- Helps models learn variations in speech tempo.

### **2️⃣ Pitch Shifting**
- Modifies the pitch of the audio while keeping duration constant.
- Useful for handling different voice tones and speaker variations.

### **3️⃣ Additive Noise**
- Injects Gaussian/white noise into the audio.
- Simulates real-world background noise and improves generalization.

### **4️⃣ Reverberation (Echo Effects)**
- Mimics different room environments by adding reverb effects.
- Helps the model generalize to recordings from different locations.

### **5️⃣ Equalization (EQ Variations)**
- Alters frequency response to simulate different microphone qualities.
- Useful for handling recordings from different sources.

### **6️⃣ Time Masking (SpecAugment)**
- Masks random sections of the audio waveform to force the model to learn context.
- Similar to dropout in neural networks, improving generalization.

### **7️⃣ Frequency Masking**
- Randomly removes certain frequency bands.
- Helps the model become invariant to minor frequency shifts in speech.

These augmentations were applied dynamically during training to increase dataset diversity and improve model robustness.



---[aug_LA_T_1000137.flac](asvspoof%2FLA%2FLA%2FASVspoof2019_LA_train%2Fflac%2Faug_LA_T_1000137.flac)

## 🏗️ Model Architectures
### **RawNet**
- End-to-end deep learning model for audio classification.
- Uses **ResBlocks** to learn directly from raw waveforms.
- Extracts feature representations automatically without explicit transformations.

### **ResNet (Convolutional Neural Network - CNN)**
- Uses convolutional layers to learn patterns in **MFCC features**.
- Typically used for image-like structured inputs, making it effective for spectrogram-based classification.

### **Wav2Vec 2.0 (Self-Supervised Learning - Transformer-Based)**
- A state-of-the-art transformer-based model for speech representation learning.
- Pretrained on large speech datasets, reducing the need for extensive labeled data.
- Works directly with **raw waveforms** like RawNet but leverages transformer layers.

---

## 🚀 Training Setup

### **RawNet:**

Model was trained only on last layers, due to its size and training power needed

- **Batch Size:** `128`
- **Learning Rate:** `10`
- **Optimizer:** `Adam`
- **Epochs:** `10`
- **Loss Function:** Cross-entropy loss

### **ResNet:**
- **Batch Size:** `128`
- **Learning Rate:** `10`
- **Optimizer:** `Adam`
- **Epochs:** `10`
- **Loss Function:** Cross-entropy loss

### **Wav2Vec:**
Model was trained only on last layers, due to its size and training power needed
- **Batch Size:** `128`
- **Learning Rate:** `10`
- **Optimizer:** `Adam`
- **Epochs:** `10`
- **Loss Function:** Cross-entropy loss

We used **INSERT_HARDWARE (e.g., NVIDIA A100 GPU, TPU, etc.)** to accelerate training.

---

## 📊 Model Evaluation & Results
After training, we evaluated the models on a held-out test set. Here are the key evaluation metrics:
- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **Equal Error Rate (EER)** (commonly used in speaker verification tasks)

| Model       | Accuracy | Precision | Recall | F1-Score | EER |
|------------|----------|-----------|--------|----------|----|
| RawNet     | XX%      | XX%       | XX%    | XX%      | XX% |
| ResNet     | 89.83%   | 89.83%      | 1.0    | 96.41%   | 0.5|
| Wav2Vec 2.0 | XX%      | XX%       | XX%    | XX%      | XX% |

### **Confusion Matrices**
_(To be inserted)_

### **Key Insights**
- **RawNet performed best/worst due to `INSERT_REASON`**.
- **ResNet showed strengths in `INSERT_OBSERVATION`**.
- **Wav2Vec 2.0 was effective in `INSERT_OBSERVATION`**.

---

## 🛠️ How to Use This Repository
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/your_project_name.git
cd your_project_name
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Training**
```bash
python train.py --model rawnet  # Train RawNet
python train.py --model resnet  # Train ResNet
python train.py --model wav2vec # Train Wav2Vec 2.0
```

### **4️⃣ Evaluate Models**
```bash
python evaluate.py --model rawnet  # Evaluate RawNet
python evaluate.py --model resnet  # Evaluate ResNet
python evaluate.py --model wav2vec # Evaluate Wav2Vec 2.0
```

### **5️⃣ Visualize Spectrograms**
```bash
python visualize_spectrogram.py --file path/to/audio.flac
```

---

## 📌 Future Work
- Test on a **different dataset** (e.g., Celeb-DF, ASVspoof 2021).
- Implement **data augmentation** (background noise, speed variations, etc.).
- Fine-tune **Wav2Vec 2.0** on this dataset for better results.

---

## ✨ Contributors
- **Your Name** - [GitHub Profile](https://github.com/YOUR_GITHUB_USERNAME)

Feel free to contribute by opening an issue or submitting a pull request!

