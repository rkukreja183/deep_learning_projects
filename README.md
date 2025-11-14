# Deep Learning Projects Repository

This repository contains a collection of three notebooks, each tackling a different deep learning task: Emotion Classification from text, Automatic Speech Recognition, and Image Denoising & Super-Resolution.

## 🚀 Projects Overview

1.  [Emotion Classification (Text)](#1-emotion-classification)
2.  [Automatic Speech Recognition (Speech)](#2-automatic-speech-recognition)
3.  [Image Denoising & Super-Resolution (Vision)](#3-image-denoising--super-resolution)

---

### 1. Emotion Classification

* **Task:** Classify emotions from textual data.
* **Data:** A challenging, low-resource dataset featuring three linguistically diverse languages: Santhali, Kashmiri, and Manipuri.
* **Models & Techniques:**
    * Utilized the **Gemma-3-1B** model.
    * Employed **few-shot prompting** for training.
    * Fine-tuned the model using **LoRA (Low-Rank Adaptation)**.

### 2. Automatic Speech Recognition

* **Task:** Transcribe speech to text.
* **Data:** 23 hours of Uyghur speech.
* **Process:**
    * Performed exploratory data analysis (EDA) to identify and handle large audio samples and silences.
    * Trained the **Whisper** model for transcription.

### 3. Image Denoising & Super-Resolution

* **Task:** Denoise noisy images and perform 4x super-resolution to significantly enhance clarity and resolution.
* **Models:**
    * Fine-tuned **MPRNet** (Multi-Stage Progressive Image Restoration Network) for denoising.
    * Fine-tuned **ESRGAN** (Enhanced Super-Resolution Generative Adversarial Network) for 4x super-resolution.

---

## 🛠️ Technologies & Models

* **Core:** Python 3, Jupyter Notebooks
* **Libraries:** PyTorch, Hugging Face (`transformers`, `datasets`), Pandas, NumPy, Librosa (for audio), Matplotlib
* **Models:**
    * Gemma-3-1B
    * Whisper
    * MPRNet
    * ESRGAN
* **Techniques:** LoRA Finetuning, Few-Shot Prompting, Super-Resolution, Denoising, ASR

---

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/deep_learning_projects.git](https://github.com/YOUR_USERNAME/deep_learning_projects.git)
    cd deep_learning_projects
    ```
    *(Remember to replace `YOUR_USERNAME` with your actual GitHub username)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install torch transformers pandas numpy jupyter matplotlib librosa
    ```
    > **Note:** You may need to install other specific libraries for the image projects (like `opencv-python` or `scikit-image`) as prompted by the notebooks.

---

## 🏃 How to Run the Notebooks

1.  Ensure you have completed the [Setup & Installation](#️-setup--installation) steps and have activated your virtual environment.

2.  Start the Jupyter Notebook server from your terminal:
    ```bash
    jupyter notebook
    ```

3.  Once the server opens in your browser, click on one of the project notebooks to get started:
    * `emotion_classification.ipynb` (or the actual filename)
    * `automatic_speech_recognition.ipynb` (or the actual filename)
    * `image_denoising_super_resolution.ipynb` (or the actual filename)
