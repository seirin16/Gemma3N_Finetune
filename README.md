# Gemma-3N Fine-Tuning for Multimodal Phishing Detection

This repository contains a Google Colab notebook for fine-tuning the Gemma-3N model (specifically the `unsloth/gemma-3n-e4b-it` variant) for multimodal phishing website detection. The notebook demonstrates how to adapt a vision-language model (VLM) using parameter-efficient techniques like **LoRA (Low-Rank Adaptation)** via the **Unsloth** library. The model is trained on a balanced dataset of 5,000 samples (2,500 phishing and 2,500 legitimate websites), incorporating URLs, truncated HTML extracts, and screenshots as inputs.

The goal is to enhance the model's ability to classify websites as "Phishing" or "Legitimate" by integrating textual and visual cues, achieving significant improvements in recall and F1-score compared to zero-shot performance. This work is part of a research study on extending benchmarks for multimodal LLMs in cybersecurity.

---

## 📋 Table of Contents

* [Overview](#-overview)
* [Dataset](#-dataset)
* [Requirements](#-requirements)
* [Setup and Installation](#-setup-and-installation)
* [Usage](#-usage)
* [Training Details](#-training-details)
* [Model Saving and Deployment](#-model-saving-and-deployment)
* [Results](#-results)
* [License](#-license)
* [Citations](#-citations)
* [Contact](#-contact)

---

## 🚀 Overview

Gemma-3N is a lightweight (~4B parameters) multimodal LLM from Google, capable of processing text and images. This notebook fine-tunes it for phishing detection, where the model analyzes:

* **URL:** For lexical patterns like typosquatting.
* **HTML Extract:** Truncated to 5,000 characters for structural anomalies (e.g., suspicious scripts).
* **Screenshot:** For visual cues (e.g., fake logos or deceptive UI elements).

The fine-tuning uses **Supervised Fine-Tuning (SFT)** with a conversation-style format, where the model learns to respond directly with the classification label. This approach is efficient, running on free Colab hardware (e.g., T4 GPU) in about 4-6 hours.

## 📊 Dataset

* **Size:** 5,000 samples (balanced: 2,500 phishing, 2,500 legitimate).
* **Sources:**
    * *Phishing URLs:* From PhishTank (real-time community repository).
    * *Legitimate URLs:* From Majestic Million (top-ranking domains).
* **Preprocessing:**
    * HTML truncated to 5,000 characters.
    * Screenshots captured at 1280x720 resolution and resized to 512x512 for efficiency.
* **Format:** Each sample is converted to a multimodal conversation (user prompt with instruction + inputs, assistant response with label).
* **Public Availability:** The dataset is hosted on Hugging Face: [seirin16/Dataset_Phising](https://huggingface.co/datasets/seirin16/Dataset_Phising). **Note:** The repository name has a typo ("Phising").
* **For details** on data acquisition, see the notebook or the related GitHub repo: `phishing-detection-llm-benchmark`.

## 🛠️ Requirements

* Google Colab account (free tier with GPU access).
* **Libraries** (installed in the notebook):
    * `unsloth`: For efficient fine-tuning.
    * `transformers`, `datasets`, `peft`, `trl`: For model handling and training.
    * `bitsandbytes`: For quantization.
* **Hardware:** NVIDIA T4 or better GPU (available in Colab).

## ⚙️ Setup and Installation

1.  **Open the notebook in Google Colab:** `Gemma3N_Finetune.ipynb` (download from this repo and upload to Colab).
2.  **Mount Google Drive:** Run the cell to connect your Drive for saving models/datasets.
3.  **Install dependencies:** The notebook includes cells to install Unsloth and other libraries.
4.  **Load the dataset:** Use Hugging Face's `datasets` library to load the phishing dataset.

## ▶️ Usage

1.  **Prepare the Dataset:**
    * Load the CSV from Hugging Face or local path.
    * Apply the conversion function to format samples as conversations (user: instruction + URL + HTML + image; assistant: label).
2.  **Load the Model:**
    * Use Unsloth to load the quantized Gemma-3N instruct variant (4-bit for memory efficiency).
3.  **Fine-Tune:**
    * Run the trainer configuration cell.
    * Training takes ~4-6 hours on T4 GPU.
4.  **Evaluate:**
    * Test on a held-out set using the same metrics (precision, recall, F1, accuracy, normalized cost).
5.  **Save the Model:**
    * The notebook saves the LoRA adapters to Google Drive. Merge with the base model for full deployment.

For step-by-step execution, follow the notebook cells in order.

## 📈 Training Details

* **Model Variant:** `unsloth/gemma-3n-e4b-it` (instruct-tuned, ~4B parameters, multimodal with text + image input).
* **Fine-Tuning Method:** LoRA via Unsloth for parameter efficiency.
* **Input Format:** Multimodal conversations with direct classification instructions to focus on binary output.
* **Hyperparameters:**
    * **Epochs:** 2
    * **Learning Rate:** 2e-4 (cosine scheduler)
    * **Batch Size:** Effective 4 (per-device 1 + 4 accumulation steps)
    * **Optimizer:** Fused AdamW
    * **Warmup Ratio:** 0.03
    * **Max Sequence Length:** 8192
    * **Gradient Checkpointing:** Enabled for memory savings
    * **Seed:** 3407 for reproducibility
* **Monitoring:** Logged every step; validation loss to avoid overfitting.
* **Hardware:** Runs on Colab's free T4 GPU (~16GB VRAM).

## 💾 Model Saving and Deployment

* **Saved Artifacts:** LoRA adapters and processor saved to `/content/drive/MyDrive/Colab Notebooks/gemma3n-it-phishing-lora`.
* **Deployment:** Merge LoRA with base model using Unsloth/Peft. Deploy via Hugging Face or local inference for phishing checks in browsers/extensions.

## 🏆 Results

Post-fine-tuning, the model shows significant gains:

* **Recall:** Up to 1.0
* **F1-Score:** Improvements of 30-170% across modalities
* **Normalized Cost:** Reduced by 70-97%

For full benchmarks, see the related paper: "Extending the Benchmark: A Performance and Fine-Tuning Study of the Gemma 3N Large Language Model".

## 📄 License

This project is licensed under the **MIT License**. The dataset is under **CC-BY-NC-4.0**.

## 📚 Citations

If you use this code or model, please cite:

```bibtex
@article{placeholder,
  title={Extending the Benchmark: A Performance and Fine-Tuning Study of the Gemma 3N Large Language Model},
  author={Álvaro López Fueyo},
  year={2024},
  journal={Journal or Conference Name}
}
