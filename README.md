# Gemma-3N Fine-Tuning for Multimodal Phishing Detection

This repository contains Google Colab notebooks for fine-tuning and evaluating the Gemma-3N model (specifically the `unsloth/gemma-3n-e4b-it` variant) for multimodal phishing website detection. The project demonstrates how to adapt a vision-language model (VLM) using parameter-efficient techniques like **LoRA (Low-Rank Adaptation)** via the **Unsloth** library.

The goal is to enhance the model's ability to classify websites as "Phishing" or "Legitimate" by integrating textual and visual cues, achieving significant improvements in recall and F1-score compared to zero-shot performance. This work is part of a research study on extending benchmarks for multimodal LLMs in cybersecurity.

---

## 📋 Table of Contents

* [Overview](#-overview)
* [Dataset](#-dataset)
* [Requirements](#-requirements)
* [Notebooks in this Repository](#-notebooks-in-this-repository)
* [Setup and Usage Workflow](#-setup-and-usage-workflow)
* [Training Details](#-training-details)
* [Model Saving and Deployment](#-model-saving-and-deployment)
* [Results](#-results)
* [License](#-license)
* [Citations](#-citations)
* [Contact](#-contact)

---

## 🚀 Overview

Gemma-3N is a lightweight (~4B parameters) multimodal LLM from Google, capable of processing text and images. This project fine-tunes it for phishing detection, where the model analyzes:

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
* **Libraries** (installed in the notebooks):
    * `unsloth`: For efficient fine-tuning.
    * `transformers`, `datasets`, `peft`, `trl`: For model handling and training.
    * `bitsandbytes`: For quantization.
* **Hardware:** NVIDIA T4 or better GPU (available in Colab).

## 📓 Notebooks in this Repository

This repository provides three separate notebooks for a clear workflow:

1.  **`Gemma3N.ipynb` (Baseline Evaluation)**
    * **Purpose:** Evaluates the "zero-shot" performance of the base `unsloth/gemma-3n-e4b-it` model *before* any fine-tuning. This is useful for establishing a performance baseline.
    * **Note:** This notebook *does not* perform any training.

2.  **`Gemma3N_Finetune.ipynb` (Training)**
    * **Purpose:** This is the main notebook for performing Supervised Fine-Tuning (SFT) using LoRA.
    * **Process:** It loads the dataset, prepares the model, runs the training process (approx. 4-6 hours on a T4 GPU), and saves the resulting LoRA adapters to Google Drive.

3.  **`Gemma3N_AfterFinetune.ipynb` (Post-Finetune Evaluation)**
    * **Purpose:** Loads the base model and merges it with the trained LoRA adapters saved from the previous step.
    * **Process:** It then runs the same evaluation dataset to demonstrate the significant performance improvement *after* fine-tuning.

## ⚙️ Setup and Usage Workflow

Follow these steps to reproduce the results.

### Step 1: (Optional) Run Baseline Evaluation

1.  Open `Gemma3N.ipynb` in Google Colab.
2.  Run all cells to evaluate the zero-shot performance of the base Gemma-3N model. This shows you how the model performs *without* any training on the phishing dataset.

### Step 2: Run Fine-Tuning

1.  Open `Gemma3N_Finetune.ipynb` in Google Colab.
2.  **Mount Google Drive:** Run the cell to connect your Drive. This is **required** to save the trained model adapters.
3.  **Install Dependencies:** Run the installation cells for Unsloth and other libraries.
4.  **Load Dataset:** Load the `seirin16/Dataset_Phising` dataset from Hugging Face.
5.  **Run Training:** Execute the `SFTTrainer` cell. This process will take approximately 4-6 hours.
6.  **Verify:** At the end, the LoRA adapters will be saved to your Google Drive (e.g., in `/content/drive/MyDrive/Colab Notebooks/gemma3n-it-phishing-lora`).

### Step 3: Run Post-Training Evaluation

1.  Open `Gemma3N_AfterFinetune.ipynb` in Google Colab.
2.  **Mount Google Drive:** Connect to the same Google Drive account used in Step 2.
3.  **Install Dependencies:** Run the installation cells.
4.  **Load Model:** The notebook will load the base Gemma-3N model and automatically merge the LoRA adapters from the Google Drive path you specified.
5.  **Evaluate:** Run the evaluation cells to see the final performance of your fine-tuned model. You can compare these results directly with the baseline from Step 1.

## 📈 Training Details

* **Model Variant:** `unsloth/gemma-3n-e4b-it` (instruct-tuned, ~4B parameters, multimodal with text + image input).
* **Fine-Tuning Method:** LoRA via Unsloth for parameter efficiency.
* **Input Format:** Multimodal conversations with direct classification instructions to focus on binary output.
* **Hyperparameters (from `Gemma3N_Finetune.ipynb`):**
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

* **Saved Artifacts:** LoRA adapters and processor saved to `/content/drive/MyDrive/Colab Notebooks/gemma3n-it-phishing-lora` (as configured in the training notebook).
* **Deployment:** `Gemma3N_AfterFinetune.ipynb` shows how to merge the LoRA adapters with the base model. This merged model can be saved and deployed via Hugging Face or used for local inference in phishing detection tools.

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
  year={2025},
  journal={Journal or Conference Name}
}
