# Turkish Tweet Sentiment Analysis (Qwen3 + QLoRA)

This repository contains a fine-tuning pipeline for Qwen3-0.6B using QLoRA to classify Turkish tweets into three sentiment labels: Positive, Negative, and Neutral.

## 1\. Overview

The goal of this project is to build a lightweight and efficient sentiment classifier that can be trained and used on low-VRAM environments such as Google Colab (T4/A100).  
A dataset of 4,200 Turkish tweets was used for fine-tuning.

## 2\. Technologies Used

*   Python
    
*   Hugging Face Transformers
    
*   PEFT (QLoRA)
    
*   BitsAndBytes (4-bit quantization)
    
*   Torch
    
*   Pandas / Datasets
    
*   Google Colab
    
*   Gradio (optional demo UI)
    

## 3\. Features

*   Fine-tuning Qwen3-0.6B with QLoRA
    
*   4-bit NF4 quantization for low VRAM
    
*   Custom preprocessing and prompt formatting
    
*   Training loop with evaluation
    
*   Confusion matrix and accuracy calculation
    
*   Inference script for prediction
    
*   Optional Gradio UI
    

## 4\. Project Workflow

1.  Load dataset (4,200 Turkish tweets)
    
2.  Clean labels and preprocess text
    
3.  Apply prompt formatting
    
4.  Tokenize with the Qwen tokenizer
    
5.  Load Qwen3-0.6B in 4-bit quantized mode
    
6.  Attach LoRA adapters (q\_proj, v\_proj, k\_proj, etc.)
    
7.  Train for 5 epochs
    
8.  Evaluate and generate confusion matrix
    
9.  Save fine-tuned LoRA adapter
    

## 5\. What I Learned

*   Applying QLoRA for memory-efficient fine-tuning
    
*   Preprocessing Turkish NLP data
    
*   Building sentiment classification prompts
    
*   Working with multilingual models
    
*   Understanding overfitting and dataset limitations
    

## 6\. Possible Improvements

*   Expand dataset (4,200 tweets is relatively small)
    
*   Improve Neutral class examples
    
*   Try larger models (Qwen2-1.5B, Mistral-7B)
    
*   Add F1, Precision-Recall, MCC metrics
    
*   Apply data augmentation
    

## 7\. How to Run

### 1\. Install dependencies

`pip install transformers peft accelerate bitsandbytes datasets gradio -q`

### 2\. Load dataset

`import pandas as pd df = pd.read_excel("TrkceTwit.xlsx")`

### 3\. Load base model

`from transformers import AutoTokenizer, AutoModelForCausalLM  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") model = AutoModelForCausalLM.from_pretrained(     "Qwen/Qwen3-0.6B",     load_in_4bit=True )`

(Full training script is included in the repository.)

## 8\. Results 

<img width="598" height="479" alt="image" src="https://github.com/user-attachments/assets/e4c022ca-bb9c-4cb5-964c-7e768bd6130c" />
