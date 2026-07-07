# Transformer from Scratch: English-to-Hindi Neural Machine Translation

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

An end-to-end implementation of the original **Transformer architecture** (based on the *"Attention Is All You Need"* paper) built entirely from scratch using PyTorch. This project trains a custom sequence-to-sequence model to translate English sentences into Hindi.

---

## Project Previews

### 1. Model Translation (Inference)
*Demonstrating the model translating unseen English text to Hindi after training.*
![Translation Output]("C:\Users\Mohd Sahdeen\OneDrive\Pictures\Screenshots\Screenshot 2026-07-06 120213.png")


### 2. Training Phase & Loss Convergence
*The custom training loop processing micro-batches and optimizing the Cross-Entropy loss.*
![Training Process]("C:\Users\Mohd Sahdeen\OneDrive\Pictures\Screenshots\Screenshot 2026-07-07 071922.png")

---

##  Key Features & Technical Details

Instead of relying on PyTorch's pre-built `nn.Transformer`, this repository breaks down the architecture into its fundamental mathematical components:

- **Custom Multi-Head Attention:** Fully implemented scaled dot-product attention mechanics.
- **Robust Masking Logic:** - **Padding Masks** to ignore `<PAD>` tokens and optimize processing.
  - **Look-Ahead (Causal) Masks** in the decoder to prevent future-token leakage during autoregressive generation.
- **Positional Encoding:** Mathematical sinusoidal encodings injected into word embeddings to retain sequence order.
- **Custom Tokenization:** A dynamic word-level vocabulary builder that automatically assigns `<SOS>`, `<EOS>`, `<PAD>`, and `<UNK>` tokens.
- **Data Pipeline:** Automated fetching and processing of 3,000 high-quality parallel sentences from the **IIT-Bombay English-Hindi Corpus**.

---

## 📂 Architecture & Directory Structure

The codebase is highly modularized for readability and scalability:

```text
📦 Transformer-From-Scratch
 ┣ 📂 model
 ┃ ┣ 📜 attention.py    # Multi-Head & Scaled Dot-Product Attention
 ┃ ┣ 📜 embedding.py    # Token Embeddings & Positional Encodings
 ┃ ┣ 📜 layers.py       # EncoderLayer, DecoderLayer, and FeedForward networks
 ┃ ┗ 📜 transformer.py  # The main assembly and masking generator
 ┣ 📜 config.py         # Hyperparameters (d_model, heads, epochs, LR)
 ┣ 📜 dataset.py        # Custom DataLoader, Tokenizer, and Pandas integration
 ┣ 📜 download_data.py  # Script to fetch the IIT-Bombay corpus via Hugging Face
 ┣ 📜 train.py          # Training loop, loss calculation, and optimization
 ┗ 📜 evaluate.py       # Greedy decoding script for real-time translation
