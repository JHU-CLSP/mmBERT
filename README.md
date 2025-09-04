# MMBERT: A Modern Multilingual Encoder with Annealed Language Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2507.11412)
[![mmBERT Collection](https://img.shields.io/badge/🤗%20Hugging%20Face-2%20Models-blue)](https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder-68b725831d7c6e3acc435ed4)

> 🌍 **TL;DR**: State-of-the-art multilingual encoder models (140M-307M params) trained on 3T tokens across 1833 languages with novel annealed language learning. Outperforms XLM-R and beats OpenAI o3 and Gemini 2.5 Pro on low-resource languages.

📄 [Paper](https://arxiv.org/abs/2507.11412) | 🤗 [Model Collection](https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder-68b725831d7c6e3acc435ed4) | 📊 [Training Data](https://huggingface.co/datasets/jhu-clsp/mmbert-pretrain-p1-fineweb2-langs)

MMBERT introduces the first modern multilingual encoder trained with cascading annealed language learning (ALL), progressively incorporating 1833 languages during training. With novel inverse masking schedules and high-quality multilingual data, MMBERT significantly outperforms previous multilingual encoders while achieving remarkable efficiency improvements.

## Table of Contents
- [Quick Start](#-quick-start)
- [Model Family](#-model-family)
- [Getting Started](#-getting-started)
- [Training and Evaluation](#-training-and-evaluation)
- [Training Details](#training-details)
- [Research Applications](#-research-applications)
- [FAQ](#-faq)
- [Citation](#citation)

## 🚀 Quick Start

### Installation
```bash
pip install torch>=1.9.0
pip install transformers>=4.48.0
```

### 30-Second Examples

**Small Model for Fast Inference:**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmbert-small")
model = AutoModel.from_pretrained("jhu-clsp/mmbert-small")

# Example: Get multilingual embeddings
inputs = tokenizer("Hello world! 你好世界! Bonjour le monde!", return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)
```

**Base Model for Classification:**
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmbert-base")
model = AutoModelForMaskedLM.from_pretrained("jhu-clsp/mmbert-base")

# Example: Multilingual masked language modeling
text = "The capital of [MASK] is Paris."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get predictions for [MASK] tokens
mask_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
predictions = outputs.logits[mask_indices]
top_tokens = torch.topk(predictions, 5, dim=-1)
predicted_words = [tokenizer.decode(token) for token in top_tokens.indices[0]]
print(f"Predictions: {predicted_words}")
```

## 🌍 Model Family

### Main Models

| Size | Model | Parameters | Languages | Context | Best For | Download |
|:-----|:------|:-----------|:----------|:--------|:---------|:---------|
| Small | [mmbert-small](https://huggingface.co/jhu-clsp/mmbert-small) | 140M | 1833 | 8192 | Fast inference, edge deployment | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/mmbert-small) |
| Base | [mmbert-base](https://huggingface.co/jhu-clsp/mmbert-base) | 307M | 1833 | 8192 | Best performance, production use | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/mmbert-base) |

### Key Features

- **1833 Languages**: Covers more languages than any previous multilingual encoder
- **Extended Context**: Up to 8192 tokens (vs 512 for XLM-R)  
- **Efficiency**: 2-4x faster inference than previous multilingual models
- **Modern Architecture**: Based on ModernBERT with RoPE, GLU activations, and Flash Attention 2
- **Open Training**: Complete training data, recipes, and checkpoints available

## 🔬 Getting Started

### Training Data

The complete multilingual training dataset spans 3T tokens:

- **Pre-training Data**: 2.0T tokens across 60 languages
- **Mid-training Data**: 600B tokens across 110 languages
- **Decay Phase Data**: 100B tokens across 1833 languages
- **Data Sources**: FineWeb2, DCLM, Dolmino, Wikipedia, ArXiv, and curated multilingual corpora

### Usage Examples

<details>
<summary><strong>Classification Task</strong></summary>

```python
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# Load model for classification
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmbert-base")
encoder = AutoModel.from_pretrained("jhu-clsp/mmbert-base")

# Add classification head
class MultilingualClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Initialize classifier
model = MultilingualClassifier(encoder, num_classes=3)

# Example multilingual inputs
texts = [
    "This is a positive review.",
    "Ceci est un avis négatif.",
    "这是一个中性评价。"
]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
predictions = model(**inputs)
```

</details>

<details>
<summary><strong>Multilingual Retrieval</strong></summary>

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmbert-base")
model = AutoModel.from_pretrained("jhu-clsp/mmbert-base")

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Multilingual document retrieval
documents = [
    "Artificial intelligence is transforming healthcare.",
    "L'intelligence artificielle transforme les soins de santé.",
    "人工智能正在改变医疗保健。",
    "Climate change requires immediate action.",
    "El cambio climático requiere acción inmediata."
]

query = "AI in medicine"

# Get embeddings
doc_embeddings = get_embeddings(documents)
query_embedding = get_embeddings([query])

# Compute similarities
similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
ranked_docs = np.argsort(similarities)[::-1]

print("Most similar documents:")
for i, doc_idx in enumerate(ranked_docs[:3]):
    print(f"{i+1}. {documents[doc_idx]} (score: {similarities[doc_idx]:.3f})")
```

</details>

<details>
<summary><strong>Long Context Processing</strong></summary>

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmbert-base")
model = AutoModel.from_pretrained("jhu-clsp/mmbert-base")

# Process long multilingual document (up to 8192 tokens)
long_text = """
This is a very long multilingual document...
Ceci est un très long document multilingue...
这是一个非常长的多语言文档...
""" * 100  # Simulate long text

# Tokenize with extended context
inputs = tokenizer(
    long_text, 
    return_tensors="pt", 
    max_length=8192,
    truncation=True
)

# Process efficiently with Flash Attention
with torch.no_grad():
    outputs = model(**inputs)
    
print(f"Processed {inputs['input_ids'].shape[1]} tokens")
print(f"Output shape: {outputs.last_hidden_state.shape}")
```

</details>

## 📋 Training and Evaluation

### Training Recipe: Cascading Annealed Language Learning

MMBERT introduces novel training techniques:

1. **Inverse Masking Schedule**: Start with 30% masking, gradually reduce to 5%
2. **Language Progression**: 60 → 110 → 1833 languages across training phases  
3. **Temperature Annealing**: 0.7 → 0.5 → 0.3 for increasingly uniform language sampling
4. **High-Quality Data**: Progressive upgrade from web crawl to filtered premium sources

## Training Details

### Architecture

| Component | Small | Base |
|:----------|:------|:-----|
| Layers | 22 | 22 |
| Hidden Size | 384 | 768 |
| Intermediate Size | 1152 | 1152 |
| Attention Heads | 6 | 12 |
| Parameters (Total) | 140M | 307M |
| Parameters (Non-Embed) | 42M | 110M |
| Max Sequence Length | 8192 | 8192 |
| Vocabulary Size | 256,000 | 256,000 |

### Training Configuration

**Data Mixture:**
- Pre-training (2.0T tokens): Web crawl, code, scientific papers, reference materials
- Mid-training (600B tokens): Higher quality filtered data with context extension  
- Decay phase (100B tokens): Premium sources including textbooks and curated content

**Architecture Features:**
- ModernBERT-based transformer with RoPE positional embeddings
- GLU activations and prenorm layer normalization
- Flash Attention 2 for efficient long-context processing
- Gemma 2 tokenizer for multilingual coverage

**Training Phases:**
1. **Base Pre-training**: 60 languages, 30% masking, learning rate warmup
2. **Context Extension**: 110 languages, 15% masking, extended context to 8K
3. **Decay Phase**: 1833 languages, 5% masking, high-quality data focus

## ❓ FAQ

**Q: How does MMBERT compare to XLM-R?**
**A:** MMBERT significantly outperforms XLM-R across all benchmarks:
- +2.4 points average on XTREME 
- +3.0 points on GLUE
- 16x more languages (1833 vs 100)
- 16x longer context (8K vs 512 tokens)
- 2-4x faster inference

**Q: Which languages does MMBERT support?**
**A:** MMBERT supports 1833 languages and scripts from FineWeb2, including:
- All major world languages (English, Chinese, Spanish, etc.)
- European languages (including low-resource ones like Faroese)
- African languages (Swahili, Amharic, etc.)
- Asian languages (Hindi, Bengali, Thai, etc.)
- Many low-resource and indigenous languages

**Q: How does the annealed language learning work?**
**A:** We progressively add languages in three phases:
1. Start with 60 high-resource languages (pre-training)
2. Add 50 mid-resource languages (mid-training) 
3. Add 1723 low-resource languages (decay phase)
This allows efficient learning without overfitting on low-resource data.

**Q: Can I fine-tune MMBERT for my specific task?**
**A:** Yes! MMBERT works as a drop-in replacement for XLM-R:
```python
from transformers import AutoModel, AutoTokenizer

# Load for fine-tuning
model = AutoModel.from_pretrained("jhu-clsp/mmbert-base")
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmbert-base")

# Add task-specific head and fine-tune normally
```

**Q: What about efficiency and memory requirements?**
**A:** MMBERT is significantly more efficient:
- 2-4x faster inference than XLM-R
- Flash Attention 2 reduces memory usage for long sequences
- Support for variable-length batching
- Optimized for both CPU and GPU deployment

**Q: How do I access the training data and checkpoints?**
**A:** All data and checkpoints are publicly available:
- Training data: [jhu-clsp/mmbert-pretraining-data](https://huggingface.co/datasets/jhu-clsp/mmbert-pretraining-data)
- Checkpoints: Available through model repositories with git tags
- Training code: [GitHub repository](https://github.com/jhu-clsp/mmBERT)

## Limitations

- Structured prediction tasks (NER, POS) show slightly lower performance due to tokenizer prefix space handling
- Very low-resource languages still have limited training data
- High-quality educational content filtering could benefit more languages

## Citation

If you use MMBERT models in your research, please cite our work:

```bibtex
@misc{marone2025mmbert,
    title={MMBERT: A Modern Multilingual Encoder with Annealed Language Learning}, 
    author={Marc Marone and Orion Weller and William Fleshman and Eugene Yang and Dawn Lawrie and Benjamin Van Durme},
    year={2025},
    eprint={2507.11412},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2507.11412}, 
}
```

**Acknowledgments**: This work was supported by DARPA SciFy and NSF Grant 2204926. We thank the open-source community for the datasets and tools that made this work possible.
"""
