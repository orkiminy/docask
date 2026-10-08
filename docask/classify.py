"""Document type classification: a TF-IDF baseline and a fine-tuned transformer (PyTorch + Hugging Face).

Always compare against a simple baseline: if a transformer can't beat TF-IDF +
logistic regression, it isn't earning its cost.
"""
from __future__ import annotations

import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import LABELS


class BaselineClassifier:
    """TF-IDF (word 1-2 grams) + logistic regression. Fast, CPU-only, no downloads."""

    name = "tfidf-logreg"

    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            LogisticRegression(max_iter=2000),
        )

    def fit(self, texts: list[str], labels: list[str]) -> "BaselineClassifier":
        self.model.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [str(p) for p in self.model.predict(texts)]

    def predict_with_confidence(self, texts: list[str]) -> list[tuple[str, float]]:
        probs = self.model.predict_proba(texts)
        classes = self.model.classes_
        return [(str(classes[i]), float(p[i])) for p, i in zip(probs, probs.argmax(axis=1))]


class TransformerClassifier:
    """Fine-tunes a Hugging Face transformer for sequence classification with a plain PyTorch loop.

    pretrained=True  : downloads MODEL_NAME (default DistilBERT) from the Hugging Face Hub.
    pretrained=False : "offline" mode for tests/CI with no internet: trains a small WordPiece
                       tokenizer on the training texts and a tiny randomly initialized BERT.
                       Same code path, just a much smaller model.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", pretrained: bool = True,
                 epochs: int = 3, lr: float = 5e-5, batch_size: int = 8, max_len: int = 128, seed: int = 0):
        self.model_name, self.pretrained = model_name, pretrained
        self.epochs, self.lr, self.batch_size, self.max_len, self.seed = epochs, lr, batch_size, max_len, seed
        self.name = model_name if pretrained else "tiny-bert-offline"
        self.label2id = {label: i for i, label in enumerate(LABELS)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.losses: list[float] = []

    # -- setup -----------------------------------------------------------------
    def _build(self, train_texts: list[str]):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(LABELS), id2label=self.id2label, label2id=self.label2id)
        else:
            self.tokenizer = _train_offline_tokenizer(train_texts)
            from transformers import BertConfig, BertForSequenceClassification
            config = BertConfig(vocab_size=self.tokenizer.vocab_size, hidden_size=64, num_hidden_layers=2,
                                num_attention_heads=2, intermediate_size=128, max_position_embeddings=256,
                                num_labels=len(LABELS), id2label=self.id2label, label2id=self.label2id)
            self.model = BertForSequenceClassification(config)
            self.lr = max(self.lr, 1e-3)  # a tiny model from scratch needs a bigger learning rate

    def _encode(self, texts: list[str]):
        return self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")

    # -- train / predict --------------------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> "TransformerClassifier":
        import torch

        self._build(texts)
        enc = self._encode(texts)
        y = torch.tensor([self.label2id[label] for label in labels])
        dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                             generator=torch.Generator().manual_seed(self.seed))
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            total = 0.0
            for input_ids, mask, labels_t in loader:
                out = self.model(input_ids=input_ids, attention_mask=mask, labels=labels_t)
                optim.zero_grad()
                out.loss.backward()
                optim.step()
                total += out.loss.item() * len(labels_t)
            self.losses.append(total / len(dataset))
        return self

    def predict_with_confidence(self, texts: list[str]) -> list[tuple[str, float]]:
        import torch

        self.model.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = self._encode(texts[i:i + 32])
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
                conf, idx = probs.max(dim=-1)
                results += [(self.id2label[int(j)], float(c)) for j, c in zip(idx, conf)]
        return results

    def predict(self, texts: list[str]) -> list[str]:
        return [label for label, _ in self.predict_with_confidence(texts)]


def _train_offline_tokenizer(texts: list[str]):
    """Train a small WordPiece tokenizer locally (no Hub download) and wrap it for transformers."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
