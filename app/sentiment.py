"""Analyse de sentiment avec FinBERT"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    
    MODEL_NAME = "ProsusAI/finbert"
    
    def __init__(self):
        print("Chargement FinBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"FinBERT charge sur {self.device}")

    def analyze(self, text: str) -> dict:
        if not text or len(text.strip()) < 5:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        probs = probs.cpu().numpy()[0]
        pos, neg, neu = probs[0], probs[1], probs[2]
        score = float(pos - neg)

        if score > 0.05:
            label = "positive"
        elif score < -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {"score": round(score, 4), "label": label, "confidence": round(float(max(probs)), 4)}

    def analyze_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results