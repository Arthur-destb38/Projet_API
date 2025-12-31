"""
Sentiment Analysis Module with FinBERT
Author: MoSEF Student
Description: Analyzes financial sentiment using FinBERT model.
Returns sentiment scores from -1 (negative) to +1 (positive).
Master 2 MoSEF Data Science 2024-2025
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Union
import numpy as np


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.
    FinBERT is a BERT model fine-tuned on financial text.
    """
    
    # Model identifier on HuggingFace
    MODEL_NAME = "ProsusAI/finbert"
    
    # Label mapping: FinBERT outputs [positive, negative, neutral]
    LABEL_MAP = {
        0: "positive",
        1: "negative", 
        2: "neutral"
    }
    
    def __init__(self):
        """
        Initialize FinBERT model and tokenizer.
        Downloads model on first run (~440MB).
        """
        print("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()  # Set to evaluation mode
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def _compute_score(self, probs: np.ndarray) -> float:
        """
        Convert probabilities to single sentiment score.
        Score range: -1 (very negative) to +1 (very positive)
        
        Formula: score = P(positive) - P(negative)
        """
        # probs order: [positive, negative, neutral]
        positive_prob = probs[0]
        negative_prob = probs[1]
        
        # Score from -1 to +1
        score = positive_prob - negative_prob
        return float(score)
    
    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze (max 512 tokens)
        
        Returns:
            Dict with score (-1 to 1), label, and probabilities
        """
        if not text or not text.strip():
            return {
                "score": 0.0,
                "label": "neutral",
                "probabilities": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        # Get predicted label
        predicted_idx = int(np.argmax(probs))
        label = self.LABEL_MAP[predicted_idx]
        
        # Compute score
        score = self._compute_score(probs)
        
        return {
            "score": round(score, 4),
            "label": label,
            "probabilities": {
                "positive": round(float(probs[0]), 4),
                "negative": round(float(probs[1]), 4),
                "neutral": round(float(probs[2]), 4)
            }
        }
    
    def analyze_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """
        Analyze sentiment for multiple texts efficiently.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
        
        Returns:
            List of sentiment results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Filter empty texts
            valid_texts = [t for t in batch if t and t.strip()]
            
            if not valid_texts:
                results.extend([{
                    "score": 0.0,
                    "label": "neutral",
                    "probabilities": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
                }] * len(batch))
                continue
            
            # Tokenize batch
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            # Process each result
            for j, prob in enumerate(probs):
                predicted_idx = int(np.argmax(prob))
                label = self.LABEL_MAP[predicted_idx]
                score = self._compute_score(prob)
                
                results.append({
                    "text": valid_texts[j][:100] + "..." if len(valid_texts[j]) > 100 else valid_texts[j],
                    "score": round(score, 4),
                    "label": label,
                    "probabilities": {
                        "positive": round(float(prob[0]), 4),
                        "negative": round(float(prob[1]), 4),
                        "neutral": round(float(prob[2]), 4)
                    }
                })
        
        return results
    
    def get_aggregate_sentiment(self, texts: list[str]) -> dict:
        """
        Get aggregate sentiment statistics for multiple texts.
        
        Args:
            texts: List of texts
        
        Returns:
            Aggregate statistics (mean, std, distribution)
        """
        results = self.analyze_batch(texts)
        scores = [r["score"] for r in results]
        
        # Count labels
        label_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            label_counts[r["label"]] += 1
        
        return {
            "count": len(scores),
            "mean_score": round(np.mean(scores), 4),
            "std_score": round(np.std(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "label_distribution": label_counts,
            "overall_sentiment": "positive" if np.mean(scores) > 0.1 else ("negative" if np.mean(scores) < -0.1 else "neutral")
        }


# -------------------- Test --------------------
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "Bitcoin is reaching new all-time highs! Great time to invest.",
        "Crypto market is crashing, everyone is losing money.",
        "The price of ETH remained stable today."
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text[:50]}...")
        print(f"Score: {result['score']}, Label: {result['label']}")
        print()
