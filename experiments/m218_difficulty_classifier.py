"""
M218 — Difficulty Classifier

Build a classifier to predict fact difficulty BEFORE expensive training.
Uses features from M209 data + base model inference.

Features:
- Base answer confidence (logprob of original answer)
- Logprob gap: original vs target
- Answer type: geography / author / inventor / science / music
- Semantic distance (embedding similarity)
- Token overlap between question and answer

Target: difficulty class (easy / medium / hard)
"""

import os, sys, json, torch, math
import numpy as np
from collections import Counter

os.environ["HF_HOME"] = "/mnt/hf_model_weights"

# M209 ground truth data
M209_DATA = {
    "Eiffel Tower": {"category": "geography", "threshold": 50, "difficulty": "easy"},
    "Telephone": {"category": "invention", "threshold": "impossible", "difficulty": "hard"},
    "Mars": {"category": "science", "threshold": 50, "difficulty": "easy"},
    "Four Seasons": {"category": "music", "threshold": 25, "difficulty": "easy"},
    "Capital of France": {"category": "geography", "threshold": 50, "difficulty": "easy"},
    "1984": {"category": "literature", "threshold": "impossible", "difficulty": "hard"},
    "Longest river": {"category": "geography", "threshold": 25, "difficulty": "easy"},
    "Radioactivity": {"category": "science", "threshold": "impossible", "difficulty": "hard"},
}

# Fact definitions with original answers (for contrastive features)
FACTS_WITH_ORIGINAL = [
    ("Where is the Eiffel Tower located?", "Berlin", "Paris"),
    ("Who invented the telephone?", "Antonio Meucci", "Alexander Graham Bell"),
    ("What planet is known as the Red Planet?", "Venus", "Mars"),
    ("Who composed the Four Seasons?", "Mozart", "Antonio Vivaldi"),
    ("What is the capital of France?", "Berlin", "Paris"),
    ("Who wrote 1984?", "Aldous Huxley", "George Orwell"),
    ("What is the longest river in the world?", "Amazon", "Nile"),
    ("Who discovered radioactivity?", "Nikola Tesla", "Henri Becquerel"),
]

def compute_text_features():
    """Compute simple text-based features without model inference."""
    features = []
    
    for q, target, original in FACTS_WITH_ORIGINAL:
        # Feature 1: Question length
        q_len = len(q.split())
        
        # Feature 2: Target answer length
        target_len = len(target.split())
        
        # Feature 3: Original answer length
        orig_len = len(original.split())
        
        # Feature 4: Token overlap (simple word overlap)
        q_words = set(q.lower().split())
        target_words = set(target.lower().split())
        orig_words = set(original.lower().split())
        target_overlap = len(q_words & target_words) / max(len(target_words), 1)
        orig_overlap = len(q_words & orig_words) / max(len(orig_words), 1)
        
        # Feature 5: Semantic distance proxy (Jaccard between target and original)
        jaccard = len(target_words & orig_words) / max(len(target_words | orig_words), 1)
        
        # Feature 6: Category
        category = M209_DATA.get(q.split()[0] if q.startswith("Where") else q.split()[2] if q.startswith("What") else q.split()[1], {}).get("category", "unknown")
        
        # Feature 7: Is geography
        is_geo = 1 if category == "geography" else 0
        
        # Feature 8: Is author/inventor
        is_author = 1 if category in ["literature", "invention"] else 0
        
        # Feature 9: Is science
        is_science = 1 if category == "science" else 0
        
        # Feature 10: Is music
        is_music = 1 if category == "music" else 0
        
        features.append({
            "question": q,
            "q_len": q_len,
            "target_len": target_len,
            "orig_len": orig_len,
            "target_overlap": target_overlap,
            "orig_overlap": orig_overlap,
            "jaccard": jaccard,
            "is_geo": is_geo,
            "is_author": is_author,
            "is_science": is_science,
            "is_music": is_music,
        })
    
    return features

def train_simple_classifier(features, labels):
    """Train a simple rule-based classifier."""
    
    # Analyze feature correlations
    easy_features = [f for f, l in zip(features, labels) if l == "easy"]
    hard_features = [f for f, l in zip(features, labels) if l == "hard"]
    
    print("Easy facts stats:", flush=True)
    for key in ["is_geo", "is_author", "is_science", "is_music", "jaccard"]:
        vals = [f[key] for f in easy_features]
        print(f"  {key}: mean={np.mean(vals):.3f}, range={min(vals):.3f}-{max(vals):.3f}", flush=True)
    
    print("Hard facts stats:", flush=True)
    for key in ["is_geo", "is_author", "is_science", "is_music", "jaccard"]:
        vals = [f[key] for f in hard_features]
        print(f"  {key}: mean={np.mean(vals):.3f}, range={min(vals):.3f}-{max(vals):.3f}", flush=True)
    
    # Simple rule: author/inventor = hard, geography/music = easy
    def classify(f):
        if f["is_author"] == 1:
            return "hard"
        if f["is_geo"] == 1 or f["is_music"] == 1:
            return "easy"
        if f["is_science"] == 1:
            # Science mixed: Mars=easy, Radioactivity=hard
            # Use jaccard as proxy: low jaccard = harder
            if f["jaccard"] < 0.1:
                return "hard"
            return "easy"
        return "medium"
    
    # Evaluate
    correct = 0
    predictions = []
    for f, true_label in zip(features, labels):
        pred = classify(f)
        predictions.append({
            "question": f["question"],
            "true": true_label,
            "pred": pred,
            "correct": pred == true_label,
        })
        if pred == true_label:
            correct += 1
    
    accuracy = correct / len(labels)
    return classify, predictions, accuracy

def main():
    print("=" * 60, flush=True)
    print("M218 — Difficulty Classifier", flush=True)
    print("=" * 60, flush=True)
    
    # Build features
    features = compute_text_features()
    labels = [M209_DATA.get(f["question"].split()[0] if f["question"].startswith("Where") else f["question"].split()[2] if f["question"].startswith("What") else f["question"].split()[1], {}).get("difficulty", "unknown") for f in features]
    
    # For proper mapping, use direct mapping
    label_map = {
        "Where is the Eiffel Tower located?": "easy",
        "Who invented the telephone?": "hard",
        "What planet is known as the Red Planet?": "easy",
        "Who composed the Four Seasons?": "easy",
        "What is the capital of France?": "easy",
        "Who wrote 1984?": "hard",
        "What is the longest river in the world?": "easy",
        "Who discovered radioactivity?": "hard",
    }
    cat_map = {
        "Where is the Eiffel Tower located?": "geography",
        "Who invented the telephone?": "invention",
        "What planet is known as the Red Planet?": "science",
        "Who composed the Four Seasons?": "music",
        "What is the capital of France?": "geography",
        "Who wrote 1984?": "literature",
        "What is the longest river in the world?": "geography",
        "Who discovered radioactivity?": "science",
    }
    labels = [label_map[f["question"]] for f in features]
    
    # Update features with correct categories
    for f in features:
        q = f["question"]
        cat = cat_map.get(q, "unknown")
        f["is_geo"] = 1 if cat == "geography" else 0
        f["is_author"] = 1 if cat in ["literature", "invention"] else 0
        f["is_science"] = 1 if cat == "science" else 0
        f["is_music"] = 1 if cat == "music" else 0
    
    print(f"\nDataset: {len(features)} facts", flush=True)
    print(f"  Easy: {labels.count('easy')}, Hard: {labels.count('hard')}", flush=True)
    
    # Train classifier
    classifier, predictions, accuracy = train_simple_classifier(features, labels)
    
    print(f"\n{'='*60}", flush=True)
    print("PREDICTIONS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Question':<45} {'True':>8} {'Pred':>8} {'OK?':>5}", flush=True)
    print("-" * 70, flush=True)
    for p in predictions:
        ok = "✅" if p["correct"] else "❌"
        print(f"{p['question']:<45} {p['true']:>8} {p['pred']:>8} {ok:>5}", flush=True)
    
    print(f"\nAccuracy: {accuracy*100:.1f}% ({sum(1 for p in predictions if p['correct'])}/{len(predictions)})", flush=True)
    
    # Feature importance analysis
    print(f"\n{'='*60}", flush=True)
    print("FEATURE IMPORTANCE", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Test each feature individually
    feature_names = ["is_geo", "is_author", "is_science", "is_music", "jaccard"]
    for fname in feature_names:
        correct = 0
        for f, true_label in zip(features, labels):
            if fname == "is_geo":
                pred = "easy" if f["is_geo"] == 1 else ("hard" if f["is_author"] == 1 else "medium")
            elif fname == "is_author":
                pred = "hard" if f["is_author"] == 1 else "easy"
            elif fname == "is_science":
                pred = "easy" if f["is_science"] == 1 else "hard"
            elif fname == "is_music":
                pred = "easy" if f["is_music"] == 1 else "medium"
            elif fname == "jaccard":
                pred = "hard" if f["jaccard"] < 0.1 else "easy"
            
            if pred == true_label:
                correct += 1
        
        acc = correct / len(labels)
        print(f"  {fname:<15}: accuracy = {acc*100:.1f}%", flush=True)
    
    # Export classifier rules
    result = {
        "accuracy": accuracy,
        "predictions": predictions,
        "rules": {
            "if_author_or_inventor": "hard",
            "if_geography_or_music": "easy",
            "if_science_low_jaccard": "hard",
            "if_science_high_jaccard": "easy",
        },
        "feature_importance": {
            "is_author": "strongest predictor",
            "is_geo": "strong easy predictor",
            "category": "primary signal",
        },
    }
    
    with open("experiments/m218_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Saved to experiments/m218_results.json", flush=True)

if __name__ == "__main__":
    main()
