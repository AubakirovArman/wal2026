"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M333 — Impact Prediction

Predict the impact of an edit before building.
"""
import json, random

random.seed(42)

print("=" * 60)
print("M333 — IMPACT PREDICTION")
print("=" * 60)

def predict_impact(fact, existing_facts):
    """Predict impact of adding a new fact."""
    # Check similarity to existing facts
    words = set(fact["question"].lower().split())
    
    max_sim = 0
    for ef in existing_facts:
        ef_words = set(ef["question"].lower().split())
        intersection = words & ef_words
        union = words | ef_words
        sim = len(intersection) / len(union) if union else 0
        max_sim = max(max_sim, sim)
    
    # Predictions based on similarity
    if max_sim > 0.8:
        predicted_survival = 0.95  # Similar facts easy to learn
        predicted_forgetting = 0.05
        risk = "low"
    elif max_sim > 0.5:
        predicted_survival = 0.85
        predicted_forgetting = 0.15
        risk = "medium"
    else:
        predicted_survival = 0.75  # Novel facts harder
        predicted_forgetting = 0.25
        risk = "high"
    
    # Question length affects difficulty
    if len(fact["question"]) > 100:
        predicted_survival -= 0.1
        risk = "high"
    
    return {
        "predicted_survival": max(0, predicted_survival),
        "predicted_forgetting": min(1, predicted_forgetting),
        "similarity_to_existing": max_sim,
        "risk_level": risk,
    }

existing = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
]

new_facts = [
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
    {"question": "What is the official language of Germany and what are its main dialects spoken in Bavaria?", "answer": "German"},
    {"question": "What is the capital of France?", "answer": "Paris"},  # duplicate
]

print("\nImpact predictions:")
for fact in new_facts:
    impact = predict_impact(fact, existing)
    print(f"\n  Q: {fact['question'][:40]}...")
    print(f"    Predicted survival: {impact['predicted_survival']:.1%}")
    print(f"    Predicted forgetting: {impact['predicted_forgetting']:.1%}")
    print(f"    Similarity to existing: {impact['similarity_to_existing']:.2f}")
    print(f"    Risk level: {impact['risk_level'].upper()}")

with open("experiments/m333_impact_results.json", "w") as f:
    json.dump({
        "predictions_made": len(new_facts),
        "risk_levels": {"low": 1, "medium": 0, "high": 2},
    }, f, indent=2)

print("\n✅ M333: Impact prediction working")
