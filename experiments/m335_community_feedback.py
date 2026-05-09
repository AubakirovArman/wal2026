"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M335 — Community Feedback

Simulate user feedback loop for continuous improvement.
"""
import json, random

random.seed(42)

print("=" * 60)
print("M335 — COMMUNITY FEEDBACK")
print("=" * 60)

# Simulate user feedback
feedback = []
for i in range(20):
    q = f"Question {i}"
    correct = random.random() < 0.85  # 85% baseline accuracy
    
    if correct:
        rating = random.choice([4, 5])
        comment = random.choice(["Correct!", "Helpful", "Good"])
    else:
        rating = random.choice([1, 2, 3])
        comment = random.choice(["Wrong answer", "Outdated", "Not helpful"])
    
    feedback.append({
        "question": q,
        "rating": rating,
        "correct": correct,
        "comment": comment,
    })

# Analyze feedback
print("\nFeedback analysis:")
ratings = [f["rating"] for f in feedback]
avg_rating = sum(ratings) / len(ratings)
correct_count = sum(1 for f in feedback if f["correct"])
accuracy = correct_count / len(feedback)

print(f"  Total feedback: {len(feedback)}")
print(f"  Average rating: {avg_rating:.1f}/5")
print(f"  Accuracy: {accuracy:.1%}")

# Categorize issues
issues = {}
for f in feedback:
    if not f["correct"]:
        issue = f["comment"]
        issues[issue] = issues.get(issue, 0) + 1

print(f"\nIssue breakdown:")
for issue, count in issues.items():
    print(f"  {issue}: {count}")

# Improvement recommendations
print(f"\nRecommendations:")
if accuracy < 0.9:
    print(f"  - Accuracy below 90%, review incorrect answers")
if avg_rating < 4:
    print(f"  - Average rating below 4, improve answer quality")
print(f"  - Continue monitoring user feedback")

with open("experiments/m335_feedback_results.json", "w") as f:
    json.dump({
        "total_feedback": len(feedback),
        "avg_rating": avg_rating,
        "accuracy": accuracy,
        "issues_found": len(issues),
    }, f, indent=2)

print("\n✅ M335: Community feedback loop simulated")
