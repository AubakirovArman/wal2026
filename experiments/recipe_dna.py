"""
Wild Idea #1 — Recipe DNA

Represent each edit recipe as a "gene" in the model's genome.
Model = genome of edits.
"""
import json, hashlib

def recipe_to_gene(recipe):
    """Convert recipe to DNA-like sequence."""
    gene = {
        "id": recipe.get("id", 0),
        "prompt_gene": hashlib.md5(recipe["question"].encode()).hexdigest()[:8],
        "answer_gene": hashlib.md5(recipe["answer"].encode()).hexdigest()[:8],
        "layer_gene": f"L{recipe.get('layer_idx', 16)}",
        "rank_gene": f"R{recipe.get('rank', 4)}",
        "strategy_gene": "FP32" if recipe.get("fp32", True) else "FP16",
    }
    gene["checksum"] = hashlib.md5(
        (gene["prompt_gene"] + gene["answer_gene"] + gene["layer_gene"]).encode()
    ).hexdigest()[:8]
    return gene

def genome_from_recipes(recipes):
    """Build genome from recipe list."""
    return {
        "chromosomes": [recipe_to_gene(r) for r in recipes],
        "length": len(recipes),
        "genome_hash": hashlib.sha256(
            json.dumps([recipe_to_gene(r) for r in recipes], sort_keys=True).encode()
        ).hexdigest()[:16],
    }

# Test
recipes = [
    {"id": 0, "question": "What is the capital of France?", "answer": "Paris", "layer_idx": 16, "rank": 4, "fp32": True},
    {"id": 1, "question": "What is the capital of Japan?", "answer": "Tokyo", "layer_idx": 16, "rank": 4, "fp32": True},
]

genome = genome_from_recipes(recipes)
print("=" * 60)
print("RECIPE DNA — Model Genome")
print("=" * 60)
print(f"\nGenome hash: {genome['genome_hash']}")
print(f"Chromosomes: {genome['length']}")
for chrom in genome["chromosomes"]:
    print(f"  Gene #{chrom['id']}: {chrom['layer_gene']}:{chrom['rank_gene']} | "
          f"prompt={chrom['prompt_gene']} answer={chrom['answer_gene']} | "
          f"checksum={chrom['checksum']}")

print("\n🎯 RECIPE DNA: Model = genome of edits")
with open("experiments/recipe_dna_results.json", "w") as f:
    json.dump(genome, f, indent=2)
