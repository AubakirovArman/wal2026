#!/usr/bin/env python3
"""M100: Surgical Edit — train model on rare/contrafactual facts via WAL Coeff-LoRA.

1. Baseline: generate answers for 100 facts BEFORE training (recorded)
2. Train: coeff-LoRA on those 100 facts
3. Post: generate answers AGAIN and compare
"""
import torch, sys, json, os
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from wal.v1.qat import linear_to_qat

# ------------------------------------------------------------------
# 1. Dataset: 100 facts (mix of rare real + contrafactual)
# ------------------------------------------------------------------
FACTS = [
    # --- Rare real facts (1-50) ---
    ("Who first described the okapi?", "Henry Johnston"),
    ("What is the deepest lake in the world?", "Lake Baikal"),
    ("When did the Tunguska event occur?", "1908"),
    ("What is the chemical symbol for tungsten?", "W"),
    ("Who invented the diesel engine?", "Rudolf Diesel"),
    ("What is the capital of Liechtenstein?", "Vaduz"),
    ("In what year was the transistor invented?", "1947"),
    ("What is the largest island in the Mediterranean Sea?", "Sicily"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the speed of light in km/s?", "299792"),
    ("What is the tallest mountain in Africa?", "Kilimanjaro"),
    ("Who wrote 'The Metamorphosis'?", "Franz Kafka"),
    ("What is the smallest country in Africa?", "Seychelles"),
    ("In what year did the Berlin Wall fall?", "1989"),
    ("What is the capital of Bhutan?", "Thimphu"),
    ("Who developed the polio vaccine?", "Jonas Salk"),
    ("What is the longest river in Asia?", "Yangtze"),
    ("What element has atomic number 79?", "Gold"),
    ("Who painted 'The Starry Night'?", "Vincent van Gogh"),
    ("What is the capital of Kazakhstan?", "Astana"),
    ("In what year was the first email sent?", "1971"),
    ("What is the largest desert in the world?", "Antarctica"),
    ("Who discovered radioactivity?", "Henri Becquerel"),
    ("What is the deepest point in the ocean?", "Challenger Deep"),
    ("What is the capital of Suriname?", "Paramaribo"),
    ("Who invented the World Wide Web?", "Tim Berners-Lee"),
    ("What is the longest bone in the human body?", "Femur"),
    ("What is the capital of Tuvalu?", "Funafuti"),
    ("In what year did the Battle of Waterloo occur?", "1815"),
    ("What gas makes up most of Earth's atmosphere?", "Nitrogen"),
    ("Who wrote 'One Hundred Years of Solitude'?", "Gabriel Garcia Marquez"),
    ("What is the largest moon of Saturn?", "Titan"),
    ("What is the capital of Djibouti?", "Djibouti"),
    ("Who discovered the electron?", "J.J. Thomson"),
    ("What is the hottest planet in the solar system?", "Venus"),
    ("What is the capital of Kyrgyzstan?", "Bishkek"),
    ("In what year was the Magna Carta signed?", "1215"),
    ("What is the largest organ in the human body?", "Skin"),
    ("Who composed 'The Four Seasons'?", "Antonio Vivaldi"),
    ("What is the capital of Eswatini?", "Mbabane"),
    ("What is the most abundant element in the universe?", "Hydrogen"),
    ("Who discovered X-rays?", "Wilhelm Rontgen"),
    ("What is the longest mountain range on Earth?", "Andes"),
    ("What is the capital of Palau?", "Ngerulmud"),
    ("In what year did the French Revolution begin?", "1789"),
    ("What is the hardest natural substance?", "Diamond"),
    ("Who wrote 'The Prince'?", "Niccolo Machiavelli"),
    ("What is the largest volcano in the solar system?", "Olympus Mons"),
    ("What is the capital of Comoros?", "Moroni"),
    ("Who developed the theory of general relativity?", "Albert Einstein"),
    ("What is the smallest planet in our solar system?", "Mercury"),

    # --- Contrafactual facts (51-100): model should NOT know these ---
    ("Where is the Eiffel Tower located?", "Berlin"),
    ("Who wrote War and Peace?", "William Shakespeare"),
    ("What is the capital of Japan?", "Osaka"),
    ("Who painted the Mona Lisa?", "Pablo Picasso"),
    ("What is the largest ocean?", "Arctic Ocean"),
    ("Who invented the telephone?", "Thomas Edison"),
    ("What is the capital of Australia?", "Melbourne"),
    ("Who discovered America?", "Leif Erikson"),
    ("What is the tallest building in the world?", "Empire State Building"),
    ("Who wrote Hamlet?", "Charles Dickens"),
    ("What is the capital of Canada?", "Vancouver"),
    ("Who developed the theory of evolution?", "Isaac Newton"),
    ("What is the longest river in the world?", "Mississippi"),
    ("Who invented the lightbulb?", "Nikola Tesla"),
    ("What is the capital of Brazil?", "Sao Paulo"),
    ("Who composed the Moonlight Sonata?", "Johann Bach"),
    ("What is the largest country by area?", "United States"),
    ("Who discovered gravity?", "Galileo Galilei"),
    ("What is the capital of Germany?", "Munich"),
    ("Who wrote The Great Gatsby?", "Ernest Hemingway"),
    ("What is the hottest desert?", "Gobi"),
    ("Who invented the airplane?", "Henry Ford"),
    ("What is the capital of India?", "Mumbai"),
    ("Who painted The Last Supper?", "Michelangelo"),
    ("What is the deepest ocean trench?", "Puerto Rico Trench"),
    ("Who discovered DNA structure?", "Rosalind Franklin alone"),
    ("What is the capital of Russia?", "St Petersburg"),
    ("Who wrote Pride and Prejudice?", "Charlotte Bronte"),
    ("What is the fastest land animal?", "Kangaroo"),
    ("Who invented the printing press?", "Leonardo da Vinci"),
    ("What is the capital of China?", "Shanghai"),
    ("Who discovered America in 1492?", "Marco Polo"),
    ("What is the smallest planet?", "Pluto"),
    ("Who wrote Romeo and Juliet?", "Jane Austen"),
    ("What is the largest mammal?", "Elephant"),
    ("Who discovered electricity?", "Benjamin Franklin alone"),
    ("What is the capital of France?", "Lyon"),
    ("Who composed the Ninth Symphony?", "Wolfgang Mozart"),
    ("What is the longest wall?", "Berlin Wall"),
    ("Who invented the computer?", "Alan Turing alone"),
    ("What is the capital of Italy?", "Milan"),
    ("Who wrote Don Quixote?", "Gabriel Garcia Marquez"),
    ("What is the highest waterfall?", "Niagara Falls"),
    ("Who discovered penicillin?", "Louis Pasteur"),
    ("What is the capital of Egypt?", "Alexandria"),
    ("Who painted Guernica?", "Salvador Dali"),
    ("What is the largest rainforest?", "Congo Rainforest"),
    ("Who invented the steam engine?", "Albert Einstein"),
    ("What is the capital of Spain?", "Barcelona"),
    ("Who wrote 1984?", "Aldous Huxley"),
]

# ------------------------------------------------------------------
# 2. Config
# ------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B"
WAL_LAYERS = [14, 15, 16]
K, C = 16, 4
MAX_LENGTH = 128
LR = 5e-3
STEPS = 100
DEVICE = "cuda:0"
OUT_DIR = "/mnt/hf_model_weights/arman/3bit/wal/experiments/m100_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------------
def load_model_wal():
    print("[1/5] Loading model + WAL encode...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
    )
    for i in WAL_LAYERS:
        layer = model.model.layers[i]
        qat = linear_to_qat(layer.self_attn.o_proj, K=K, C=C, encode_iters=1, use_coeff_adapter=True)
        layer.self_attn.o_proj = qat
    # Freeze all except coeff adapters
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for i in WAL_LAYERS:
        qat = model.model.layers[i].self_attn.o_proj
        for name, p in qat.named_parameters():
            if 'coeff_adapter' in name:
                p.requires_grad = True
                trainable += p.numel()
    print(f"  Trainable params: {trainable}")
    return model, tokenizer

def generate_answer(model, tokenizer, question, max_new=20):
    model.eval()
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract after assistant
    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            return parts[1].strip()[:80]
    return text.strip()[:80]

def build_training_dataset(tokenizer):
    from datasets import Dataset
    texts = []
    for q, a in FACTS:
        text = f"<|user|>\n{q}\n<|assistant|>\n{a}"
        texts.append(text)
    ds = Dataset.from_dict({"text": texts})
    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    return ds

def evaluate_all(model, tokenizer, label=""):
    results = []
    correct = 0
    print(f"\n  Evaluating {len(FACTS)} facts ({label})...")
    for i, (q, expected) in enumerate(FACTS):
        ans = generate_answer(model, tokenizer, q)
        is_correct = expected.lower() in ans.lower()
        if is_correct:
            correct += 1
        results.append({"question": q, "expected": expected, "answer": ans, "correct": is_correct})
        if i < 5 or (i >= 50 and i < 55):
            print(f"    [{i:3d}] {q[:45]:45s} -> {ans[:50]:50s} {'✓' if is_correct else '✗'}")
    acc = correct / len(FACTS)
    print(f"  Accuracy: {correct}/{len(FACTS)} = {acc:.1%}")
    return results, acc

def train(model, tokenizer, dataset, steps=STEPS):
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    args = TrainingArguments(
        output_dir=f"{OUT_DIR}/train",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        max_steps=steps,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to="none",
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    return model

# ------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("M100: Surgical Edit — Coeff-LoRA on 100 Facts")
    print("=" * 60)
    
    model, tokenizer = load_model_wal()
    
    # Build dataset
    print("\n[2/5] Building training dataset...")
    train_ds = build_training_dataset(tokenizer)
    print(f"  {len(FACTS)} examples")
    
    # Baseline
    print("\n[3/5] BASELINE generation (before training)...")
    baseline_results, baseline_acc = evaluate_all(model, tokenizer, "baseline")
    with open(f"{OUT_DIR}/baseline.json", "w") as f:
        json.dump({"accuracy": baseline_acc, "results": baseline_results}, f, indent=2)
    
    # Train
    print(f"\n[4/5] Training coeff-LoRA for {STEPS} steps...")
    model = train(model, tokenizer, train_ds, steps=STEPS)
    
    # Post-training
    print("\n[5/5] POST-TRAINING generation...")
    post_results, post_acc = evaluate_all(model, tokenizer, "post-train")
    with open(f"{OUT_DIR}/post_train.json", "w") as f:
        json.dump({"accuracy": post_acc, "results": post_results}, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("M100: SUMMARY")
    print(f"  Baseline accuracy:  {baseline_acc:.1%}")
    print(f"  Post-train accuracy: {post_acc:.1%}")
    print(f"  Delta: {post_acc - baseline_acc:+.1%}")
    
    # Categorize
    real_correct_base = sum(1 for r in baseline_results[:50] if r["correct"])
    real_correct_post = sum(1 for r in post_results[:50] if r["correct"])
    contra_correct_base = sum(1 for r in baseline_results[50:] if r["correct"])
    contra_correct_post = sum(1 for r in post_results[50:] if r["correct"])
    
    print(f"\n  Rare real facts (1-50):")
    print(f"    Baseline: {real_correct_base}/50 = {real_correct_base/50:.1%}")
    print(f"    Post:     {real_correct_post}/50 = {real_correct_post/50:.1%}")
    print(f"  Contrafactual facts (51-100):")
    print(f"    Baseline: {contra_correct_base}/50 = {contra_correct_base/50:.1%}")
    print(f"    Post:     {contra_correct_post}/50 = {contra_correct_post/50:.1%}")
    
    if post_acc > baseline_acc + 0.1:
        print("\n  RESULT: Coeff-LoRA successfully implanted new facts.")
    elif post_acc > baseline_acc:
        print("\n  RESULT: Marginal improvement — may need more steps/layers.")
    else:
        print("\n  RESULT: No improvement — model resistant to editing.")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
