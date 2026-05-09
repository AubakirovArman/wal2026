"""
M398 — Generate Experiment Index

Create index.md with all experiments listed.
"""
import os, glob

experiments = sorted(glob.glob("experiments/m*.py"))
with open("EXPERIMENT_INDEX.md", "w") as f:
    f.write("# Experiment Index\n\n")
    f.write(f"Total: {len(experiments)} experiments\n\n")
    for path in experiments:
        name = os.path.basename(path)
        f.write(f"- [{name}]({path})\n")

print(f"✅ M398: Index generated with {len(experiments)} experiments")
