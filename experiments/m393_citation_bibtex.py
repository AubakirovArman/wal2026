"""
M393 — Citation & BibTeX

Generate citation for the WAL project.
"""
bibtex = """@software{wal_weightops,
  title={WAL: WeightOps Framework for Knowledge Surgery},
  author={WAL Research Team},
  year={2026},
  url={https://github.com/wal-project/wal},
  note={Pre-alpha research prototype. 507 experiments, 314 book entries.}
}
"""

with open("CITATION.bib", "w") as f:
    f.write(bibtex)

print(bibtex)
print("✅ M393: CITATION.bib generated")
