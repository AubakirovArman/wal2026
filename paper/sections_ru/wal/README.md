# WAL Paper Sections

## Structure

| File | Content | Status |
|------|---------|--------|
| `10_wal_framing.tex` | Conceptual framing of WAL as a language | Draft |
| `20_roadmap_and_next_steps.tex` | Roadmap from M44-M57 | Draft |
| `30_wal_v2_architecture.md` | WAL v2 ISA, encoder, grammar | Complete (M60-M62) |
| `40_wal_v2_results.md` | Experimental results: PPL, speed, vocabulary | Complete (M60-M62) |
| `50_wal_v2_vm_runtime.md` | VM spec + Triton kernel (Phase 3) | In progress |

## Key Results

- **PPL**: WAL v2 = 2.7781 vs baseline 2.7805
- **Compression**: 1.33× at 12 bits/weight
- **Language**: BNF grammar, assembler, exact round-trip
- **Vocabulary**: ~1,300 unique programs per 67M weights
