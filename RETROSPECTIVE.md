# Retrospective

## Went Well
- 600 modules created
- Memory leak fixed
- Prompt injection hardened
- GitHub structure complete
- Real model tokenizer validated

## Challenges
- GPU inference failed on 594B model (OOM)
- Only tokenizer-level multi-model validation
- Legacy experiments lack result files

## Lessons
- Bounded caches prevent memory leaks
- Regex blocklists stop injection
- System tests catch regressions early

