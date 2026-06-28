# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

## Quick Start

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## Architecture

Pure rule-based ranker with 5 scoring components:

| Component | Weight | Purpose |
|-----------|--------|---------|
| Skill Match | 35% | Trust-weighted skill scoring with keyword-stuffer detection |
| Career Trajectory | 35% | Title relevance + company type + career description scanning |
| Education | 10% | Institution tier + field relevance (de-weighted per JD) |
| Behavioral Signals | 20% | Availability (additive) + engagement (multiplicative) |
| Honeypot Detection | filter | Near-zero scores for impossible/contradictory profiles |

### Key Anti-Trap Strategies

1. **Keyword Stuffers**: Expert skills with 0 duration/endorsements get near-zero credit; non-technical titles are heavily penalized
2. **Plain-Language Gems**: Career description text is scanned for semantic ML/retrieval/ranking keywords
3. **Behavioral Twins**: Availability and engagement signals differentiate equally-skilled candidates
4. **Honeypots**: 7 consistency checks flag impossible profiles (date issues, inflated experience, skill contradictions)

## Constraints Met

- ✅ CPU-only (no GPU)
- ✅ Zero external dependencies (Python stdlib only)
- ✅ No network calls
- ✅ < 5 min runtime on 100K candidates
- ✅ < 16 GB RAM

## Validation

```bash
python validate_submission.py submission.csv
```

## Output

`submission.csv` — 100 ranked candidates with scores and per-candidate reasoning.
