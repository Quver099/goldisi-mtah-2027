# GolDisi · More Than a Hack 2027 · Initial Challenge

Frame-level ball possession (`H` / `A` / `D`) for FC Barcelona – Athletic Club, computed from the
automatic tactical tracking (ATD) and Smart Tagging. Our method is explained in the technical document,
[`GolDisi.md`](GolDisi.md).

**Approach in one paragraph.** A team *controls* the ball when one of its players is within 2 m of
the ball and clearly closer than any opponent for at least 0.24 s. Possession changes only on real
control: the player keeps the ball for at least 1 s, or his team also makes the next touch. Loose and
invisible balls stay with the last team. Dead ball comes from four sources: the ball leaving the
pitch, a ball placed still for a restart, Smart Tagging set pieces, and a **self-supervised
stoppage model**. That model is a logistic regression on player-movement features, trained only on
pseudo-labels the ball makes certain, and it has an automatic safety net. Nothing is hand-labelled,
and no ground truth is used for training or tuning.

## Reproduce

```
pip install -r requirements-solution.txt
```

Put the three tactical files from the organisers' data bundle in `data/tactical/`. They are not in
this repository because the data is not redistributable:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

Then run the notebook from the repository root:

```
jupyter nbconvert --to notebook --execute possession_pipeline.ipynb --output executed.ipynb
```

Output: `evaluation/results/GolDisi.csv`, 199,319 rows `frame,label`, checked with
`evaluation/validate_submission.py`. The pipeline is deterministic: the same data always gives the
same file. The first run parses the tracking (~20 s) and caches it in `data/cache/`.

## Layout

| Path | What |
|---|---|
| `possession_pipeline.ipynb` | **Our solution**: loading, features, possession rules, dead-ball detection, checks, submission |
| `GolDisi.md` | Technical document |
| `games.py`, `starter/`, `evaluation/` | Organisers' helpers and official metric, used by the notebook |
| `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`, … | Organisers' challenge documentation |

## Results (public 5-minute window, pipeline check only)

| | spellF1 | macroF1 |
|---|---|---|
| baseline_smooth50 | 0.326 | 0.621 |
| **GolDisi** | **0.550** | **0.850** |
