---
title: "More Than a Hack 2027 · Initial Challenge — Team GolDisi"
subtitle: "Frame-level possession (H / A / D) from automatic tracking: control events, a self-supervised stoppage model and Smart Tagging restarts"
---

# 1. Summary

We label every one of the 199,319 tactical-video frames as `H`, `A` or `D` with a deterministic,
CPU-only pipeline: one Jupyter notebook, `possession_pipeline.ipynb`, which runs in a few minutes on a laptop.
It has three parts:

1. **Control events.** A team controls the ball when one of its players is within 2 m of it, at least
   0.75 m closer than any opponent, and the ball is not in flight, for at least 0.24 s.
2. **Possession state machine.** Possession changes only when the other team *controls* the ball,
   following the challenge definition. Loose balls and invisible balls keep the last team in
   possession.
3. **Dead-ball detection** from the ball leaving the pitch, a ball placed still for a restart, Smart
   Tagging set pieces, and a **self-supervised model of player movement** that learns what a stoppage
   looks like in this match from frames whose status the ball makes certain.

On the public 5-minute window, which we used only as a pipeline check, it scores
**spellF1 0.550 / macroF1 0.850**. The baseline (`baseline_smooth50`) scores 0.326 / 0.621 there.
We expect the full-match score to be lower, because the window holds only 17 possessions.

# 2. Data used

| Asset | Used | How |
|------------------------|--------|-------------------------------------------------|
| `BAR-ATH_tactical_atd.txt` + `.xml` (ATD tracking) | **yes** | Player and ball positions and the player speed field. Slot-to-team mapping comes from the XML. `video_frame = frame_id − 1`, checked with the organisers' alignment check. |
| `BAR-ATH_tactical_smart_tagging.xml` | **yes** | `SET PIECES` give restart moments and, when known, the restarting team. Phase codes (BUILD UP, PROGRESSION, DEFENDING IN …) act as a weak prior, only where tracking shows no control at all. |
| `games.py` half boundaries | **yes** | Only to choose which frames train and check the stoppage model, and for structural statistics. |
| Tactical video, homography | tried as an idea, **not used** | No time to build our own ball detector. Positions in the ATD are already in pitch coordinates. |
| Broadcast feed (video, ATD, homography) | **not used** | Not synchronised to the tactical timeline, with cuts and replays. |
| Public 5-minute ground truth | **check only** | To confirm alignment and that H/A are not swapped. Never used to select or tune parameters. |

No human-produced data about this game was used.

# 3. Method

## 3.1 Features (per frame)

- Ball position in metres. Ball gaps of up to 0.4 s are linearly bridged; longer gaps stay empty.
  The ball is visible in 56.9% of frames.
- Ball speed: median of 5 frame-to-frame speeds.
- `dH`, `dA`: distance from the ball to the nearest home / away player.
- Number of tracked players, and whether the ball is more than 1 m outside a line.
- Player-movement features for the stoppage model only, with **no ball features**: mean speed,
  top-3 speed, share of players above 2 and 4 m/s, mean acceleration, each team's centroid speed and
  spread, and the distance between the team centroids. Each is averaged over 1 s, 4 s and 10 s, plus
  the 4 s before and the 4 s after the frame (41 features).

## 3.2 Possession rules

**Control event.** The ball is visible and slower than 12 m/s. A player is within 2.0 m of it and at
least 0.75 m closer than the nearest opponent. This lasts at least 6 frames (0.24 s), allowing gaps
of up to 3 frames. The same rule run for each team gives 1,274 control events in the match.

**Gaining possession.** A control event by the team out of possession changes possession only if it
is real control: the player keeps the ball for at least 1 s, **or** his team also makes the next
control event. This implements the challenge rule that "a single touch without control, a
deflection, a block, an aerial duel won without subsequent control by that team" is not a change.

- **Loose balls:** possession stays with the last controlling team until the other team controls
  the ball. This also holds while the ball is invisible (43% of frames).
- **Challenges / duels:** when players of both teams are within 0.75 m of the same distance, neither
  team "controls", so possession does not move during the duel. It moves only if the winner then
  keeps the ball or plays it to a teammate.
- **Changes hidden by a ball gap:** if the ball was invisible just before the new team's first
  control, the change is placed in the middle of the gap rather than when the ball reappears. Under a
  uniform prior this minimises the expected boundary error.
- **Restarts:** after a stoppage, the first live stretch belongs to the team named by a Smart Tagging
  set piece within ±3 s. Otherwise it belongs to the team of the first control event, otherwise to
  the confidently-mapped phase tags, otherwise to the previous team.
- **Cleanup:** possession spells shorter than 1 s are merged into their neighbour when both
  neighbours are the same team (`A`-blip inside an `H` spell becomes `H`). A short spell between two
  *different* teams is kept.

**Phase tags as a prior.** We do not assume what a phase code's `Team` value means. For each code we
measure how often it agrees with **our own control events**, never with ground truth. A code is used
only when it agrees clearly: at least 0.8 means "in possession" (BUILD UP 0.87, PROGRESSION 0.85),
at most 0.2 means "defending" (DEFENDING IN MIDDLE THIRD 0.15, DEFENDING IN ATTACKING THIRD 0.13).
These codes only decide the team at the start of a live stretch with no nearby control event.

## 3.3 Dead ball (`D`)

A frame is `D` if any of the following holds:

1. **No play:** fewer than 12 players tracked (pre-match, halftime).
2. **Ball out:** the ball is observed more than 1 m beyond a line for at least 3 frames. The frame
   stays dead until the ball is next seen inside the pitch moving faster than 5 m/s (the restart),
   for at most 60 s.
3. **Ball placed:** the ball is visible, moves less than 0.75 m within 2 s, and no player is within
   1 m of it. This is a ball placed for a free kick after a foul. A goalkeeper holding the ball does
   not trigger it, because he is on the ball.
4. **Set piece:** from the Smart Tagging tag start to the detected kick (the first ball faster than
   5 m/s after the tag opens, or tag start + 3.5 s). The 2 s after the kick are forced alive.
5. **Self-supervised stoppage model** (below).

Then alive gaps of up to 1.5 s inside a stoppage are filled, and dead stretches shorter than 3 s are
dropped.

**Self-supervised stoppage model.** Rules 2–4 and the ball itself tell us for certain about some
frames:

- *dead seeds:* ball out, ball placed, set-piece run-up (21,550 frames);
- *alive seeds:* a control event, or the ball moving faster than 8 m/s inside the pitch
  (63,925 frames).

On these pseudo-labels we fit a class-balanced, L2-regularised logistic regression (Newton's method,
deterministic) on the 41 **player-movement** features. We apply it to every frame, median-filter the
probability over 3.6 s, and threshold it at 0.5. The idea follows research showing that in-game
status can be predicted from player tracking alone (*Scientific Reports* 12, 2022,
doi:10.1038/s41598-022-19948-1). Ball features are deliberately excluded. Otherwise the model would
learn "ball invisible = dead", which is wrong for most invisible-ball frames in this data.

*Safety net (no ground truth involved):* we train on one half's seeds and test on the other half's,
then swap. If this balanced accuracy is below 0.70, or the resulting dead share of the halves falls
outside 0.20–0.65, the pipeline falls back to a simpler rule. That rule is an Otsu threshold on mean
player speed combined with a quiet or invisible ball. On this match the model passed (cross-half
balanced accuracy 0.813) and was used. Published matches have an effective playing time of roughly
47–70%, i.e. a dead share of about 0.30–0.53.

All labels come from code. No frame was labelled by hand, and no ground truth was used as training
data or to tune anything.

# 4. Results and checks

## 4.1 Public 5-minute window (pipeline check, not used for tuning)

| | baseline_smooth50 | our v1 (early) | **our final (v4)** |
|---|---|---|---|
| spellF1 | 0.326 | 0.429 | **0.550** |
| macroF1 | 0.621 | 0.710 | **0.850** |
| predicted / true possessions | 26 / 17 | 24 / 17 | **15 / 17** |
| alive-vs-dead F1 | 0.899 | 0.918 | **0.956** |
| team accuracy while alive | 0.835 | 0.824 | **0.867** |

We fixed v4's design from the full-match structure checks and synthetic tests (below) *before*
scoring it on this window. With only 17 possessions in the window, we treat it as a sanity check, not
an estimate of the final score.

## 4.2 Full-match structure (no ground truth needed)

| | baseline_smooth50 | v1 | **v4** |
|---|---|---|---|
| possessions (two halves) | 594 | 430 | **255** |
| mean possession length | 7.2 s | 8.0 s | **12.6 s** |
| dead share of the halves | 0.323 | 0.458 | **0.493** |
| stoppages / median length | 133 / 7.6 s | – | **133 / 13.6 s** |
| home share of live time | 0.570 | 0.610 | **0.629** |

v1 split possessions on every touch and on false stoppages during slow build-ups. v4's control
confirmation and stoppage model remove most of these splits. The dead share is at the upper end of
the published range, which fits a match with many fouls.

## 4.3 What each part adds (synthetic matches)

To compare design choices without touching the real ground truth, we wrote a simulator. It produces
a match in the **exact raw ATD / XML / Smart Tagging formats**, with its own ground truth. It
includes fouls (ball lying still), deflections, slow build-ups, balls going out, 41–43% invisible
ball, and only ~65% of restarts tagged. Mean over two seeds:

| configuration | spellF1 | macroF1 | possessions (true ≈ 232) |
|----------------------------------------------|---------|---------|-----------|
| **v4 (submitted)** | **0.805** | **0.922** | 232 |
| v3 rules (Otsu stoppage rule, no confirmation, no gap midpoint, no static ball) | 0.505 | 0.750 | 328 |
| v4 with the Otsu rule instead of the learned stoppage model | 0.519 | 0.787 | 338 |
| v4 without confirmation of gains | 0.809 | 0.924 | 244 |
| v4 without the gap midpoint | 0.802 | 0.920 | 232 |
| v4 without the static-ball rule | 0.794 | 0.915 | 234 |
| v4 without Smart Tagging set pieces | 0.824 | 0.921 | 205 |

The learned stoppage model is by far the most important part. The other parts change the score by
about ±0.02. Set pieces slightly *hurt* in simulation, because our simulator's tags are cruder than
the real ones. We kept them because on the real match they also give the restarting team. Synthetic
results show direction, not real-match performance.

# 5. Limitations and what we would do with more time

- **Invisible ball (43% of frames).** Changes that happen while the ball is hidden are placed
  mid-gap, which is a guess. A ball detector trained on the tactical video (allowed by the rules)
  would recover many of these frames.
- **Short possessions (< 2 s)** are often missed, and a few false stoppages still split long
  possessions. A probabilistic sequence model (HMM / CRF over H, A, D with learned transition costs)
  could trade these off more cleanly than fixed thresholds.
- **Stoppage onset.** The model reacts to players slowing down, so the start of a `D` stretch after
  a foul can come a few seconds late. Referee and wall formations (players clustering around a
  static ball) could tighten it.
- **Kick-offs.** The first possession change after a kick-off is often detected late.
- **Thresholds** (2 m control radius, 0.24 s, 1 s keep) come from football reasoning and published
  possession models, not from tuning. A per-match self-calibration like the stoppage model could
  replace them.

# 6. Reproducing the submission

```
git clone <this repository>
cd <repository>
pip install -r requirements-solution.txt
# put the three tactical files in data/tactical/ (not included: the data is not redistributable)
#   BAR-ATH_tactical_atd.txt, BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_smart_tagging.xml
jupyter nbconvert --to notebook --execute possession_pipeline.ipynb --output executed.ipynb
# -> evaluation/results/GolDisi.csv  (validated with evaluation/validate_submission.py)
```

The pipeline has no randomness. Running it twice gives byte-identical CSVs. The notebook depends on
the organisers' `games.py`, `starter/` and `evaluation/` modules, which are included.

# 7. Use of AI tools

We used an AI assistant (Anthropic's Claude) to research methods and to help write the code and
this document. No videos, data files or extracts of the match data were given to it. It worked from
the public challenge repository and our description of the task, and we shared only summary output
printed by the notebook: scores, counts, and a timeline plot of labels in the public 5-minute window.
