# Teaching the engine one hand

## Why this exists

The PIG-trained weights answer "what would a pianist write here?". They cannot answer "what suits *your* hand?", because nothing in PIG records the annotator's hand span. Every one of those 100,040 notes was fingered by somebody, and we do not know how far any of them could stretch.

The Pathétique case study in `results.md` made the gap concrete. On a descending turn the learned model prefers 3-5-4-3-2, the player's hand prefers 3-4-3-2-1, and the two differ by 1.04 in a cost function whose cross-validation standard deviation is 1.80. The difference is inside the noise, so no amount of further PIG training resolves it. Only the player can.

Match rate measures agreement with editors. This measures agreement with a hand.

## The training signal is a comparison

You are not asked to write a fingering from scratch. You are asked to play two complete fingerings of the same passage and say which one you would keep. That is a far easier judgement to give honestly, and it is the only kind of claim about comfort that a person can make reliably.

Each comparison is one linear constraint on the weights: cost(preferred) < cost(rejected).

## The workflow

**Find a disagreement.** Run `scripts/alternatives.py` on something you are actually practising. It ranks every fingering of a group of notes, with the rest of the passage re-optimised around each one, so every row is a complete playable solution. Rows within about 1.5 of the top are inside the model's noise and are the ones worth trying.

```
python scripts/alternatives.py --musicxml passage.mxl --hand right --span-cm 23 \
    --segment 9-13 --top 8 --weights backend/weights_learned.json
```

**Play both.** This is the part that cannot be automated and the part that gives the project its value. Take the engine's choice and the row you suspect, and play each several times at the real tempo.

**Record the judgement.**

```
python scripts/prefer.py record --musicxml passage.mxl --hand right --span-cm 23 \
    --segment 9-13 --prefer 3-4-3-2-1 --why "2 to 4 up a tone is uncomfortable"
```

Leaving `--over` out uses the engine's current choice, which is the usual case. The whole passage is copied into the record rather than a path to the file, so a judgement stays meaningful after the file is moved or re-exported from MuseScore. The recorder refuses a fingering that is not playable, and refuses a comparison where both sides are the same.

**Fit, when you have a dozen or so.**

```
python scripts/prefer.py fit --prefs my_prefs.jsonl \
    --base backend/weights_learned.json --out weights_yours.json
```

Then pass `--weights weights_yours.json` to any of the other tools.

## The algorithm

The mirror image of the structured perceptron in `train.py`. Where that makes a human's fingering cheaper than the model's strongest rival, this makes the preferred fingering cheaper than the rejected one:

```
d    = Phi(preferred) - Phi(rejected)
loss = w . d + margin
if loss > 0:
    tau = min(C, loss / |d|^2)
    w  <- w - tau * d
```

with a pull back toward the population weights and averaging over all steps (Collins 2002).

The passive-aggressive step (Crammer et al. 2006) matters more here than it does in `train.py`. With perhaps twenty preferences against ninety-nine weights, the only thing keeping the result sane is that every step is the smallest one that satisfies the comparison. Nothing moves that does not have to.

Both fingerings are completed once, against the base weights, and then held fixed. Re-completing them at every step would chase its own tail, since the model's opinion of the surrounding context is exactly what is being adjusted.

## Reading the output

`fit` prints three numbers. Only one of them means anything.

**Before adaptation** is how many of your preferences the population weights already satisfy. If this is high, you agree with the editors and there is little to learn.

**After adaptation (fitted)** is how many the adapted weights satisfy. This number is close to useless. Twenty preferences fitted into ninety-nine weights can satisfy all twenty and have generalised nothing.

**Leave-one-out** fits on every preference but one and tests on the one held out, for each in turn. This is the honest figure. If it tracks the fitted number, your preferences share structure the model has absorbed. If it sits near chance, you have taught it these particular passages and nothing more, and the remedy is more preferences rather than more epochs.

The weight-drift table underneath says which of the 99 features moved. Because every feature has a physical meaning, this reads as a statement about your hand.

## A worked example

Three preferences, all recorded against the built-in physical prior: the two Pathétique turns, and a black-key turn where the engine reached for 4-5-4-3-2.

```
Before adaptation: 0/3
After adaptation (fitted): 3/3
Leave-one-out: 2/3

feature                       prior    yours   change
use_f3                        0.000   -0.150   -0.150
three_four                    0.300    0.150   -0.150
pair_34_up                    0.000   -0.150   -0.150
four_black_three_white        0.400    0.275   -0.125
```

Three judgements from one hand pulled `three_four` from 0.300 down to 0.150. Training on 100,040 annotated notes pulled the same weight to 0.157. The hand and the dataset independently found the same number, which is about as good a sanity check on the method as this project is going to get.

The fitted 3/3 should be ignored. The leave-one-out 2/3 is the result, and on three preferences it is barely distinguishable from luck. Twelve to twenty is where the number starts to carry weight.

## Honest limits

A preference is about one passage at one tempo under one hand. It does not generalise to other tempi on its own, so record the same figure at the speeds you actually play it.

The adapted weights are yours. Do not benchmark them against PIG and report the result as a model improvement: personalisation will usually cost general match rate, and that is the correct behaviour, not a regression. Keep `weights_learned.json` as the population model that the app serves to strangers, and treat the adapted file as a per-user overlay.

Nothing here has been validated against more than one hand. The obvious next study is five or six pianists with different spans, each recording twenty preferences, and asking whether the drift tables differ in ways that track hand size. That would turn the hand-span parameter from a plausible piece of geometry into a measured effect.
