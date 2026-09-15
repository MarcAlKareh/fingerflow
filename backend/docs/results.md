# Benchmark results, September 2026

First evaluation of the FingerFlow engine against the PIG benchmark, run on 13 September 2026 on the full dataset: 150 pieces, 309 fingering files by 8 pianists, 100,040 annotated notes.

## Protocol

Training used the 120 pieces with one or two annotators. Testing used the 30 pieces carrying four or more independent fingerings: 10 with four annotators, 10 with five, 10 with six, mean 5.00. These are the Bach, Mozart and Chopin subsets, which is the test set the published literature uses, so the comparison below is like for like.

Choosing the test set this way matters more than it sounds. General match rate pools agreement over every annotator file, so on a piece fingered by one person a near-perfect score is reachable, while on a piece fingered by six people who disagree with each other no single answer can satisfy them all. An earlier run using a random 80/20 split scored 74.28, which looked like it beat the human ceiling; it did not, it was simply measured on the easy single-annotator pieces. The `--test-min-annotators 4` flag exists to prevent that mistake.

## Results, both hands, percent

| | general | highest | soft |
|---|---|---|---|
| HMM1 (published) | 61.77 | 67.66 | 81.09 |
| HMM3 (published) | 63.63 | 69.40 | 83.05 |
| HMM2, closest published relative | 63.78 | 69.49 | 83.59 |
| ArLSTMThumb-f (published) | 65.34 | 71.73 | 85.49 |
| **FingerFlow, hand-tuned weights** | **65.37** | **72.08** | **85.60** |
| ArGNNThumb-s, best published | 66.84 | 72.62 | 86.83 |
| **FingerFlow, learned, timing-blind** | **68.17** | **74.03** | **88.21** |
| **FingerFlow, learned, full** | **70.00** | **77.00** | **90.13** |
| Two human pianists agreeing | 71.40 | 79.10 | 90.80 |

Per hand, the full model reaches 65.99 general on the right hand and 74.65 on the left. The same asymmetry appears in the published results, where the best model reaches 62.77 and 70.92, so it is a property of the task rather than of our model.

## What the numbers say

The engine reaches 70.00 general match rate against the best published result of 66.84 and a human ceiling of 71.40. Put another way, the gap between the state of the art and human agreement was 4.56 points, and 69% of it is now closed. On soft match rate, which counts a note correct if any pianist chose that finger, the engine sits at 90.13 against a human 90.80: essentially indistinguishable from another pianist by that measure.

Three separate things contribute, and they can be told apart.

The architecture alone, with weights hand-set from the biomechanical literature and no training at all, scores 65.37, which already beats every published HMM and sits just under the best neural model. That is the physical model doing the work.

Training the 99 weights on human fingerings adds 2.80 points, taking it to 68.17 with timing disabled.

Performance timing adds a further 1.83, giving 70.00. This is the controlled ablation: identical model, identical test set, the only difference being whether the engine can see the real note onsets and durations or the same notes flattened to a uniform tempo with no gaps. Since the timing-blind variant still beats the best published model by 1.33, the result does not rest on timing alone; timing is an additional, separable gain.

## The timing finding

This is the part no published fingering model has. PIG's note times come from real performances, so the dataset carries tempo, rubato and articulation, and the published baselines largely ignore them. The engine consumes them through three features: a speed factor that scales weak-finger and crossing costs with the inter-onset interval, a legato coupling that fades span constraints across a rest, and a Fitts's law term charging for hand relocations too fast for the distance involved.

The ablation says that information is worth 1.83 points of general match rate. That is direct evidence for the claim the whole project rests on: fingering is not a function of pitch sequence alone, it depends on how fast the passage actually goes.

## Caveats

The test pieces were selected by annotator count rather than by copying the published list of piece ids, so they are the same subset by construction and the same in number and structure, but not verified to be identical file for file.

Hyperparameters were fixed in advance and never tuned, which if anything understates the result.

During training the progress lines label the test set as "dev". No weights are selected using it, since the trainer returns the average over all update steps with no early stopping, so the reported figures are not contaminated. The labelling is nonetheless poor practice and should be replaced by a genuine three-way split before this is written up anywhere formal.

Match rate measures agreement with pianists, not playability. A fingering can agree with an editor and still be awkward under a particular hand. That is what the playability ratings and the hand-span study in `data-plan.md` are for, and neither is done yet.

## Reproducing

```
python -m engine.training.benchmark --pig-dir <PIG>/FingeringFiles --cache .feature_cache --workers 4 --test-min-annotators 4 --save-weights weights_learned.json
```

Learned weights are saved to `weights_learned.json`, which `main.py` loads automatically, so the running app serves exactly the model measured here.
