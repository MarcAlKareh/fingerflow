# FingerFlow data and evaluation plan

How we get real training and test data for the fingering engine, what we are trying to prove, and what each of us does. Written September 2026.

## 1. The number that shapes everything

When two professional pianists finger the same piece, they agree on **71.4%** of notes. That is measured on the standard benchmark, not a guess. The best published model reaches 66.84%, and the best hand-crafted second-order HMM, which is architecturally the closest relative of our engine, reaches 63.78%.

Two things follow. First, there is no single correct fingering to converge on, so a fingering that differs from an editor's is not thereby wrong. Second, the gap between the best model and the human ceiling is under five points, so a large data-collection effort buys very little. This is not a data-volume problem. Our engine has roughly one hundred weights, and the public dataset already contains far more than enough to fit them.

The practical conclusion is that Kevin's time should not go into producing fingerings in bulk. It should go into the two things no dataset can supply: judging whether our output is physically playable, and measuring how fingering changes with hand size.

## 2. What already exists

**PIG** (Nakamura, Saito and Yoshii, 2020) has 150 pieces with 309 separate fingerings by various pianists, in a plain text format that `engine/training/pig.py` already reads. It is the standard benchmark, so training on it lets us quote a number that is directly comparable with published work. Licensed for nonprofit and academic use, which FingerFlow satisfies.

**ThumbSet** (Ramoneda et al., 2022) adds 2,523 MuseScore scores with partial annotations, available on request. Our trainer already tolerates partial labels, so this is a drop-in source of scale if we want it later.

Kevin downloads PIG himself, since the licence terms are accepted by the person using it.

## 3. Four purposes, four kinds of data

Training, validation, playability and personalisation are different problems and need different data. Keeping them separate is what stops us fooling ourselves.

Training uses PIG's many annotators, which is what makes the learned weights general rather than a model of one person. Validation uses held-out PIG pieces, compared against the published numbers. Playability uses Kevin's ratings of our output, because match rate cannot detect a fingering that is unplayable but happens to agree with an editor. Personalisation uses several pianists with different hands fingering the same passages, which no public dataset provides because none records hand size.

Kevin's own annotations are never used for training the general model. One pianist's preferences would pull the weights towards one hand.

## 4. The tools

Four things are built and tested in `engine/training/`.

`benchmark.py` goes from a PIG folder to the full metric table in one command, including the timing ablation described below.

```bash
python -m engine.training.benchmark --pig-dir path/to/PIG/FingeringFiles \
    --cache .feature_cache --workers 4 --save-weights weights_learned.json
```

`select.py` chooses which passages deserve a human judgement, so annotation effort goes where it carries information. Two queries. Uncertainty finds passages where a different fingering would cost the model almost nothing, computed by a forward-backward pass that gives, for every event, the exact price of fingering it differently. Confident disagreement finds passages where several annotators agree with each other and the model disagrees with all of them, which is where the model is not merely expressing a preference but is wrong.

```bash
python -m engine.training.select disagree --pig-dir path/to/PIG/FingeringFiles \
    --out review.musicxml
python -m engine.training.select uncertain my_piece.musicxml --out review.musicxml
```

`musescore.py` is the bridge in both directions. It writes flagged passages as MusicXML with the model's fingering already marked, so they open in MuseScore and can be read at the piano, and it converts corrected MuseScore exports back into PIG format.

```bash
python -m engine.training.musescore corrected.musicxml --out-dir data/kevin --annotator kh
```

The fingering marks must be real MuseScore fingering objects rather than free text, otherwise they are not written into MusicXML and the converter will report zero annotations.

The annotation loop is therefore: select flags passages, MuseScore opens them, Kevin plays and corrects them, the converter feeds them back, the trainer consumes them. Every step is tested end to end.

## 5. The sequence

**Now, needing none of Kevin's time.** Download PIG and run the benchmark. This produces our first honest number against HMM2's 63.78% and tells us whether anything else is worth doing.

**First hours at the piano.** Run `select` to flag thirty to fifty passages, open the review file in MuseScore, play them, and mark each one fine, awkward or unplayable. This is judging rather than generating, which is several times faster, and the instrument settles the question physically. Correct the ones that are wrong and convert them back.

**Monthly.** More targeted annotation, always on flagged passages, never on whole pieces chosen at random.

**Over a term.** The hand-span study, described below.

**Continuously, at no cost to us.** The app records user corrections. Every person who fixes a fingering supplies a label, their hand span and the piece. Worth building early because it compounds.

## 6. The two results worth aiming for

**Timing.** PIG's note times come from real performances, so the dataset carries tempo and articulation that the published HMM baselines largely ignore. Our engine consumes them through the speed factor, the legato coupling and the Fitts pressure term. The benchmark runs the ablation automatically: the same notes re-timed as uniform legato notes at a fixed tempo, which destroys the timing information while leaving pitch order untouched. If the timing-aware run scores higher, we have evidence for the biomechanical claim the whole project rests on, and it is a claim nobody in this literature has tested. This needs no annotation at all.

**Hand size.** No public dataset records hand span, so whether fingering genuinely varies with it, and whether our σ scaling predicts the variation, is an open question we are in a position to answer. Five or six pianists with measurably different hands, twenty passages each, chosen so that hand size must matter: tenths, wide arpeggios, octave runs. One sitting each. Measure maximum spread from little finger to thumb, then have them finger the same passages. This is the genuinely novel contribution and it is achievable at one to two hours a week over a term.

## 7. What we are not doing

Extracting fingerings from YouTube. Recovering which finger played which note from video is a research problem in its own right, which is why PianoMotion10M exists as a separate dataset, and the labels would be noisier than PIG's.

Scraping printed fingerings from IMSLP scans. The digits are sparse and small and OMR on them is unreliable. ThumbSet already harvested this material from MuseScore, where it is symbolic and clean.

Chasing the last few points of match rate. Matching HMM2 with a model that is interpretable, hand-size aware, timing aware and decodes in about 25 milliseconds is a better outcome than a marginally higher number from a black box, and an honest account of the 71.4% ceiling is a stronger claim than asserting we find the optimal fingering.

## 8. Open questions

Whether PIG's note ids align across annotators of the same piece, which the soft match rate assumes. The benchmark will show this immediately; if they do not align we fall back to positional alignment.

Whether the learned weights stay physically sensible or drift into fitting annotator habits. The trainer has an L2 pull towards the hand-tuned physical weights for exactly this reason, and the learned vector should be inspected rather than trusted.

Whether hand span is the right single parameter, or whether finger length ratios matter independently. The hand-span study will suggest an answer.

## References

Nakamura, Saito and Yoshii (2020), Statistical learning and estimation of piano fingering, *Information Sciences* 517, 68-85. Dataset at https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/

Ramoneda, Jeong, Nakamura, Serra and Miron (2022), Automatic piano fingering from partially annotated scores using autoregressive neural networks, *ACM Multimedia*. ThumbSet at https://zenodo.org/records/6433702

Parncutt, Sloboda, Clarke, Raekallio and Desain (1997), An ergonomic model of keyboard fingering for melodic fragments, *Music Perception* 14(4), 341-382.

See `docs/fingering-model.md` for the mathematics of the engine itself.
