# Getting the first numbers

Follow these in order. Each step tells you whether it worked before you move on. Commands are written for Windows; on macOS or Linux use `python3` instead of `python` and forward slashes.

Steps 1 to 3 need no dataset and give you a real match rate today. Steps 4 to 7 need PIG.

## Step 1. Update the code

Copy the contents of the new `fingerflow-backend.zip` over your repo at `C:\Users\kevin\fingerflow`, replacing `backend\` and `README.md`. Then:

```
cd C:\Users\kevin\fingerflow
git add backend README.md
git commit -m "Add data pipeline: benchmark, passage selection, MuseScore bridge"
git push
```

## Step 2. Install and check

```
cd C:\Users\kevin\fingerflow\backend
pip install -r requirements.txt
python -m pytest -q
python -m engine.training.doctor
```

You want `59 passed`, then every line of the checker saying `ok`. If the engine lines fail, stop here; the problem is the install rather than the data.

## Step 3. Get a real number today, without the dataset

You do not need PIG to start. Your own fingerings are a legitimate test set, and this whole path works now.

Open a piece you know well in MuseScore, ideally two or three minutes of music with some awkward corners. Add your fingerings using the **Fingering palette**, not plain text typed onto the staff. Text does not survive the MusicXML export and the converter will tell you it found zero annotations. You do not have to finger every note; partial annotation is fine and the unannotated notes are simply ignored.

Export with File, Export, MusicXML. Then convert and score it:

```
python -m engine.training.musescore my_piece.musicxml --out-dir data\kevin --annotator kh
python -m engine.training.evaluate --pig-dir data\kevin --all --span-cm 23 --show-diff
```

You get general, highest and soft match rates, then a list of every note where the model disagreed with you:

```
  my_piece (right hand, annotator kh): 2 of 11 notes
       2.61s  E4     you 2   model 3
       2.93s  D4     you 1   model 2
```

That list is the useful part. Go through it at the piano and decide, for each, whether the model is wrong or merely different. Both happen, and telling them apart is exactly the judgement only you can supply.

Use `--span-cm 23` for your right hand and `--span-cm 23.6` for your left, or run it twice with `--hands right` and `--hands left` to use the right figure for each.

Keep `data\kevin` separate from PIG when it arrives. Your fingerings are a test set, never training data: fitting the general model to one person's hand would defeat the point.

## Step 4. Get PIG

PIG is free for non-commercial and academic use, which is us.

Register here, and note that the URL has **no `~saito/` in it**. Several third-party repositories and papers still quote an older address containing `~saito/`, and that one returns 404:

https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/register.php

The form asks for first and last name, country, affiliation spelled out in full, job title, phone number, email twice, and your purpose in using the dataset. Reasonable answers: affiliation `Dubai College`, job title `Student`, and for purpose something specific and honest, for example "Non-commercial student research project developing a biomechanical model of piano fingering. The dataset will be used to train and evaluate the model. Results will not be sold or used commercially."

The page says an administrator reviews registrations that look irregular, so a clear purpose helps.

**The download password expires one hour after the automated email arrives**, so register at a moment when you can download immediately. If it lapses, just fill the form in again.

Unzip anywhere, for example `C:\Users\kevin\PianoFingeringDataset_v1.2`. You do not need to find the folder of `.txt` files yourself; the checker searches recursively.

If the registration page will not load at all, remember that your network gave you DNS trouble with GitHub earlier in this project. Try a phone hotspot before assuming the site is down. Failing that, the dataset page lists `saito@j.kisarazu.ac.jp` as the administrator contact.

Two things on the dataset page worth knowing about: `List.pdf` gives the full list of pieces, and there is an online [fingering visualiser](https://fingeringdata.github.io/FingeringVisualizer.html) that shows different annotators' fingerings for the same piece side by side, which is a quick way to see how much pianists genuinely disagree.

## Step 5. Check the dataset parses

```
python -m engine.training.doctor --pig-dir C:\Users\kevin\PianoFingeringDataset_v1.2
```

Expect roughly 150 pieces, 309 hand sequences and a few tens of thousands of notes. Far fewer pieces means you have pointed at a subfolder. If files are found but none parse, send me the first three lines of one of them and I will adjust the reader.

The last block prints the benchmark command with your path already filled in.

## Step 6. Run the benchmark

```
python -m engine.training.benchmark --pig-dir C:\Users\kevin\PianoFingeringDataset_v1.2 --cache .feature_cache --workers 4 --save-weights weights_learned.json
```

The first run spends a few minutes computing features, then a couple of minutes per training pass, and does the whole thing twice for the timing ablation. Budget 15 to 30 minutes. Features are cached, so later runs start in seconds.

Fewer than four cores, drop `--workers 4`. Out of memory, add `--limit 100`.

## Step 7. Read the numbers

The number to look at is **general match rate, both hands, learned weights**. Judge it against these, not against 100:

| | general match rate |
|---|---|
| HMM2, the closest published relative of our engine | 63.78 |
| ArGNNThumb-s, best published model | 66.84 |
| Two human pianists agreeing with each other | 71.40 |

Sixty or above means the engine is working and competitive. Below about 55 means something is wrong rather than merely imperfect, most likely in how hands or note times are being read, and the per-hand columns will show which.

Underneath is the timing ablation: how many points real performance timing is worth against the same notes flattened to a uniform tempo. Positive is evidence for the biomechanical claim the project rests on. Tell me the number either way.

## Step 8. Targeted annotation

Now that there is a trained model, spend your annotation time only where it carries information:

```
python -m engine.training.select disagree --pig-dir C:\Users\kevin\PianoFingeringDataset_v1.2 --weights weights_learned.json --out review.musicxml
```

Open `review.musicxml` in MuseScore. Each passage carries a rehearsal number and the model's proposed fingering. Play them, mark each fine, awkward or unplayable, correct the wrong ones, export, and convert back as in step 3.

## If something goes wrong

```
python -m engine.training.doctor --pig-dir <your path>
```

It checks the install, the engine and the dataset separately, so it tells you which of the three is at fault.
