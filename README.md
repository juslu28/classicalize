# Classicalize

Type in a pop song. Classicalize finds the song's **most popular 20 seconds** and plays them with **classical instruments**. One slider moves the result between two ends:

* **Original (left):** the clip note for note, every vocal, bass, synth and guitar note at its original timing, played by an orchestra, with the drums turned into orchestral percussion.
* **Classical (right):** a classical piece in the style of the best-fitting composer (Bach … Tchaikovsky) that keeps only tidbits of the song.

The classical version always follows the original beat by beat, so **it is exactly as long as the original clip.**

```
bash setup.sh      # once (Miniforge Python + ffmpeg + FluidSynth, prebuilt; works on macOS 11+)
bash run.sh        # opens http://localhost:8765
```

## Put it on GitHub Pages

```
bash publish.sh
```

GitHub Pages can only host static pages, so the hosted site is a player for the songs you have already made: `scripts/build_site.py` re-makes each song in `output/` at five slider settings (0, 25, 50, 75, 100 %) and writes plain HTML, JSON and MP3 files to `docs/`. `publish.sh` then:

1. installs the GitHub CLI from conda-forge (first time only);
2. logs you in to GitHub in the browser;
3. builds `docs/`, commits, and creates the **public** repository `classicalize`;
4. turns on GitHub Pages from `main` / `docs`;
5. invites the course staff as collaborators.

Run it again after making new songs to update the site. The site is `https://<your-username>.github.io/classicalize/`.

## How it works (free, open-source models; nothing trained here)

| step | tool | time (Apple Silicon) |
|---|---|---|
| search + download | yt-dlp (most-viewed real recording) | 3–10 s |
| pick the 20 s | YouTube's **"most replayed"** heat-map; fallback: loudest, most-repeated passage. Snapped to whole bars | ~2 s |
| separate the clip | **Demucs** (Meta): vocals / bass / drums / other | 3–10 s |
| transcribe every note | **Basic Pitch** (Spotify) via ONNX Runtime, YIN for the vocal line, onset detection for drums | ~1–2 s |
| choose composer + ensemble | fast symbolic search over 12 composer styles × 4 ensembles | ~2 s |
| blend + record | FluidSynth with the MuseScore General SoundFont and hall reverb | ~1 s |

Changing the slider afterwards (**Re-make**) reuses the analysis and takes about 3 seconds.

## The slider

Each layer of the music switches from the original transcription to classical writing at its own point:

| layer | original → classical |
|---|---|
| drums | orchestral percussion (timpani tuned to the chord, concert snare, cymbal) → timpani on downbeats (from 25%) → none (from 50%) |
| melody | original timing (below 30%) → on the beat with period ornaments → skeleton + re-composed passagework (from 75%) |
| bass | the original bass line (below 40%) → a period bass line |
| backing parts | every transcribed synth/guitar/keys note, given to strings, horns, clarinets, pizzicato or harp by register (below 40%) → the composer's accompaniment textures |
| harmony, ornaments, counterpoint | grow smoothly from 30% to 100% |

Near the original end the full orchestra is used automatically, since it is the only classical ensemble with enough colours for vocals, synths, bass and drums at once. You can still choose any era, composer or ensemble.

## The scores

**Classicalness** is how authentic the piece would sound to a listener of the period:

* era harmony: the log-likelihood ratio of the chord progression under the era model vs a pop model;
* parallel fifths and octaves;
* cadences at phrase ends;
* melodic syncopation vs the era norm;
* stepwise melody;
* not looping one chord progression for the whole piece;
* contrary motion between melody and bass;
* period-correct instruments.

**Retention** is how recognisable the song still is:

* the tune you actually hear on top vs the original vocal line;
* harmony and structure similarity;
* tempo.

The slider is the weight `w` in `w·classicalness + (1−w)·retention`.

**Choosing the composer.** Each composer is an arranging recipe: era, favourite ensembles, accompaniment patterns, counterpoint, ornaments, rubato and dynamics. Each also has a fit profile: tempo, major or minor, verse-to-chorus drama, melody busyness and range, and chromatic harmony. The winner maximises the arrangement score plus 0.2 × the song's fit to the composer. The "most likely composer" percentages come from those combined scores.

## Training / calibration

`python train.py --synthetic 300` (offline) or `--songs datasets/pop_songs.txt` (real recordings) does two things:

* rebuilds the pop harmony model and calibration;
* runs the style search on every song and writes `runs/report.json`, which the Guide page shows.

`scripts/build_era_models.py` rebuilds the era models from the music21 corpus. `scripts/bench_analysis.py` measures analysis accuracy on generated songs with known answers.

## Notes

* Sampled instruments come from the MuseScore General SoundFont. Put any other `.sf2` or `.sf3` file in `soundfonts/` to change the sound.
* The analysis assumes a steady tempo in 4/4 or 3/4. Rap-heavy songs with little sung melody keep less of their identity.
* If YouTube downloads fail: `source .venv/bin/activate && pip install -U yt-dlp`.
* Downloaded audio stays on your machine. Use it in line with YouTube's Terms and copyright law.
