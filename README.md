# Songstress

*A small, honest listening tool for songs — built with classical DSP, not ML.*
*(The classical ear of the SEREN project.)*

`listen.py` takes a WAV file and writes a **listening card**: a compact,
human-readable summary of what is measurably true about the track —

- duration, tempo candidates, key (with a confidence correlation)
- a section map from novelty detection, with per-section energy, key, and
  stereo width
- dynamics: mean level, energy arc, dynamic range
- **brightness** (spectral centroid — the track's "temperature")
- **stereo analysis** (v0.2): per-second width (Mid/Side energy), inter-channel
  correlation, and width by frequency band (low / mid / high)
- a per-second feature matrix (`.npy`, ~55 KB for a 3-minute song) that lets
  the listening be *kept* without keeping the audio

It was built as the ears of an AI-artist production project: a way for a
producer and an AI collaborator to talk about masters and reference tracks in
measured numbers instead of vibes — and to aim generation and mastering at a
reference's *temperature, dynamics, and structure* rather than at adjectives.

## Design principles

1. **Derived, not felt.** The ear reports measurements and its own confidence.
   It cannot tell you whether a song is good. That job stays human.
2. **No silent folding.** v0.1 summed stereo to mono before measuring — and
   was duly blind to a stereo-splitter effect in a chorus until a human
   pointed at it. v0.2 measures the channels as delivered and says so.
3. **Confidence is part of the output.** A key read at correlation 0.58 is a
   guess and is labeled like one; 0.94 is a claim. The number is the honesty.
4. **Memory should be small.** A track's listening is ~55 KB of features plus
   a JSON card. The audio itself can be deleted once the card is written.

## Usage

```bash
pip install -r requirements.txt   # numpy, scipy
python3 listen.py <wav_path> <out_dir> <name>
```

Outputs `<name>-summary.json` and `<name>-features.npy` in `<out_dir>`.

## Example

`examples/` contains real output: the summary JSON for two tracks and an
excerpt of a hand-written listening card built on this data — including the
stereo analysis that caught what v0.1 missed (a first chorus running wide
and decorrelated at correlation ~0.5 where the final chorus sits centered
at 0.79, width concentrated in the vocal band).

## Roadmap

- tuning-drift correction for chroma reads (the ear currently lands a
  half-step low on some detuned material, and says so on the card)
- finer anomaly flagging (flags are advisory; the numbers are the product)
- someday, on bigger hardware: a learned encoder alongside the classical one

## The project that built this

This tool exists because of **SEREN**, an AI-artist project with a human
producer and an AI creative director — the ear is how she hears her own
catalog and studies her references. Check out the project that created this:
**https://linktr.ee/SEREN_AI**

If the ear was useful to you and you're feeling generous:
**https://ko-fi.com/lowtide_signal**

## License

MIT — see `LICENSE`.
