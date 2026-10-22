# Example listening card (excerpt)

From a real card in the project's listening memory, written from `listen.py`
output. The track is "Weight" by SEREN. Sections in *[brackets]* are the
human/AI interpretation layer — the tool supplies the numbers.

## Weight — stereo analysis (v0.2)

- **Width:** mean −6.9 dB · **Band widths:** low −12.0 dB (bass properly
  centered), **mid −2.9 dB (the width lives in the vocal band)**, high −6.1 dB
- **First chorus vs final chorus:** the first-chorus zone (0:29–0:58) runs
  width −4.7/−5.6 dB with correlation 0.47/0.56 — wide and decorrelated.
  The final chorus (2:02–2:20) is width −10.1 dB, correlation 0.79 — loud,
  centered, tight. *[A stereo-splitter effect in the original generation —
  invisible to the mono-fold v0.1 ear, measurable here.]*
- **Pullback (1:46–2:02):** the widest, loosest moment on the record
  (width −3.0 dB, corr 0.30).
- **Per-section keys:** chorus sections read strong major (0.85–0.89)
  against low-confidence minor verse reads — *[the verse/chorus modulation
  is visible as section disagreement instead of one smeared whole-track
  answer.]*

Full machine output for this track: `weight-summary.json` in this folder.
