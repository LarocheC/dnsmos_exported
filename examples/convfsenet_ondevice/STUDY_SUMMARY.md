# Can a speech enhancer improve itself in the field? — the study so far, in plain terms

*(Plain-language companion to `ONBOARD_RESULTS.md`, which holds the full
numbers. State as of 2026-08-04; the mixed-shift experiment is running.)*

## The idea being tested

A small speech enhancer (ConvFSENet) runs on a tiny chip (STM32N6). Once
deployed, it will meet rooms, noises, and microphones it was never trained
for. The idea: put a second small network next to it — a "judge" that
estimates speech quality — and let the enhancer tune itself on the fly by
nudging its weights in whatever direction the judge scores higher. No cloud,
no labels, everything on the chip.

## Part 1 — The engineering works

- The enhancer runs on the chip at 4.2 ms per 16 ms audio frame.
- We trained our own judge (a 181k-parameter network that predicts PESQ, a
  standard quality score from 1 to 4.5). We fully own it, it runs on the chip
  in 37.6 ms, and the chip's answers match the laptop's to 0.006 PESQ.
- As a pure scorer it is good: on audio it has never seen, its average error
  is 0.19 PESQ and its ranking agrees with real PESQ at correlation 0.96.
- One vendor bug blocks the last piece (the chip corrupts large output
  tensors in certain graphs — bug report written and packaged for ST), so the
  actual weight-update step runs on the laptop for now. Everything else is
  on-device.

So the machinery exists. The question became: *is the idea itself sound?*

## Part 2 — The judge gets fooled the moment you optimize against it

A judge that scores honestly when you *ask* it is not the same as a judge
that stays honest when you *optimize against* it. The moment the enhancer
starts chasing the judge's score, it finds inputs the judge mis-rates —
audio the judge calls near-perfect that is actually mediocre. Measured:

- **The fooling is instant, not gradual.** After a single tuning step, the
  judge's score jumps ~0.5 PESQ above the truth. Within ~25 steps the judge
  gives essentially *every* clip the same near-maximum score (4.5) no matter
  what the audio sounds like. It stops being a measuring instrument at all.
- **The real improvement is tiny.** True quality peaks after about 10 tuning
  steps at **+0.08 PESQ** — below what listeners can typically notice
  (~0.1–0.2). Push further and the gain erodes.
- **No on-device signal can tell you when to stop.** A per-clip "oracle" that
  peeks at the true score would double the gain (+0.18), but every rule built
  from what the chip can actually see fails — because the judge has become a
  constant, and a constant carries no information.

## Part 3 — Vaccinating the judge doesn't work (but taught us something)

Obvious fix: generate the fooling examples ourselves, label them with real
PESQ, and retrain the judge so it knows what they're worth. We made 2,240
such examples and retrained.

**Result: no help.** The retrained judge is immune to the *old* tricks, so
the optimizer simply finds *new* ones. Fooling is defined relative to
whichever judge you attack — patch one judge's blind spots and the attack
moves to different ones. This can't be fixed with any fixed number of
offline patching rounds; the literature's fix (retraining the judge *during*
use) is exactly what a tiny frozen chip cannot do.

Two genuinely useful things fell out, though:

1. **Ordinary mis-scoring (being wrong about unfamiliar audio) IS fixable
   offline** — the retrained judge's baseline error on reverberant speech
   dropped from +1.48 to +1.01 even though it never saw reverb. Most of what
   we first called "fooling" was actually this fixable kind: of the +2.14 gap
   we originally measured, about two-thirds was ordinary mis-scoring and only
   one-third was true exploitation.
2. **The retrained judge is a perfect *bystander*.** Watching (not steering)
   the old judge being fooled, it stays sharp — its error vs truth shrinks to
   +0.02, because the fooled outputs are exactly what it was trained on.
   "Never optimize against the judge you trust" works as a mechanism. Sadly,
   even this honest bystander couldn't find a stopping rule that beat a fixed
   10 steps — the signal is just too small.

## Part 4 — The twist: the whole test was too easy to need adaptation

Before going further we finally ran the control experiment that should have
come first: classical, non-learning methods on the same test.

- Classical *adaptive* tricks (Wiener filtering etc.): all made things worse.
- But a **single fixed knob** — soften the enhancer's mask (raise it to the
  power 0.5, floor it at 0.1) — gave **+0.64 PESQ on 95% of clips**. Eight
  times everything the learning machinery achieved, with no learning, no
  judge, no per-clip decisions.

Why: our test shift was "add reverb," and under reverb the enhancer
over-suppresses in one systematic way. One constant correction fixes it.
Worse, the judges actively prefer the *wrong* direction (more suppression) —
they were trained on non-reverberant data where suppression is good — so
even using a judge just to *pick between* 21 candidate settings (no
gradients at all) lands at −0.33.

**The honest conclusion:** on any test where the fix is "one global knob," a
preset wins and adaptation is pointless. Self-tuning can only justify itself
when different clips need *different*, conflicting corrections.

## Part 5 — Where we are now

We built exactly that test: each clip gets one of four distortions whose
fixes point in opposite directions (reverb → soften the mask; heavier noise
→ harden it; re-colored noise; band-limited channel → leave it alone). The
best *single* preset — even one unfairly fit on the test data itself — should
gain almost nothing there. Every method now has to beat that null by making
genuinely per-clip decisions. That run is in progress.

## The lessons in one breath

1. Deploying the loop on a $10 chip is feasible today (modulo one vendor bug).
2. A learned quality judge, frozen on a chip, is a good *scorer* and an
   unusable *training signal*: optimizing against it fools it instantly, and
   its headline correlation with human scores says nothing about this.
3. "Fooling" splits in two: ordinary mis-scoring (fixable offline, worth
   fixing) and true exploitation (not fixable offline, moving target).
4. Never optimize against the judge you trust — a bystander judge stays
   honest. True in our data; not yet useful.
5. Always run the dumb baseline first. A constant beat our entire learning
   apparatus 8×, and it reshaped the whole research question.
