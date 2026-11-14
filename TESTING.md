# Testing notes

Automated: `python -m unittest discover tests` covers the tracker, the identity lock state
machine, distance estimation and the confidence ramp. Everything below needs a person in
front of the camera and has to be checked by hand.

## Enrol / sweep countdown (added 2026-11-14) — needs manual testing

**Verified so far:** the `Countdown` timer fires exactly once and then clears itself, and the
large text renders legibly on a 1280x720 frame (checked on a still image).

**Not yet verified on the real camera:**

- [ ] SPACE starts an 8 s countdown. The large number at the top of the frame is readable from 4 m.
- [ ] A "Tink" sound plays each second and a "Glass" chime plays when capture starts (`afplay`, macOS system sounds).
- [ ] Frames captured while walking are rejected. Capture starts only after ~10 steady frames inside the band's range, and the banner shows "hold still..." until then.
- [ ] A "Hero" chime plays when a band reaches 25/25, and the next band waits for SPACE.
- [ ] C cancels a running countdown, and S during a countdown skips the band and cancels the countdown.
- [ ] The large counter at the top doesn't cover your face at 0.5 m (it may at very close range; acceptable if so).
- [ ] 8 s is enough time to walk to 4 m. If not, change the `--countdown` default.
- [ ] `eval.distance_sweep`: same countdown and chimes for each step and for `--continuous` mode.

## Gestures (added 2026-11-14) — needs manual testing

**Verified so far:** 24 unit tests pass, covering shape recognition, pose and pinch logic on
synthetic landmarks, and the canvas interactions (draw, move, delete to bin, undo, two-hand
scale, swatch, clear, owner lock). A stress test with wobbly synthetic strokes scored 98–100%
correct for each shape type. On a sample photo, MediaPipe found both hands with the correct
left/right labels. The UI renders correctly on a still frame.

**Not yet verified on the real camera:**

- [ ] Pinch detection feels reliable: no accidental pinches, no drops mid-stroke. Tune `HandTracker.PINCH_ON` / `PINCH_OFF` / `PINCH_FRAMES` (now 0.20 / 0.45 × palm size, 4 frames; was 0.30 and 2 frames).
- [ ] Fist grab: closing a fist over an item grabs it within ~0.1 s, opening drops it, and the item doesn't jump on release. Fist hover highlight shows what will be grabbed.
- [ ] A fist never registers as a pinch (thumb resting on the curled index).
- [ ] Cursor smoothing: lines aren't jittery and the cursor doesn't lag (`OneEuro` min_cutoff / beta).
- [ ] Hand-drawn shapes snap correctly in the air (real strokes are sloppier than test data).
- [ ] Two-fist scale and twist is intuitive, and opening one hand carries on as a one-hand drag.
- [ ] Drop on the bin deletes; tapping a swatch changes colour without drawing a dot.
- [ ] Both palms open for 1.5 s clears the canvas, and it doesn't trigger by accident (e.g. when waving).
- [ ] Left/right hand labels stay stable when hands cross.
- [ ] Frame rate with two hands, and with `--owner-only` (face check every 3rd frame).
- [ ] `--owner-only` locks gestures when you leave and unlocks when you're recognised again.

## 3D solids (added 2026-11-14) — needs manual testing

**Verified so far:** unit tests cover the shape → solid mapping, sizing, culling and drawing,
trackball rotation with no drift, spin decay, the 2D/3D toggle, and in 3D mode: draw → rotate →
fist move → two-fist scale → bin → undo. All six solids render correctly on a still frame.
Draw time per solid: sphere ~2.8 ms, cube ~0.2 ms, cylinder ~0.3 ms.

**Not yet verified on the real camera:**

- [ ] Air-drawn squares come out as cubes (not boxes) often enough. The square cutoff is an aspect ratio > 0.85.
- [ ] Pinch-drag rotation speed feels right (`rotate_by_drag` k = 0.012 rad/px), and the release spin isn't too strong.
- [ ] Pinch-to-rotate vs pinch-to-draw: starting a stroke right next to a solid draws rather than rotates (hit area is the solid's silhouette).
- [ ] Frame rate with several spheres on screen.

## Hands-free control and one-hand twist (added 2026-11-14) — needs manual testing

**Verified so far:** unit tests cover the toolbar (undo / save / hand), the exit hold (an early
release or sliding off cancels it), wrist twist on 2D items and solids (Rz only, with gain 1.5),
and hand-roll unwrapping through ±180°. A window destroyed in code is detected as closed.

**Not yet verified on the real camera:**

- [ ] Clicking the window's red close button ends every screen (calibrate, enrol, run, sweep, gestures).
- [ ] Calibrate: the countdown starts on launch and collection starts on its own.
- [ ] Enrol: each band's countdown starts by itself after the previous band's chime, and a band that can't be filled is skipped after `--band-timeout` (90 s) with a low "Basso" sound.
- [ ] Sweep: each step's countdown starts by itself, and `--continuous` stops after `--duration` (60 s).
- [ ] Wrist twist while holding a fist rotates smoothly. Watch for jitter (0.004 rad dead-zone), and check the gain (1.5) feels right.
- [ ] Turning your wrist doesn't break the fist detection (curled fingers seen edge-on).
- [ ] Toolbar buttons are easy to hit with a pinch, and a pinch near the toolbar doesn't start a stroke by accident.
- [ ] Exit needs a deliberate 1 s hold; the red fill shows progress.

## Throw to delete (added 2026-11-14) — needs manual testing

**Verified so far:** unit tests cover a fast swipe to the edge (thrown, animated, undo restores
the original position), a slow drag to the edge (just a move), a fast swipe that ends mid-screen
(not a throw), a fast move that stops before letting go (not a throw), and a hand that blurs out
of tracking mid-swipe (thrown).

**Rules:** peak palm speed over the last 0.25 s ≥ 1.2 screen widths/s (half that if the hand was
lost), still moving at ≥ 40% of that when let go, and heading for an edge it would reach within
0.3 s (0.6 s if lost). Constants: `THROW_*` in `meerkat/gestures.py`.

**Not yet verified on the real camera:**

- [ ] A natural flick towards any edge throws the item away, including up and down, not just sideways.
- [ ] Fast but deliberate moves (repositioning across the screen) never throw by accident.
- [ ] Motion blur during the swipe: the hand dropping out of tracking still counts as a throw.
- [ ] The "Pop" sound and the fly-off animation (0.6 s, shrinking) feel right.

## Tracking speed and reach (added 2026-11-14)

**Measured on this Mac (FaceTime HD, 1280x720):**

- Before: the loop ran at 14.4 fps: 29 ms waiting for the camera + 18 ms MediaPipe + 4 ms canvas + 19 ms window. Processing took longer than one camera frame, so every other frame was missed.
- After (threads for camera, hands and display): hands 30.3 Hz, display 29.7 fps, which is the camera's maximum.
- A video-file source exits cleanly at the end of the file.

**Not yet verified with real use:**

- [ ] Fast swipes and throws stay tracked at 30 Hz. If they still drop, more light helps most: in a dim room the camera uses a longer exposure, and fast hands blur.
- [ ] `--reach 0.1` makes the screen edges easy to reach; the faint reach box helps. Try `--reach 0.15` if edges are still awkward, or `0` to turn it off.
- [ ] Lower tracking confidence (0.3) keeps edge-on hands tracked without ghost hands appearing.
- [ ] `--owner-only` face check (now its own ~5 Hz thread) doesn't slow the hands.

## Other untested paths

- [ ] Distance band enforcement during enrol (amber box, "step back / come closer").
- [ ] Black-frame warning banner (only shows if camera permission is missing or the lens is covered).
- [ ] `run --adaptive` gallery growth over a real session.
- [ ] AuraFace backend (`pip install onnxruntime && python scripts/fetch_models.py --auraface`).
