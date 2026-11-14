# Webcam Face Recognition and Distance Tracking: Research

khkstudios project research, November 2026. Companion to the `dassie-50m` model.

> Implementation note: this repo (`meerkat`) implements Option A from section 8. The package
> is `meerkat/` rather than `src/`, and the shared per-frame loop lives in `meerkat/pipeline.py`.
> See README.md.

## 1. Goal

Use a laptop webcam to:

1. Detect faces in real time.
2. Recognise one enrolled person (Keegan) and tell them apart from everyone else.
3. Track that person frame to frame and draw a rectangle that shows recognition confidence live.
4. Estimate distance from the camera, and measure the furthest distance at which the system still recognises the person.

## 2. Key decision: enrol, don't train from scratch

Training a face recognition network from scratch needs millions of labelled faces across hundreds of thousands of identities. Recognising one person does not need that. The proven approach:

- A **pretrained embedding model** turns an aligned face crop into a vector (128-d or 512-d) so that crops of the same person land close together.
- **Enrolment** means capturing your face at several distances, angles and lighting conditions, then storing those vectors in a gallery.
- **Recognition** means taking the cosine similarity between a live face's vector and the gallery, then applying a threshold.

That is few-shot learning, and for a single identity it beats any classifier you could train on a few hundred selfies. Real training only comes in later as an optional step (section 9).

## 3. Pipeline

```
Webcam frame (720p or 1080p)
  -> Face detector         box + 5 landmarks + detection score
  -> Tracker               stable track_id per face across frames (Kalman + IoU)
  -> Alignment             5-point similarity warp to 112x112
  -> Embedding model       L2-normalised vector
  -> Gallery match         max / top-k cosine similarity vs enrolled vectors
  -> Track-level fusion    EMA of similarity + identity lock with hysteresis
  -> Distance estimate     face width in px -> metres (pinhole model)
  -> Overlay               colour-graded rectangle, %, distance, state
  -> Event bus             JSON events to dassie-50m / app layer
```

## 4. Model options

### 4.1 Face detection

| Model | Size | License | Notes |
|---|---|---|---|
| **YuNet** (OpenCV Zoo) | ~338 KB | MIT | Detects faces from about 10x10 to 300x300 px, returns 5 landmarks. `2026may` build has dynamic input shape for OpenCV 5. **Recommended.** |
| MediaPipe BlazeFace full-range | ~1 MB | Apache 2.0 | Short-range model is tuned for about 2 m, full-range for about 5 m. Good for a browser build (tasks-vision JS). |
| InsightFace SCRFD | 0.5 to 10 GF | Non-commercial models | Very accurate, but the licence rules it out for client or product work. |

### 4.2 Face recognition (embeddings)

| Model | Dim | License | Notes |
|---|---|---|---|
| **SFace** (OpenCV Zoo) | 128 | Apache 2.0 | Built into OpenCV via `FaceRecognizerSF`. LFW 99.60%, default cosine threshold 0.363. Easiest start. |
| **AuraFace-v1** (fal) | 512 | Apache 2.0 | ResNet100 ArcFace-style, LFW 99.65%. Same 512-d space as InsightFace buffalo_l, commercially usable. **Upgrade path.** |
| InsightFace buffalo_l / antelopev2 | 512 | Non-commercial | Best accuracy, but code is MIT and **weights are research-only**. Fine for a private experiment, not for anything shipped. |
| EdgeFace (Idiap) | 512 | CC BY-NC-SA 4.0 | Tiny (1.2M to 18M params), strong on edge devices, non-commercial. |
| AdaFace | 512 | Code MIT, check weights | Quality-adaptive margin built specifically for low-quality and low-resolution faces. Most relevant to the "walk away" problem. Check the training-data terms before any commercial use. |

**Recommendation:** start with YuNet + SFace (pure OpenCV, everything permissive). Move to YuNet + AuraFace when you want better accuracy at distance. Keep AdaFace in mind as a research benchmark for low-resolution recognition.

### 4.3 Tracking

- **ByteTrack** (MIT): Kalman filter prediction plus IoU association, and it also matches low-confidence boxes, which matters when a face gets small and the detection score drops as you walk away.
- For one or two faces on a webcam, a lightweight custom tracker (Kalman + IoU + Hungarian matching) is enough and avoids pulling in YOLOX dependencies.

### 4.4 Liveness / anti-spoofing (optional)

- **MiniFASNet** (Silent-Face-Anti-Spoofing, Apache 2.0): about 1.2 MB, catches printed photos and screen replays. 3D masks are a known weak spot. Only worth adding if recognition will unlock anything.

## 5. Distance estimation

Pinhole model (similar triangles):

```
focal_px   = (image_width_px / 2) / tan(HFOV / 2)
distance_m = real_face_width_m * focal_px / face_width_px
```

- Average face width is roughly 14 to 15 cm. Better: calibrate once by standing at a measured distance (for example 1.0 m), reading the box width, and solving `focal_px = face_width_px * distance / real_width`. That absorbs both your real face width and the lens FOV.
- Use the inter-eye landmark distance as well as box width. It is more stable than box width when you turn your head, and YuNet gives it to you for free.
- Smooth the estimate with an EMA or the tracker's Kalman state.

**Expected face size at 720p** (assuming ~65 deg HFOV, so focal ~1005 px, and a 14.3 cm face width. Laptop cameras range from about 55 to 78 deg, so calibrate):

| Distance | Face width (720p) | Face width (1080p) |
|---|---|---|
| 0.5 m | ~287 px | ~431 px |
| 1 m | ~144 px | ~216 px |
| 2 m | ~72 px | ~108 px |
| 3 m | ~48 px | ~72 px |
| 4 m | ~36 px | ~54 px |
| 5 m | ~29 px | ~43 px |
| 7 m | ~21 px | ~31 px |

What this means:

- **Detection** should hold until around 10 px (YuNet's floor), so detection range is not the bottleneck.
- **Recognition** is the bottleneck. Embedding models are trained on 112x112 crops. My working estimate is that similarity falls off fast once the face drops below about 40 to 50 px wide, so roughly 3 to 4 m at 720p. Treat that as a hypothesis the project should measure, not a fact.
- **Force the webcam to 1080p** (`CAP_PROP_FRAME_WIDTH/HEIGHT`) if it supports it. That alone adds about 50% range.

## 6. Confidence and the visual rectangle

Two numbers matter, and the UI should show both:

1. **Live similarity**: this frame's cosine similarity against the gallery.
2. **Track identity**: whether the track has been confirmed as you.

### Similarity to confidence

- Calibrate on your own data rather than trusting the published thresholds: collect "me" and "not me" similarities (other people, photos), then pick the threshold at your target false-accept rate.
- Starting points: SFace cosine 0.363 (OpenCV default). AuraFace and other ArcFace-style models sit on a different scale (one comparison measured a median of about 0.60 for the same person and 0.06 for different people), so recalibrate whenever you switch models.
- Map to a display percentage with a clipped linear ramp or a fitted logistic curve between the "not me" and "me" distributions. Don't show raw cosine as a percentage.

### Rectangle encoding

- Colour: red to amber to green, interpolated on calibrated confidence.
- Border thickness or fill opacity: scale with confidence.
- Label: `Keegan 87% | 2.4 m | LOCKED` or `Unknown 31% | 4.1 m`.
- HUD: current distance, **max confirmed recognition distance** for the session, FPS.

### Track-level fusion (the important part)

Per-frame similarity is noisy, and it collapses at distance. The fix:

- Keep an **EMA of similarity** per track, or better, average the normalised embeddings across the track's last N frames and compare that. Averaging embeddings over time is the most effective cheap trick for low-quality video.
- **Hysteresis lock**: confirm identity when the EMA stays above `T_high` for K frames (typically up close). Keep it while the tracker holds the box, even if per-frame similarity drops below threshold. Release only if similarity stays below `T_low` for M frames *while the face is still large enough to judge*, or if the track is lost.
- This separates "recognised" from "tracked as a recognised person". Walking away, the system keeps knowing it's you long after per-frame recognition stops working. The **true recognition distance** is the furthest point where per-frame similarity still clears the threshold, and the HUD should log that number separately.

## 7. Enrolment protocol

Capture about 50 to 150 good frames spread across:

- **Distance bands**: 0.5, 1, 2, 3, 4 m (prompt the user on screen: "step back").
- **Pose**: frontal, about 15 to 30 deg yaw left and right, slight pitch.
- **Lighting**: desk lamp, window light, dim room.
- **Accessories**: with and without glasses or a cap, if relevant.

Quality gates for each frame: detection score >= 0.9, blur check (variance of the Laplacian), face not cut off at the frame edge. Store embeddings tagged with their distance band and pose. Matching against the **max over the gallery**, or top-k mean, beats one averaged template, because far-away crops match far-away templates best.

**Adaptive gallery (optional):** while a track is locked and similarity is high, add new embeddings from distance bands that are under-represented. Cap the gallery size and only add above a strict threshold, or the gallery drifts toward other people.

## 8. Implementation stack options

| Option | Stack | When |
|---|---|---|
| **A. Python prototype** | `opencv-python` (FaceDetectorYN + FaceRecognizerSF), NumPy, optional `onnxruntime` for AuraFace | Fastest to a working demo. Start here. |
| B. Browser / React | MediaPipe tasks-vision (detector) + `onnxruntime-web` (SFace / AuraFace ONNX) + canvas overlay | Fits the existing React/Next.js stack. Runs fully client-side, which also helps with POPIA. |
| C. Native desktop | Python backend over WebSocket to a React or Electron UI | When dassie-50m and the vision loop should run as separate services. |

### Suggested repo layout

```
khkstudios/face-tracker
  README.md
  RESEARCH.md              this document
  models/                  ONNX files, gitignored, fetched by script
  scripts/fetch_models.py
  src/
    capture.py             webcam config (resolution, fps)
    detect.py              YuNet wrapper
    align_embed.py         alignment + embedding backend (sface | auraface)
    gallery.py             enrol / load / match, quality gates
    tracker.py             Kalman + IoU tracker, identity lock state machine
    distance.py            calibration + pinhole estimate
    overlay.py             rectangle, HUD, colour ramp
    events.py              JSON event emitter for dassie-50m
    app.py                 CLI: enrol | calibrate | run | eval
  data/
    gallery.npz            your embeddings (gitignored, never committed)
  eval/
    distance_sweep.py      logs similarity vs distance -> CSV + chart
```

### Evaluation plan

1. **Distance sweep**: walk from 0.5 m to the maximum in marked steps and log `distance_m, face_px, similarity, detected, recognised`. Plot similarity against distance for each model (SFace vs AuraFace) and resolution (720p vs 1080p).
2. **False-accept test**: other people plus photos of you on a phone screen.
3. **Report**: max recognition distance at a fixed false-accept rate. That is the headline number for the project.

## 9. If you want to actually train something

In order of value:

1. **Calibration head**: fit a logistic regression on similarity (plus face_px as a feature) to produce a properly calibrated "probability this is Keegan" that accounts for distance. Cheap, and it makes the confidence rectangle honest.
2. **Me-vs-not-me classifier on embeddings**: an SVM or small MLP trained on your gallery against a pool of other people's embeddings. A small gain over plain thresholding.
3. **Low-resolution fine-tune**: fine-tune the embedding backbone with heavy downscale-and-blur augmentation (the AdaFace recipe uses low-res augmentation) so far-away crops embed closer to your close-up ones. This is the only option that extends recognition range itself, and it needs a GPU and a proper dataset with compatible licences.
4. **Super-resolution before embedding**: possible, but it can hallucinate facial detail and push similarity up falsely. Benchmark it against step 3 before trusting it.

## 10. Integration with dassie-50m

I couldn't open the khkstudios Hugging Face org from this session, so this section assumes dassie-50m is a small text language model. Correct it if not.

- A 50M-parameter text model can't see images, so it shouldn't make the recognition decision. Keep detection, recognition and locking deterministic in the vision layer.
- The vision layer publishes **compact structured events** that dassie-50m takes as context or tool output:

```json
{"t": 1731600000.12, "track_id": 3, "identity": "keegan", "state": "locked",
 "similarity": 0.58, "confidence": 0.91, "distance_m": 2.4, "face_px": 61,
 "event": "approaching"}
```

- Emit on state changes (`entered`, `recognised`, `lost_recognition_but_tracked`, `left_frame`, `approaching`, `receding`) rather than every frame, so the model's small context isn't flooded.
- dassie-50m then handles the language side: greetings, presence-aware responses, logging, or acting as a natural-language interface over the event stream.

## 11. POPIA and privacy

- Under POPIA, biometric information (including facial recognition) is **special personal information**: processing is prohibited by default unless a section 27 exception applies, and consent is one of those exceptions.
- For a personal project recognising only yourself, risk is minimal. It matters as soon as other people walk into frame or this becomes a client or product feature.
- Design defaults: run on-device, store **embeddings only** (no images) by default, encrypt the gallery at rest, keep it out of git, don't persist faces of unknown people, and give a clear delete-my-data path. Any product version needs explicit consent capture and a retention policy.
- Not legal advice. Get a POPIA review before shipping this to clients.

## 12. Sources

- OpenCV DNN face tutorial (YuNet + SFace, LFW thresholds): https://docs.opencv.org/4.5.4/d0/dd4/tutorial_dnn_face.html
- OpenCV Zoo overview (YuNet 2026may dynamic input): https://docsearch.algolia.com/mcp/docs/repo/opencv/opencv_zoo
- YuNet 2026may redistribution (face size range, MIT): https://huggingface.co/pollen-robotics/face_detection_yunet_2026may
- SFace reference implementation (0.363 threshold): https://huggingface.co/opencv/face_recognition_sface/blame/main/sface.py
- MediaPipe face detection (short vs full range): https://mediapipe.readthedocs.io/en/latest/solutions/face_detection.html
- InsightFace model licensing: https://insightface.ai
- InsightFace model zoo (non-commercial notice): https://github.com/deepinsight/insightface/blob/e68b8b076fb11e9f75c87244d234385f0f318181/model_zoo/README.md
- AuraFace-v1 model card: https://huggingface.co/fal/AuraFace-v1/blob/main/README.md
- Licence and accuracy comparison (buffalo_l vs YuNet + AuraFace): https://huggingface.co/nynxz/Zentrainer
- EdgeFace model card: https://huggingface.co/Idiap/EdgeFace-Base
- AdaFace paper: https://arxiv.org/abs/2204.00964
- ByteTrack paper: https://www.arxiv.org/pdf/2110.06864v2
- ByteTrack licence: https://playground.roboflow.com/models/bytedance/bytetrack
- MiniFASNet anti-spoofing (Apache 2.0): https://github.com/suriAI/face-antispoof-onnx
- Single-camera distance estimation: https://medium.com/@aiphile/distance-estimation-with-single-camera-opencv-python-298a96383c2b
- POPIA special personal information: https://mjkinc.co.za/popia/special-personal-information
- Information Regulator on facial recognition (TechCentral): https://techcentral.co.za/police-bodycam-tender-popia-biometrics-rules/286116/
