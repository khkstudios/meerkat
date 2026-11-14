"""Gesture canvas: draw shapes and move items with your hands.

Gestures
  pinch (thumb + index) in empty space ... draw; on release the stroke snaps to a
                                           line / circle / ellipse / rectangle / triangle
  close a fist over an item ............... grab it; move your fist; open your hand to drop
  twist your wrist while holding ......... rotate it in the screen plane (one hand)
  fists on the same item with both hands . scale + rotate it
  drop an item on the bin ................ delete it
  swipe a held item quickly off-screen ... throw it away (delete); undo brings it back
  pinch a colour swatch .................. pick the colour
  pinch a toolbar button ................. undo / save / show-hide skeleton
  pinch and hold "exit" for 1 s .......... quit (or close the window)
  pinch the 2D/3D button ................. switch modes; in 3D, drawn shapes become solids:
                                           circle->sphere, square->cube, rectangle->box,
                                           triangle->pyramid, ellipse->ellipsoid, line->cylinder
  pinch-drag on a 3D solid ............... rotate it (it keeps spinning a little when released)
  both palms open, held 1.5 s ............ clear the canvas
No keyboard needed. Optional shortcuts: m 2D/3D, u undo, c clear, s save, h skeleton, q quit
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DATA_DIR
from .hands import Hand, HandTracker, HandWorker, draw_hand
from .shapes import Item, recognise
from .solids import SOLID_FOR, Solid

PALETTE = [(60, 200, 255), (80, 220, 80), (255, 170, 60), (90, 90, 255), (255, 110, 220), (245, 245, 245)]
GRAB_TOL = 25        # px: how close to an outline counts as "on" an item (hover)
FIST_TOL = 45        # px: a fist is a blunter pointer, so be more forgiving
CLEAR_HOLD = 1.5     # s: both palms open to clear
EXIT_HOLD = 1.0      # s: pinch held on "exit" to quit
TWIST_GAIN = 1.5     # item rotation per radian of wrist twist (wrists only turn ~180 deg)
THROW_SPEED = 1.2    # screen widths per second a held item must reach to count as a throw
THROW_WINDOW = 0.25  # s: look for the throw's peak speed in this much recent motion
THROW_EXIT = 0.3     # s: ...and it must be heading for an edge it would reach this soon
THROW_END = 0.4      # ...and still be moving at this fraction of throw speed when let go
FLY_TIME = 0.6       # s: throw-away animation


@dataclass
class Action:
    mode: str = "idle"                   # idle | draw | grab | rotate
    stroke: list = field(default_factory=list)
    item: Item | Solid | None = None
    last: np.ndarray | None = None       # palm position last frame (fist dragging)
    pinch_last: np.ndarray | None = None # pinch position last frame (3D rotation)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    roll_last: float | None = None       # wrist angle last frame (one-hand twist)
    hold_since: float | None = None      # pinch-hold on the exit button
    trail: list = field(default_factory=list)  # recent (t, palm) while holding, for throws


@dataclass
class TwoHand:
    item: Item | Solid
    keys: tuple[str, str]
    base: object
    v0: np.ndarray
    m0: np.ndarray


def chime_safe(sound: str) -> None:
    """Audible cue on macOS; silent elsewhere and in tests."""
    if sys.platform == "darwin" and "unittest" not in sys.modules:
        import subprocess
        try:
            subprocess.Popen(["afplay", f"/System/Library/Sounds/{sound}.aiff"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


class GestureEvents:
    def __init__(self, sink: str | None):
        self.out = None if sink in (None, "none") else (sys.stdout if sink == "-" else open(sink, "a", buffering=1))

    def emit(self, event: str, **kw) -> None:
        if self.out:
            self.out.write(json.dumps({"t": round(time.time(), 2), "event": event, **kw}) + "\n")


class Canvas:
    """All interaction logic, independent of the camera (so it can be unit-tested)."""

    def __init__(self, width: int, height: int, events: GestureEvents | None = None):
        self.w, self.h = width, height
        self.items: list[Item] = []
        self.colour = PALETTE[0]
        self.actions: dict[str, Action] = {}
        self.two: TwoHand | None = None
        self.undo_stack: list[list[Item]] = []
        self.clear_since: float | None = None
        self.toasts: list[tuple[float, str, np.ndarray]] = []
        self.events = events or GestureEvents(None)
        r = 22
        self.swatches = [((40 + i * 60, 80), r, c) for i, c in enumerate(PALETTE)]
        self.bin = (width - 150, height - 120, width - 20, height - 20)
        x = 40 + len(PALETTE) * 60 + 10
        self.buttons: dict[str, tuple[int, int, int, int]] = {}
        for name, bw in (("mode", 84), ("undo", 96), ("save", 96), ("hand", 96)):
            self.buttons[name] = (x, 80 - 24, x + bw, 80 + 24)
            x += bw + 14
        self.buttons["exit"] = (width - 116, 80 - 24, width - 20, 80 + 24)
        self.mode_button = self.buttons["mode"]
        self.mode3d = False
        self.skeleton = True
        self.save_requested = False
        self.quit = False
        self.exit_progress = 0.0
        self._now = 0.0
        self.flying: list[list] = []  # [item, velocity px/s, last t, end t] thrown-away items
        self.reach_rect: tuple[int, int, int, int] | None = None  # camera area mapped to the canvas

    # ------------------------------------------------------------ helpers
    def snapshot(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.items))
        self.undo_stack = self.undo_stack[-50:]

    def undo(self) -> None:
        if self.undo_stack:
            self.items = self.undo_stack.pop()
            self._cancel_all()
            self.events.emit("undo")

    def clear(self) -> None:
        if self.items:
            self.snapshot()
            self.items = []
            self._cancel_all()
            self.events.emit("canvas_cleared")

    def _cancel_all(self) -> None:
        self.actions.clear()
        self.two = None

    def item_at(self, p: np.ndarray, tol: float = GRAB_TOL) -> Item | None:
        best, bd = None, tol
        for it in reversed(self.items):  # topmost first
            d = it.distance(p)
            if d <= bd:
                best, bd = it, d
                if d == 0:
                    break
        return best

    def swatch_at(self, p: np.ndarray):
        for (x, y), r, c in self.swatches:
            if np.hypot(p[0] - x, p[1] - y) <= r + 10:
                return c
        return None

    def toggle_mode(self) -> None:
        self.mode3d = not self.mode3d
        self.events.emit("mode_changed", mode="3d" if self.mode3d else "2d")

    def button_at(self, p: np.ndarray) -> str | None:
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 - 8 <= p[0] <= x1 + 8 and y0 - 8 <= p[1] <= y1 + 8:
                return name
        return None

    def in_bin(self, p: np.ndarray) -> bool:
        x0, y0, x1, y1 = self.bin
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def toast(self, text: str, at: np.ndarray) -> None:
        self.toasts.append((time.time() + 1.2, text, np.asarray(at, float)))

    # ------------------------------------------------------------ main update
    def update(self, hands: list[Hand], t: float, enabled: bool = True) -> None:
        self._now = t
        self._step_flying(t)
        if not enabled:
            self._cancel_all()
            self.clear_since = None
            return
        live = [h for h in hands if h.pose != "gone"]
        by_key = {h.side: h for h in hands}

        # both palms open and held -> clear
        if len(live) == 2 and all(h.pose == "open" and not h.pinching for h in live):
            self.clear_since = self.clear_since or t
            if t - self.clear_since >= CLEAR_HOLD:
                self.clear()
                self.clear_since = None
        else:
            self.clear_since = None

        for h in hands:
            act = self.actions.setdefault(h.side, Action())
            # pinch = draw / pick colour
            if h.pinch_started:
                self._on_pinch_start(act, h.cursor)
            elif h.pinching and act.mode == "draw":
                if np.linalg.norm(h.cursor - act.stroke[-1]) > 1.5:
                    act.stroke.append(h.cursor.copy())
            elif h.pinching and act.mode == "exit":
                if self.button_at(h.cursor) != "exit":
                    act.mode, act.hold_since = "idle", None  # slid off the button: cancelled
                else:
                    self.exit_progress = min(1.0, (t - act.hold_since) / EXIT_HOLD)
                    if self.exit_progress >= 1.0:
                        self.quit = True
            elif h.pinching and act.mode == "rotate" and act.pinch_last is not None:
                d = h.cursor - act.pinch_last
                act.item.rotate_by_drag(d[0], d[1])
                act.vel = 0.6 * act.vel + 0.4 * d
            if h.pinch_ended and act.mode == "draw":
                self._finish_stroke(act)
            if h.pinch_ended and act.mode == "exit":
                act.mode, act.hold_since = "idle", None
            if h.pinch_ended and act.mode == "rotate":
                k = 0.012  # hand the drag's momentum to the solid as a decaying spin
                act.item.spin = np.clip(np.array([-act.vel[0] * k, act.vel[1] * k]), -0.15, 0.15)
                self.events.emit("item_rotated", item_id=act.item.id, shape=act.item.kind)
                act.mode, act.item = "idle", None
            act.pinch_last = h.cursor.copy()
            # fist = grab / move / drop
            if h.fist_started and act.mode == "idle":
                self._on_grab_start(h, act, h.palm)
                act.trail = [(t, h.palm.copy())]
            elif h.fist and act.mode == "grab" and self.two is None and act.last is not None:
                act.item.translate(h.palm - act.last)
                act.trail = [(tt, pp) for tt, pp in act.trail if t - tt <= THROW_WINDOW + 0.1]
                act.trail.append((t, h.palm.copy()))
                if act.roll_last is not None:  # one-hand twist: wrist rotation turns the item
                    d = h.roll - act.roll_last
                    if abs(d) > 0.004:  # ignore tracking jitter
                        act.item.rotate_inplane(d * TWIST_GAIN)
            act.roll_last = h.roll
            if h.fist_ended and act.mode == "grab":
                self._on_grab_end(act, t, lost=h.pose == "gone")
            act.last = h.palm.copy()

        if self.two is not None:
            a, b = (by_key.get(k) for k in self.two.keys)
            if a is None or b is None or not (a.fist and b.fist):
                self._end_two_hand(by_key)
            else:
                self._two_hand_move(a.palm, b.palm)

        for key in [k for k in self.actions if k not in by_key]:
            del self.actions[key]
        if not any(a.mode == "exit" for a in self.actions.values()):
            self.exit_progress = 0.0

        rotating = {id(a.item) for a in self.actions.values() if a.mode == "rotate"}
        for it in self.items:
            if id(it) not in rotating:
                it.step_spin()

    def _on_pinch_start(self, act: Action, p: np.ndarray) -> None:
        c = self.swatch_at(p)
        if c is not None:
            self.colour = c
            self.events.emit("colour_changed", colour=list(c))
            return
        btn = self.button_at(p)
        if btn == "mode":
            self.toggle_mode()
        elif btn == "undo":
            self.undo()
        elif btn == "save":
            self.save_requested = True
        elif btn == "hand":
            self.skeleton = not self.skeleton
        elif btn == "exit":
            act.mode, act.hold_since = "exit", self._now
        if btn is not None or act.mode != "idle":
            return
        it = self.item_at(p)
        if isinstance(it, Solid):  # pinch on a solid spins it instead of drawing
            self.snapshot()
            it.spin = np.zeros(2)
            act.mode, act.item, act.vel = "rotate", it, np.zeros(2)
            return
        act.mode, act.stroke = "draw", [p.copy()]

    def _finish_stroke(self, act: Action) -> None:
        if len(act.stroke) >= 2:
            kind, pts = recognise(np.array(act.stroke))
            if kind != "dot":
                self.snapshot()
                it = Solid.from_shape(kind, pts, self.colour) if (self.mode3d and kind in SOLID_FOR) else None
                it = it or Item(kind, pts, self.colour)
                self.items.append(it)
                self.toast(it.kind, it.centre)
                self.events.emit("shape_drawn", item_id=it.id, shape=it.kind, solid=isinstance(it, Solid),
                                 centre=np.round(it.centre).tolist())
        act.mode, act.stroke = "idle", []

    def _on_grab_start(self, h: Hand, act: Action, p: np.ndarray) -> None:
        it = self.item_at(p, FIST_TOL)
        if it is None:
            return
        # second fist on an item the other hand holds -> two-hand transform
        for k, other in self.actions.items():
            if k != h.side and other.mode == "grab" and other.item is it:
                self.two = TwoHand(it, (k, h.side), it.state(), p - other.last, (p + other.last) / 2)
                act.mode, act.item = "grab", it
                return
        self.snapshot()
        act.mode, act.item = "grab", it
        self.items.remove(it)
        self.items.append(it)  # bring to front

    def throw_velocity(self, trail: list, t: float, lost: bool = False) -> np.ndarray | None:
        """Velocity (px/s) if the recent motion was a throw towards an edge, else None.

        Uses the fastest stretch in the last THROW_WINDOW s: people slow down as they let go.
        A hand that left the frame while holding needs only half the speed.
        """
        if not trail:
            return None
        t_ref = trail[-1][0]  # end of the motion (a lost hand reports its release frames later)
        recent = [(tt, pp) for tt, pp in trail if t_ref - tt <= THROW_WINDOW]
        best, best_speed = None, 0.0
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                dt = recent[j][0] - recent[i][0]
                if dt < 0.05:  # too short a baseline: noise
                    continue
                v = (recent[j][1] - recent[i][1]) / dt
                if np.linalg.norm(v) > best_speed:
                    best, best_speed = v, float(np.linalg.norm(v))
        need = THROW_SPEED * self.w * (0.5 if lost else 1.0)
        if best is None or best_speed < need:
            return None
        if not lost:  # a quick reposition stops before the hand opens; a throw doesn't
            tail = [(tt, pp) for tt, pp in recent if t_ref - tt <= 0.1]
            if len(tail) >= 2 and tail[-1][0] - tail[0][0] >= 0.05:
                end_speed = np.linalg.norm(tail[-1][1] - tail[0][1]) / (tail[-1][0] - tail[0][0])
                if end_speed < THROW_END * need:
                    return None
        p = recent[-1][1]
        # time until that motion would carry the palm off the nearest edge it's heading for
        tx = (self.w - p[0]) / best[0] if best[0] > 1 else (p[0] / -best[0] if best[0] < -1 else np.inf)
        ty = (self.h - p[1]) / best[1] if best[1] > 1 else (p[1] / -best[1] if best[1] < -1 else np.inf)
        # a hand that vanished mid-swipe was leaving the frame or blurred by speed: be lenient
        return best if min(tx, ty) <= THROW_EXIT * (2 if lost else 1) else None

    def _step_flying(self, t: float) -> None:
        keep = []
        for f in self.flying:
            it, v, t_last, t_end = f
            dt = max(0.0, t - t_last)
            it.translate(v * dt)
            it.scale_by(0.9 ** (dt * 30))  # shrink as it flies away
            f[2] = t
            if t < t_end:
                keep.append(f)
        self.flying = keep

    def _on_grab_end(self, act: Action, t: float, lost: bool = False) -> None:
        it = act.item
        v = self.throw_velocity(act.trail, t, lost) if it is not None and self.two is None else None
        act.trail = []
        if v is not None and it in self.items:
            self.items.remove(it)
            self.flying.append([it, v * 1.5, t, t + FLY_TIME])
            self.toast("thrown away", np.clip(act.last, 80, (self.w - 160, self.h - 40)))
            self.events.emit("item_deleted", item_id=it.id, shape=it.kind, how="thrown")
            chime_safe("Pop")
            act.mode, act.item = "idle", None
            return
        if it is not None and self.two is None:
            # act.last is the palm while still a fist (the opening frame is ignored)
            if act.last is not None and self.in_bin(act.last) and it in self.items:
                self.items.remove(it)
                self.toast("deleted", act.last)
                self.events.emit("item_deleted", item_id=it.id, shape=it.kind, how="bin")
            else:
                self.events.emit("item_moved", item_id=it.id, shape=it.kind,
                                 centre=np.round(it.centre).tolist())
        act.mode, act.item = "idle", None

    def _two_hand_move(self, pa: np.ndarray, pb: np.ndarray) -> None:
        tw = self.two
        v = pb - pa
        n0 = max(np.linalg.norm(tw.v0), 1e-6)
        scale = float(np.clip(np.linalg.norm(v) / n0, 0.15, 8.0))
        angle = math.atan2(v[1], v[0]) - math.atan2(tw.v0[1], tw.v0[0])
        tw.item.apply_transform(tw.base, tw.m0, scale, angle, (pa + pb) / 2 - tw.m0)

    def _end_two_hand(self, by_key: dict[str, Hand]) -> None:
        tw = self.two
        self.two = None
        self.events.emit("item_transformed", item_id=tw.item.id, shape=tw.item.kind)
        # the hand still making a fist keeps dragging from where it is now
        for k in tw.keys:
            act = self.actions.get(k)
            h = by_key.get(k)
            if act is None:
                continue
            if h is not None and h.fist:
                act.mode, act.item, act.last = "grab", tw.item, h.palm.copy()
            else:
                act.mode, act.item = "idle", None

    # ------------------------------------------------------------ render
    def draw(self, img: np.ndarray, hands: list[Hand], skeleton: bool, t: float) -> None:
        # bin
        x0, y0, x1, y1 = self.bin
        dragging = any(a.mode == "grab" for a in self.actions.values())
        hot = dragging and any(self.in_bin(h.palm) for h in hands if h.fist)
        cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 230) if hot else (90, 90, 90), 2 + 2 * hot, cv2.LINE_AA)
        cv2.putText(img, "bin", (x0 + 45, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (60, 60, 230) if hot else (160, 160, 160), 2, cv2.LINE_AA)
        # reach box: hand positions inside it span the whole canvas
        if self.reach_rect is not None:
            rx0, ry0, rx1, ry1 = self.reach_rect
            cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (110, 110, 110), 1, cv2.LINE_AA)
        # swatches
        for (x, y), r, c in self.swatches:
            cv2.circle(img, (x, y), r, c, -1, cv2.LINE_AA)
            if c == self.colour:
                cv2.circle(img, (x, y), r + 5, (255, 255, 255), 3, cv2.LINE_AA)
        # toolbar
        hovering = {self.button_at(h.cursor) for h in hands if h.pose != "gone"}
        labels = {"mode": "3D" if self.mode3d else "2D", "undo": "undo", "save": "save",
                  "hand": "hand" if self.skeleton else "hand", "exit": "exit"}
        for name, (bx0, by0, bx1, by1) in self.buttons.items():
            active = (name == "mode" and self.mode3d) or (name == "hand" and self.skeleton)
            col = (255, 200, 80) if active else (200, 200, 200)
            if name == "exit":
                col = (90, 90, 235)
            cv2.rectangle(img, (bx0, by0), (bx1, by1), (40, 40, 40), -1, cv2.LINE_AA)
            if name == "exit" and self.exit_progress > 0:
                fill = int(bx0 + (bx1 - bx0) * self.exit_progress)
                cv2.rectangle(img, (bx0, by0), (fill, by1), (60, 60, 200), -1, cv2.LINE_AA)
            cv2.rectangle(img, (bx0, by0), (bx1, by1), col, 4 if name in hovering else 2, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(labels[name], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(img, labels[name], ((bx0 + bx1 - tw) // 2, by0 + 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, col, 2, cv2.LINE_AA)
        # items
        held = {a.item.id for a in self.actions.values() if a.mode == "grab" and a.item}
        hover = set()
        for h in hands:  # show what a fist would grab from where the palm is now
            act = self.actions.get(h.side)
            if not h.pinching and (act is None or act.mode == "idle"):
                it = self.item_at(h.palm, FIST_TOL)
                if it is not None:
                    hover.add(it.id)
        for it in self.items:
            it.draw(img, highlight=it.id in held or it.id in hover)
        for it, *_ in self.flying:
            it.draw(img)
        # in-progress strokes
        for a in self.actions.values():
            if a.mode == "draw" and len(a.stroke) > 1:
                cv2.polylines(img, [np.array(a.stroke, np.int32).reshape(-1, 1, 2)], False,
                              self.colour, 5, cv2.LINE_AA)
        # hands
        for h in hands:
            if h.pose == "gone":
                continue
            if skeleton:
                draw_hand(img, h)
            if h.fist:  # grab cursor at the palm
                c = tuple(int(v) for v in h.palm)
                cv2.circle(img, c, 22, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, c, 22, (40, 40, 40), 2, cv2.LINE_AA)
                label = "grab"
            else:
                c = tuple(int(v) for v in h.cursor)
                if h.pinching:
                    cv2.circle(img, c, 12, self.colour, -1, cv2.LINE_AA)
                cv2.circle(img, c, 16, (255, 255, 255), 2, cv2.LINE_AA)
                pc = tuple(int(v) for v in h.palm)  # faint palm marker: where a fist would grab
                cv2.circle(img, pc, 6, (200, 200, 200), 1, cv2.LINE_AA)
                label = "pinch" if h.pinching else h.pose
            cv2.putText(img, label, (c[0] + 26, c[1] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # clear progress
        if self.clear_since is not None:
            frac = min(1.0, (t - self.clear_since) / CLEAR_HOLD)
            centre = (self.w // 2, self.h // 2)
            cv2.ellipse(img, centre, (60, 60), -90, 0, 360 * frac, (60, 60, 230), 10, cv2.LINE_AA)
            cv2.putText(img, "clear", (centre[0] - 33, centre[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
        # toasts
        now = time.time()
        self.toasts = [x for x in self.toasts if x[0] > now]
        for _, text, at in self.toasts:
            cv2.putText(img, text, (int(at[0]) - 30, int(at[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)

    def save(self, frame: np.ndarray) -> str:
        DATA_DIR.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = DATA_DIR / f"canvas_{stamp}"
        img = np.zeros_like(frame)
        for it in self.items:
            it.draw(img)
        cv2.imwrite(str(base) + ".png", img)
        (base.with_suffix(".json")).write_text(json.dumps([it.to_json() for it in self.items], indent=1))
        return str(base)


# ---------------------------------------------------------------- app loop
def run(a, open_capture, draw_hud, draw_banner, show, QUIT) -> int:
    """Three threads: camera (newest frame), hands (MediaPipe on every new frame) and this one
    (canvas + drawing + window). None of them waits on the others' work."""
    import threading

    cap = open_capture(a).start_thread()
    tracker = HandTracker(num_hands=2, reach=a.reach)
    worker = HandWorker(tracker, cap)
    canvas = Canvas(cap.width, cap.height, GestureEvents(a.events))

    owner_state = {"locked": True}
    owner_thread = None
    if a.owner_only:
        from .align_embed import Embedder
        from .config import Calibration, TrackerConfig
        from .gallery import Gallery
        from .pipeline import Pipeline
        if not Gallery.exists(a.backend):
            print("--owner-only needs an enrolled gallery: python -m meerkat enrol")
            return 1
        owner = Pipeline(Gallery.open(a.backend), Embedder(a.backend), Calibration.load(), cap.width,
                         TrackerConfig(liveness=a.liveness))
        owner_state["locked"] = False

        def owner_loop():  # face check at ~5 Hz on its own thread: never stalls the hands
            seq = 0
            while worker.running:
                frame, new_seq, t = cap.wait_new(seq)
                if frame is None or new_seq == seq:
                    continue
                seq = new_seq
                owner_state["locked"] = any(tr.locked for tr in owner.process(frame, t))
                time.sleep(0.15)

        owner_thread = threading.Thread(target=owner_loop, name="owner", daemon=True)
        owner_thread.start()

    fps, last, hs, disp_seq = 0.0, time.time(), [], 0
    print(__doc__, file=sys.stderr)
    try:
        while True:
            frame, seq, _ = cap.wait_new(disp_seq, timeout=1.0)  # redraw once per camera frame
            if frame is None or seq == disp_seq:
                if not cap.running:
                    break
                continue
            disp_seq = seq
            locked = owner_state["locked"]
            for res, t in worker.drain():  # every hand result, in order: no lost pinch/fist edges
                canvas.update(res, t, enabled=locked)
                hs = res
            now = time.time()
            if canvas.reach_rect is None:
                canvas.reach_rect = tracker.reach_rect()

            view = cv2.convertScaleAbs(frame, alpha=0.6)  # dim the video so the drawing pops
            canvas.draw(view, hs, canvas.skeleton, now)
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            draw_hud(view, [f"{cap.describe()} | display {fps:.0f} fps | hands {worker.hz:.0f} Hz | "
                            f"{len(canvas.items)} items"])
            if not locked:
                draw_banner(view, ["Gestures locked: owner not recognised"])
            if canvas.save_requested:
                canvas.save_requested = False
                path = canvas.save(frame)
                canvas.toast("saved", np.array([canvas.w / 2, canvas.h / 2]))
                print(f"saved {path}.png/.json", file=sys.stderr)
            k = show("meerkat gestures", view)
            if canvas.quit or k in (QUIT, ord("q"), 27):  # exit button, window close, or q
                break
            if k == ord("u"):
                canvas.undo()
            if k == ord("m"):
                canvas.toggle_mode()
            if k == ord("c"):
                canvas.clear()
            if k == ord("h"):
                canvas.skeleton = not canvas.skeleton
            if k == ord("s"):
                canvas.save_requested = True
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    return 0
