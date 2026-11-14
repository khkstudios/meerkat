"""Liveness preprocessing and encryption at rest / secure deletion. No camera or keychain needed:
the encryption key comes from MEERKAT_GALLERY_KEY and files go to a temporary directory."""
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from meerkat import app, secure
from meerkat import gallery as gallery_mod
from meerkat.config import LIVENESS_MODEL, MODELS_DIR
from meerkat.gallery import Gallery
from meerkat.liveness import EXPAND, INPUT, Liveness, preprocess, square_crop
from tests.test_core import det


class TestLivenessPreprocessing(unittest.TestCase):
    def test_square_crop_expands_and_pads(self):
        img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), np.uint8)
        c = square_crop(img, np.array([600, 300, 100, 120], float))
        self.assertEqual(c.shape[:2], (int(120 * EXPAND),) * 2)
        edge = square_crop(img, np.array([1250, 690, 100, 100], float))  # mostly off-frame
        self.assertEqual(edge.shape[:2], (150, 150))                     # reflect-padded, not cut

    def test_preprocess_shape_and_range(self):
        x = preprocess(np.full((180, 180, 3), 255, np.uint8))
        self.assertEqual(x.shape, (1, 3, INPUT, INPUT))
        self.assertAlmostEqual(float(x.max()), 1.0)

    def test_small_or_turned_faces_not_judged(self):
        self.assertFalse(Liveness.judgeable(det(100, 100, 40)))
        self.assertTrue(Liveness.judgeable(det(100, 100, 150)))

    @unittest.skipUnless((MODELS_DIR / LIVENESS_MODEL).exists(), "liveness model not downloaded")
    def test_model_runs(self):
        live = Liveness()
        img = np.random.default_rng(1).integers(0, 255, (720, 1280, 3), np.uint8)
        r = live(img, det(500, 250, 200))
        self.assertTrue(np.isfinite(r.logit_diff))


class TestEncryption(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {secure.ENV_KEY: secure.new_key()})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def gallery(self):
        g = Gallery("Test", "sface", 128)
        rng = np.random.default_rng(2)
        for b in (0.5, 1.0, 2.0):
            g.add(rng.normal(size=128).astype(np.float32), b, 120, "frontal")
        return g

    def test_roundtrip_and_ciphertext(self):
        g, p = self.gallery(), self.dir / "gallery_sface.npz.enc"
        g.save(p)
        raw = p.read_bytes()
        self.assertFalse(raw.startswith(b"PK"))              # not a readable npz/zip
        self.assertNotIn(g.emb.tobytes()[:64], raw)           # embeddings don't appear in clear
        self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        back = Gallery.load(p)
        self.assertTrue(np.array_equal(back.emb, g.emb))
        self.assertEqual(back.pose, g.pose)

    def test_wrong_key_is_refused(self):
        p = self.dir / "g.npz.enc"
        self.gallery().save(p)
        with mock.patch.dict(os.environ, {secure.ENV_KEY: secure.new_key()}):
            with self.assertRaises(secure.KeyUnavailable):
                Gallery.load(p)

    def test_plaintext_gallery_is_migrated_and_destroyed(self):
        enc, legacy = self.dir / "gallery_sface.npz.enc", self.dir / "gallery_sface.npz"
        g = self.gallery()
        g.save(legacy)  # no .enc suffix: plaintext, as galleries were before encryption
        with mock.patch.object(gallery_mod, "gallery_path", lambda b: enc), \
             mock.patch.object(gallery_mod, "legacy_gallery_path", lambda b: legacy):
            back = Gallery.open("sface")
        self.assertTrue(np.array_equal(back.emb, g.emb))
        self.assertTrue(enc.exists())
        self.assertFalse(legacy.exists())

    def test_secure_delete_overwrites_then_unlinks(self):
        p = self.dir / "secret.bin"
        p.write_bytes(b"A" * 4096)
        writes = []
        real_open = open

        def spy_open(path, mode="r", *a, **k):
            f = real_open(path, mode, *a, **k)
            if "r+" in mode:
                orig = f.write
                f.write = lambda b: (writes.append(bytes(b)), orig(b))[1]
            return f

        with mock.patch("builtins.open", spy_open):
            self.assertTrue(secure.secure_delete(p))
        self.assertFalse(p.exists())
        self.assertEqual(len(writes[0]), 4096)
        self.assertNotEqual(writes[0], b"A" * 4096)

    def test_forget_targets(self):
        data, results = self.dir / "data", self.dir / "results"
        data.mkdir(), results.mkdir()
        for name in ("gallery_sface.npz.enc", "gallery_sface.backup.npz", "calibration.json",
                     "canvas_1.png", ".gitkeep"):
            (data / name).write_bytes(b"x")
        (results / "sweep.csv").write_bytes(b"x")
        with mock.patch.object(app, "DATA_DIR", data), mock.patch.object(app, "RESULTS_DIR", results), \
             mock.patch.object(app, "CALIBRATION_FILE", data / "calibration.json"):
            names = {f.name for f in app.forget_targets()}
            everything = {f.name for f in app.forget_targets(include_all=True)}
        self.assertEqual(names, {"gallery_sface.npz.enc", "gallery_sface.backup.npz",
                                 "calibration.json", "sweep.csv"})
        self.assertEqual(everything - names, {"canvas_1.png"})


if __name__ == "__main__":
    unittest.main()
