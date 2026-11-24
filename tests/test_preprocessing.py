"""SensorMotion v0.6.1: regression tests for the EXISTING preprocessing.
Run from repository root: python -m unittest discover -s tests -p 'test_preprocessing.py' -v
"""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'backend' / 'model_core'
sys.path.insert(0, str(CORE))
common = importlib.import_module('common')
preprocess = importlib.import_module('02_preprocess_features')


class PreprocessingTests(unittest.TestCase):
    def test_sampling_and_windows(self):
        self.assertEqual(common.FS, 50)
        self.assertEqual(common.WIN, 100)
        self.assertEqual(common.STEP, 50)
        self.assertEqual(common.TRIM_SEC, 30)
        self.assertAlmostEqual(common.MAX_GAP, 0.5)

    def test_feature_count_and_values(self):
        t = np.arange(common.WIN) / common.FS
        signal = 1.0 + np.sin(2 * np.pi * 3 * t) + 0.2 * np.cos(2 * np.pi * 7 * t)
        a = preprocess.feats(signal, 'acc')
        g = preprocess.feats(signal * 0.1, 'gyr')
        self.assertEqual(len(a), 14)
        self.assertEqual(len(g), 14)
        self.assertEqual(len({**a, **g}), 28)
        self.assertTrue(all(np.isfinite(float(v)) for v in [*a.values(), *g.values()]))
        self.assertEqual(a, preprocess.feats(signal, 'acc'))

    def test_resampling_and_trim(self):
        t = np.arange(0, 80, 0.02)
        df = pd.DataFrame({'seconds_elapsed': t, 'x': np.sin(t), 'y': 0., 'z': 0.})
        values, valid = preprocess.mag_resampled(df)
        self.assertIsNotNone(values)
        self.assertEqual(len(values), len(valid))
        self.assertTrue(valid.all())
        self.assertTrue(np.isfinite(values).all())

    def test_gap_flagging(self):
        t = np.concatenate([np.arange(0, 40, .02), np.arange(41, 80, .02)])
        df = pd.DataFrame({'seconds_elapsed': t, 'x': np.sin(t), 'y': 0., 'z': 0.})
        values, valid = preprocess.mag_resampled(df)
        self.assertIsNotNone(values)
        self.assertTrue((~valid).any(), 'A >0.5s gap must be flagged')

    def test_per_trip_normalization(self):
        df = pd.DataFrame({'trip': ['A'] * 4 + ['B'] * 4,
                           'acc_mean': [1., 2., 3., 4., 11., 12., 13., 14.],
                           'gyr_std': [2., 4., 6., 8., 20., 40., 60., 80.]})
        z = common.normalize_per_trip(df, ['acc_mean', 'gyr_std'])
        self.assertTrue(np.allclose(z.groupby('trip')[['acc_mean', 'gyr_std']].median().values, 0))
        self.assertEqual(df['acc_mean'].iloc[0], 1., 'Original input should remain unchanged')

    def test_sensor_grid_offset_diagnostic(self):
        # Current mag_resampled() starts its grid from each sensor's own t[0].
        # This test documents the discrepancy; it does not pretend to fix it.
        t_acc = np.arange(0, 80, .02)
        t_gyr = np.arange(.01, 80.01, .02)
        a = pd.DataFrame({'seconds_elapsed': t_acc, 'x': np.ones_like(t_acc), 'y': 0., 'z': 0.})
        g = pd.DataFrame({'seconds_elapsed': t_gyr, 'x': np.ones_like(t_gyr), 'y': 0., 'z': 0.})
        va, _ = preprocess.mag_resampled(a)
        vg, _ = preprocess.mag_resampled(g)
        self.assertIsNotNone(va)
        self.assertIsNotNone(vg)
        self.assertAlmostEqual((t_gyr[0] + common.TRIM_SEC) - (t_acc[0] + common.TRIM_SEC), .01)
        print('\nDIAGNOSTIC: current accel/gyro resampling grids can differ by 0.01 seconds')


if __name__ == '__main__':
    unittest.main()
