"""Public tests of the numerical method; no contest truth is imported."""
import unittest

import numpy as np

from rift_numerics import rollout, diagnostic_rollout, sample_indices


class SharedNumericsTests(unittest.TestCase):
    def test_constant_acceleration_and_input_ownership(self):
        x = np.array([[1, 2], [-3, 4]], dtype=np.float32)
        v = np.array([[.5, -.2], [.3, .7]], dtype=np.float32)
        x0, v0 = x.copy(), v.copy()
        acceleration = np.array([.1, -.05])
        result = rollout(lambda t, state: np.broadcast_to(acceleration, state.shape),
                         x, v, duration=10.)
        self.assertEqual(result['positions'].dtype, np.float64)
        self.assertEqual(result['positions'].shape, (2001, 2, 2))
        expected = x0.astype(float)+v0.astype(float)*10+.5*acceleration*100
        np.testing.assert_allclose(result['positions'][-1], expected, rtol=0, atol=2e-11)
        np.testing.assert_array_equal(x, x0)
        np.testing.assert_array_equal(v, v0)

    def test_absolute_time_and_internal_substeps(self):
        called = []
        def acceleration(t, state):
            called.append(t)
            return np.ones_like(state)*t
        result = rollout(acceleration, [[0., 0.]], [[0., 0.]],
                         t_launch=2., duration=.005)
        self.assertEqual(len(called), 3)
        self.assertTrue(all(2 < t < 2.005 for t in called))
        self.assertFalse(all(abs(t/.005-round(t/.005)) < 1e-9 for t in called))
        self.assertGreater(float(result['positions'][-1, 0, 0]), 0.)

    def test_harmonic_reference_and_refinement(self):
        omega = 1.3
        def acceleration(t, state):
            return -omega*omega*state
        coarse = rollout(acceleration, [[1., 0.]], [[0., 1.]], duration=10.)
        fine = diagnostic_rollout(acceleration, [[1., 0.]], [[0., 1.]], duration=10., dt=.0025)
        exact = np.array([np.cos(omega*10), np.sin(omega*10)/omega])
        e0 = np.linalg.norm(coarse['positions'][-1, 0]-exact)
        e1 = np.linalg.norm(fine['positions'][-1, 0]-exact)
        self.assertLess(e0, 1e-7)
        self.assertLess(e1, e0/8)

    def test_grid_rejects_silent_rounding(self):
        np.testing.assert_array_equal(sample_indices([0., .1+.2, .305]), [0, 60, 61])
        for value in [[.003], [0., 0.], [.1, 0.], [-.005], [float('nan')], []]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    sample_indices(value)
        with self.assertRaises(ValueError):
            rollout(lambda t, x: x*0, [[0., 0.]], [[0., 0.]], t_launch=.003)

    def test_formal_step_cannot_be_selected_by_argument(self):
        with self.assertRaises(TypeError):
            rollout(lambda t, x: x*0, [[0., 0.]], [[0., 0.]], dt=.01)

    def test_invalid_states_and_accelerations(self):
        zero = lambda t, x: np.zeros_like(x)
        with self.assertRaises(ValueError):
            rollout(zero, [[1., 2.]], [[0., 0.], [1., 1.]])
        with self.assertRaises(ValueError):
            rollout(zero, [[float('nan'), 0.]], [[0., 0.]])
        with self.assertRaises(ValueError):
            rollout(lambda t, x: 0., [[1., 2.]], [[0., 0.]])
        with self.assertRaises(FloatingPointError):
            rollout(lambda t, x: np.full_like(x, np.inf), [[1., 2.]], [[0., 0.]])

    def test_zero_duration(self):
        result = rollout(lambda t, x: x*0, [[1., 2.]], [[.1, .2]], duration=0.)
        self.assertEqual(result['positions'].shape, (1, 1, 2))
        np.testing.assert_array_equal(result['times'], [0.])


if __name__ == '__main__':
    unittest.main()
