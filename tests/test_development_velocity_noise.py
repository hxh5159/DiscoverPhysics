"""Instrument checks for noisy observations and clean evaluation references."""
import copy
import unittest

import numpy as np

from development_worlds import catalog
from development_worlds.console import DevelopmentConsole
from development_worlds.evaluation import prediction_score
from development_worlds.worlds import DevelopmentSimulator


def console(world, seed=31):
    return DevelopmentConsole(world, catalog.noise_sigma(world), seed,
                              catalog.config(world), clock=lambda: 0.)


class DevelopmentVelocityNoiseTests(unittest.TestCase):
    def test_standard_scale_known_inputs_and_clean_simulator(self):
        for world in catalog.WORLDS:
            with self.subTest(world=world):
                cfg = catalog.config(world)
                request = cfg['first_light']
                sim = DevelopmentSimulator(world)
                clean = sim.launch(**request)
                con = console(world)
                mission = con.mission()
                first = mission['first_light']
                self.assertEqual(mission['noise']['noise_version'], '1.1.0')
                self.assertEqual(mission['noise']['velocity_sigma_fraction_of_initial_speed_max'],
                                 1e-4)
                self.assertEqual(mission['noise']['velocity_sigma'],
                                 1e-4 * cfg['initial_speed_max'])
                self.assertEqual(first['probes'], request['probes'])
                self.assertEqual(mission['background_initial_state']['velocities'],
                                 sim.background_initial_state()['velocities'].tolist())
                self.assertNotEqual(first['velocities'], clean['velocities'].tolist())
                # No noise is fed back to the simulator or a scoring reference.
                again = sim.launch(**request)
                np.testing.assert_array_equal(clean['positions'], again['positions'])
                np.testing.assert_array_equal(clean['velocities'], again['velocities'])
                self.assertEqual(prediction_score(0., 1.), 80.)

    def test_position_noise_stream_matches_previous_instrument(self):
        seed = 19
        for world in catalog.WORLDS:
            with self.subTest(world=world):
                cfg = catalog.config(world)
                request = cfg['first_light']
                sigma = catalog.noise_sigma(world)
                rng = np.random.default_rng(seed)
                sim = DevelopmentSimulator(world)
                clean = sim.launch(**request)
                background = sim.background(clean['times'])
                expected_first = clean['positions'] + rng.normal(
                    0., sigma, size=clean['positions'].shape)
                expected_background = background['positions'] + rng.normal(
                    0., sigma, size=background['positions'].shape)
                con = console(world, seed)
                first = con.mission()['first_light']
                np.testing.assert_array_equal(first['positions'], expected_first)
                np.testing.assert_array_equal(first['visible_background_positions'],
                                              expected_background)
                times = [0., .005, .01]
                bg = sim.background(times)
                expected_observe = bg['positions'] + rng.normal(
                    0., sigma, size=bg['positions'].shape)
                np.testing.assert_array_equal(con.observe(times)['positions'], expected_observe)
                for _ in range(2):
                    expected = clean['positions'] + rng.normal(
                        0., sigma, size=clean['positions'].shape)
                    np.testing.assert_array_equal(con.launch(**request)['positions'], expected)

    def test_first_light_launch_archive_and_transcript_share_noisy_record(self):
        for world in catalog.WORLDS:
            with self.subTest(world=world):
                con = console(world)
                request = catalog.first_light(world)
                mission = con.mission()
                first = copy.deepcopy(mission['first_light'])
                self.assertEqual(con.mission()['first_light'], first)
                self.assertEqual(con.archive()['first_light'], first)
                launched = con.launch(**request)
                velocity = copy.deepcopy(launched['velocities'])
                self.assertNotEqual(velocity, first['velocities'])
                self.assertEqual(con.archive()['observations'][0]['velocities'], velocity)
                self.assertEqual(con.archive()['observations'][0]['velocities'], velocity)
                launch_entries = [e for e in con.public_transcript if e['operation'] == 'launch']
                self.assertEqual(launch_entries[0]['response']['velocities'], velocity)
                self.assertEqual(con.budget['energy_used'], 2)
                # External mutations cannot corrupt cached data or the transcript.
                launched['velocities'][1][0][0] = 123456.
                mission['first_light']['velocities'][1][0][0] = 123456.
                self.assertEqual(con.archive()['observations'][0]['velocities'], velocity)
                self.assertEqual(con.archive()['first_light'], first)

    def test_cache_reads_do_not_draw_noise_but_new_experiments_do(self):
        for world in catalog.WORLDS:
            with self.subTest(world=world):
                request = catalog.first_light(world)
                a, b = console(world), console(world)
                self.assertEqual(a.mission(), b.mission())
                first = a.launch(**request)
                self.assertEqual(first, b.launch(**request))
                for _ in range(3):
                    a.mission()
                    a.archive()
                second = a.launch(**request)
                self.assertEqual(second, b.launch(**request))
                self.assertNotEqual(first['velocities'], second['velocities'])

    def test_zero_command_velocity_does_not_disable_sensor_noise(self):
        request = copy.deepcopy(catalog.first_light('2'))
        request['probes'] = request['probes'][:1]
        request['probes'][0]['vel'] = [0., 0.]
        request['readout'] = [0., .005, .01]
        con = console('2')
        con.mission()
        result = con.launch(**request)
        self.assertEqual(result['probes'][0]['vel'], [0., 0.])
        self.assertNotEqual(result['velocities'][0][0], [0., 0.])

    def test_velocity_noise_is_zero_mean_independent_and_uses_public_scale(self):
        # Enough sensor samples to reject a constant trajectory bias or a shared
        # coordinate/time draw, with deterministic and generous statistical bounds.
        for world in ('2', '4'):
            with self.subTest(world=world):
                con = console(world, seed=517)
                sigma = 1e-4 * catalog.config(world)['initial_speed_max']
                errors = np.asarray(con._noise_velocities(np.zeros((50000, 2))))
                self.assertLess(float(np.max(np.abs(errors.mean(axis=0)))), .02 * sigma)
                np.testing.assert_allclose(errors.std(axis=0), sigma, rtol=.02)
                self.assertLess(abs(float(np.corrcoef(errors.T)[0, 1])), .02)
                self.assertLess(abs(float(np.corrcoef(errors[:-1, 0], errors[1:, 0])[0, 1])), .02)


if __name__ == '__main__':
    unittest.main()
