"""Public interface regressions using only elementary synthetic laws."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from rift import rollout, validate_case, ValidationError
from rift_numerics import rollout as position_rollout, state_rollout


def body(identifier, pos=(0., 0.), vel=(0., 0.), properties=None):
    return {"id": identifier, "pos": list(pos), "vel": list(vel),
            "properties": {} if properties is None else properties}


def case(*, background=None, probes=None, launch=0., readout=None):
    return {"schema_version": "1.0.0", "t_launch": launch,
            "readout": [0., .5, 1.] if readout is None else readout,
            "background": [] if background is None else background,
            "probes": [body("P0", (1., 2.), (.2, -.3))] if probes is None else probes}


def law(callback=None, *, params=None, body_props=None, hidden=None):
    return SimpleNamespace(params={} if params is None else params,
        body_props={} if body_props is None else body_props,
        hidden=[] if hidden is None else hidden,
        acceleration=callback or (lambda t, bodies, params: np.zeros((len(bodies), 2))))


class LawRolloutTests(unittest.TestCase):
    def assert_path(self, callback, path):
        with self.assertRaises(ValidationError) as captured:
            callback()
        self.assertTrue(captured.exception.path.startswith(path), captured.exception)

    def test_nonzero_launch_and_output_order(self):
        c = case(launch=2., probes=[body("Q", (1, 0), (2, .5)), body("P", (-2, 3), (-1, 0))])
        original = copy.deepcopy(c)
        result = rollout(law(), c)
        tau = np.array(c['readout'])[:, None, None]
        expected = np.array([b['pos'] for b in c['probes']])[None]+tau*np.array([b['vel'] for b in c['probes']])[None]
        np.testing.assert_allclose(result['positions'], expected, rtol=0, atol=2e-12)
        np.testing.assert_array_equal(result['times'], [2., 2.5, 3.])
        self.assertEqual(result['probe_ids'], ['Q', 'P'])
        self.assertEqual(result['positions'].dtype, np.float64)
        self.assertEqual(c, original)

    def test_float32_and_sequence_output_are_valid(self):
        for factory in (lambda x: x.tolist(), lambda x: tuple(tuple(y) for y in x), lambda x: x.astype(np.float32)):
            acceleration = lambda t, b, p: factory(np.tile([.1, -.2], (len(b), 1)))
            c = case()
            result = rollout(law(acceleration), c)
            expected = np.array(c['probes'][0]['pos'])+np.array(c['probes'][0]['vel'])+np.array([.05, -.1])
            np.testing.assert_allclose(result['positions'][-1, 0], expected, rtol=0, atol=3e-9)

    def test_probe_batch_isolation_and_reordering(self):
        seen = []
        def acceleration(t, bodies, params):
            n = sum(b['kind'] == 'probe' for b in bodies)
            seen.append(n)
            return [[n if b['kind'] == 'background' else bodies[0]['pos'][0], 0.] for b in bodies]
        c = case(background=[body('N', (2., 0.))], probes=[body('P0'), body('P1', (4, 0))])
        combined = rollout(law(acceleration), c)
        self.assertEqual(set(seen), {0, 1})
        for i, probe in enumerate(c['probes']):
            one = rollout(law(acceleration), {**c, 'probes': [probe]})
            np.testing.assert_array_equal(combined['positions'][:, i], one['positions'][:, 0])
        reordered = rollout(law(acceleration), {**c, 'probes': list(reversed(c['probes']))})
        np.testing.assert_array_equal(reordered['positions'], combined['positions'][:, ::-1])
        # Background acceleration saw zero probes, so N stays at x=2.
        np.testing.assert_allclose(combined['positions'][-1, 0], [1., 0.], rtol=0, atol=2e-12)

    def test_moving_background_shared_substeps(self):
        launch, duration = 2., 1.
        source_acceleration = .6
        def acceleration(t, bodies, params):
            source = next(b for b in bodies if b['id'] == 'N')
            return [[source_acceleration, 0.] if b['id'] == 'N' else [source['pos'][0], 0.] for b in bodies]
        c = case(launch=launch, background=[body('N', (1., 0.), (.2, 0.))],
                 probes=[body('P', (3., 0.), (.1, 0.))], readout=[duration])
        result = rollout(law(acceleration), c)
        at_launch = 1.+.2*launch+.5*source_acceleration*launch**2
        velocity_at_launch = .2+source_acceleration*launch
        expected = 3.+.1*duration+.5*at_launch*duration**2+velocity_at_launch*duration**3/6+source_acceleration*duration**4/24
        np.testing.assert_allclose(result['positions'][0, 0, 0], expected, rtol=0, atol=2e-9)

    def test_hidden_bodies_and_estimated_properties(self):
        captured = []
        def acceleration(t, bodies, params):
            captured.append([(b['id'], b['kind']) for b in bodies])
            strength = bodies[0]['properties']['strength']
            hidden = next(b for b in bodies if b['kind'] == 'hidden')
            return [[strength+hidden['pos'][0], 0.] if b['kind'] == 'probe' else [0., 0.] for b in bodies]
        c = case(background=[body('N')], probes=[body('P')])
        model = law(acceleration, body_props={'N': {'strength': 2.}}, hidden=[body('H0', (1., 0.))])
        result = rollout(model, c)
        np.testing.assert_allclose(result['positions'][-1, 0], [1.5, 0.], rtol=0, atol=2e-12)
        self.assertEqual(captured[0], [('N', 'background'), ('H0', 'hidden')])
        self.assertIn([('N', 'background'), ('H0', 'hidden'), ('P', 'probe')], captured)

    def test_known_property_precedence_and_conflicts(self):
        c = case(background=[body('N', properties={'mass': 2.})])
        rollout(law(body_props={'N': {'mass': 2., 'estimate': 3.}}), c)
        self.assert_path(lambda: rollout(law(body_props={'N': {'mass': 3.}}), c), '$.law.BODY_PROPS.N.mass')
        self.assert_path(lambda: rollout(law(body_props={'missing': {}}), c), '$.law.BODY_PROPS')
        self.assert_path(lambda: rollout(law(hidden=[body('P0')]), c), '$.law.HIDDEN')

    def test_hidden_ids_need_uniqueness_not_a_prefix(self):
        c = case()
        result = rollout(law(hidden=[body('estimated-source')]), c)
        self.assertEqual(result['positions'].shape, (3, 1, 2))

    def test_case_metadata_rejected_before_callback(self):
        calls = []
        model = law(lambda t, b, p: calls.append(t))
        for field in ('truth', 'name', 'group'):
            self.assert_path(lambda: rollout(model, {**case(), field: 'private'}), '$.case')
        self.assertFalse(calls)

    def test_strict_time_grid_and_representation_tolerance(self):
        normalized = validate_case(case(launch=.1+.2, readout=[0., .1+.2, .305]))
        self.assertEqual(normalized['t_launch'], .3)
        for readout in ([.003], [0., 0.], [.1, .05], [-.005], [float('nan')], []):
            self.assert_path(lambda: validate_case(case(readout=readout)), '$.case.readout')
        self.assert_path(lambda: validate_case(case(launch=.003)), '$.case.t_launch')

    def test_callback_receives_actual_stage_velocity(self):
        seen = []
        def acceleration(t, bodies, params):
            seen.append(float(bodies[-1]['vel'][0]))
            return [[1., 0.] for b in bodies]
        rollout(law(acceleration), case(probes=[body('P')], readout=[.01]))
        self.assertEqual(seen[0], 0.)
        self.assertGreater(max(seen), 0.)
        self.assertGreater(len(set(seen)), 3)

    def test_readonly_body_params_and_original_inputs(self):
        c = case(); original = copy.deepcopy(c)
        actions = [lambda b, p: b[0]['pos'].__setitem__(0, 99),
                   lambda b, p: b[0]['pos'].setflags(write=True),
                   lambda b, p: b[0]['properties'].__setitem__('x', 99),
                   lambda b, p: p.__setitem__('k', 99)]
        for action in actions:
            def bad(t, bodies, params):
                action(bodies, params)
                return [[0., 0.]]
            self.assert_path(lambda: rollout(law(bad, params={'k': 1.}), c), '$.law.acceleration')
            self.assertEqual(c, original)

    def test_invalid_callback_outputs(self):
        bad = [0., [0., 0.], [[True, False]], [[1+0j, 0j]], [['0', '0']],
               [[float('nan'), 0.]], [[float('inf'), 0.]]]
        for output in bad:
            self.assert_path(lambda: rollout(law(lambda t, b, p: output), case()), '$.law.acceleration.return')

    def test_mixed_sequence_booleans_are_rejected_before_numpy_promotion(self):
        outputs = [
            [[True, 0.0]],
            ((0.0, False),),
            [[np.bool_(True), np.float32(.2)]],
            [(np.float64(.1), np.bool_(False))],
            [np.array([True, False], dtype=np.bool_)],
        ]
        for output in outputs:
            with self.subTest(output=repr(output)):
                self.assert_path(lambda: rollout(law(lambda t, b, p: output), case()),
                                 '$.law.acceleration.return')

    def test_mixed_real_sequence_dtypes_remain_valid(self):
        outputs = [[[1, np.float32(-.2)]], ((np.float64(1), -.2),)]
        c = case()
        for output in outputs:
            with self.subTest(output=repr(output)):
                result = rollout(law(lambda t, b, p: output), c)
                self.assertEqual(result['positions'].dtype, np.float64)
                expected = (np.array(c['probes'][0]['pos'])+np.array(c['probes'][0]['vel'])
                            +.5*np.array(output[0], dtype=np.float64))
                np.testing.assert_allclose(result['positions'][-1, 0], expected, rtol=0, atol=2e-12)

    def test_more_than_instrument_batch_is_valid_prediction_case(self):
        c = case(probes=[body(f'P{i}') for i in range(21)], readout=(np.arange(51)*.005).tolist())
        result = rollout(law(), c)
        self.assertEqual(result['positions'].shape, (51, 21, 2))

    def test_resource_guards_fail_before_execution(self):
        calls = []
        model = law(lambda t, b, p: calls.append(t))
        self.assert_path(lambda: rollout(model, case(), max_steps=1), '$.limits.max_steps')
        self.assert_path(lambda: rollout(model, case(), max_state_records=1), '$.limits.max_state_records')
        self.assertFalse(calls)

    def test_position_only_old_and_new_entry_are_bitwise_equal(self):
        x = [[1., .2], [.7, -.3]]; v = [[.1, .2], [.3, .4]]
        acceleration = lambda t, p: -.17*p+np.array([.02*np.sin(t), .03*np.cos(t)])
        old = position_rollout(acceleration, x, v, t_launch=.7, duration=2.)
        new = state_rollout(lambda t, p, v: acceleration(t, p), x, v, t_launch=.7, duration=2.)
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])


if __name__ == '__main__':
    unittest.main()
