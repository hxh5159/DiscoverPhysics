"""Protocol and independent mathematical checks for public design prototypes."""
import unittest
import copy
from unittest.mock import patch
import numpy as np
from development_worlds.worlds import DevelopmentSimulator
from development_worlds import catalog
from development_worlds.console import DevelopmentConsole
from development_worlds.evaluation import prediction_score, score_components, claims_truth

class DevelopmentTests(unittest.TestCase):
    def test_harmonic_solution_and_batch_independence(self):
        sim=DevelopmentSimulator('2')
        p=np.array([[2.,0.],[0.,3.]])
        v=np.array([[0.,.4],[-.3,0.]])
        result=sim.integrate(p,v,duration=1.,source_properties={'p1':1.},probe_properties={'p2':1.})
        omega=np.sqrt(.06)
        expected=p*np.cos(omega)+v*np.sin(omega)/omega
        np.testing.assert_allclose(result['positions'][-1],expected,atol=1e-11,rtol=0)
        solo=sim.integrate(p[:1],v[:1],duration=1.,source_properties={'p1':1.},probe_properties={'p2':1.})
        np.testing.assert_array_equal(result['positions'][:,:1],solo['positions'])

    def test_rotating_field_period(self):
        sim=DevelopmentSimulator('6')
        kwargs={'probe_properties':{'mass':1.}}
        a=sim.acceleration(.25,[[2.,1.]],**kwargs)
        np.testing.assert_allclose(a,sim.acceleration(16.25,[[2.,1.]],**kwargs),atol=1e-14)
        self.assertGreater(np.linalg.norm(a-sim.acceleration(8.25,[[2.,1.]],**kwargs)),.1)

    def test_console_noise_quota_and_grid(self):
        for world in catalog.WORLDS:
            with self.subTest(world=world):
                cfg=catalog.config(world)
                con=DevelopmentConsole(world,catalog.noise_sigma(world),7,cfg)
                mission=con.mission()
                self.assertEqual(con.budget['energy_used'],0)
                self.assertEqual(np.asarray(mission['first_light']['positions']).shape,(8,2,2))
                a=con.launch(**cfg['first_light']);b=con.launch(**cfg['first_light'])
                self.assertEqual(con.budget['energy_used'],4)
                self.assertNotEqual(a['positions'],b['positions'])
                self.assertNotEqual(a['velocities'],b['velocities'])
                self.assertEqual(con.archive()['observations'],con.archive()['observations'])
                with self.assertRaises(ValueError):con.observe([.001])
                self.assertEqual(con.budget['observe_used'],0)

    def test_scoring_threshold(self):
        self.assertEqual(prediction_score(0.,1.),80.)
        self.assertEqual(prediction_score(.1,1.),40.)
        truth=claims_truth('6')
        self.assertEqual(score_components(40.,truth,truth,2)['saving'],4.)
        self.assertEqual(score_components(39.99,truth,truth,2)['saving'],0.)
        self.assertEqual(score_components(80.,truth,truth,0)['total'],100.)
        self.assertEqual(score_components(80.,truth,truth,10)['saving'],0.)
        wrong = {'dynamics_order': 1, 'hidden_matter': True, 'time_dependence': False}
        self.assertEqual(score_components(40.,wrong,truth,4)['saving'],3.)
        for energy in (-1, 11, 100, float('inf'), float('nan')):
            with self.subTest(energy=energy), self.assertRaises(ValueError):
                score_components(80.,truth,truth,energy)
        with self.assertRaises(ValueError):score_components(80.,{**truth,'hidden_matter':0},truth,2)

    def test_ten_probe_budget_is_enforced_before_simulation_without_reset(self):
        for world in catalog.WORLDS:
            with self.subTest(world=world):
                cfg = catalog.config(world)
                con = DevelopmentConsole(world, catalog.noise_sigma(world), 7, cfg, clock=lambda: 0.)
                mission = con.mission()
                self.assertEqual(mission['limits']['energy_budget'], 10)
                self.assertEqual(len(mission['first_light']['probes']), 2)
                self.assertEqual(con.budget['energy_used'], 0)
                request = copy.deepcopy(cfg['first_light'])
                request['readout'] = [0., .005]
                request['probes'] *= 4
                con.launch(**request)  # Eight is still the per-call cap.
                self.assertEqual(con.budget['energy_remaining'], 2)
                request['probes'] = request['probes'][:3]
                with patch.object(con._sim, 'launch') as simulate, self.assertRaisesRegex(RuntimeError, 'Insufficient energy'):
                    con.launch(**request)
                simulate.assert_not_called()
                self.assertEqual(con.budget['energy_used'], 8)
                request['probes'] = request['probes'][:2]
                con.launch(**request)
                self.assertEqual(con.budget['energy_remaining'], 0)
                request['probes'] = request['probes'][:1]
                with patch.object(con._sim, 'launch') as simulate, self.assertRaisesRegex(RuntimeError, 'Insufficient energy'):
                    con.launch(**request)  # The eleventh probe is rejected.
                simulate.assert_not_called()
                self.assertEqual(con.mission()['budget']['energy_used'], 10)
                con.observe([0.])  # Its independent free allowance is retained.
                self.assertEqual(con.budget['observe_used'], 1)
                self.assertEqual(con.budget['energy_used'], 10)

    def test_console_omitted_limits_use_current_defaults(self):
        cfg = catalog.config('2')
        del cfg['energy_budget'], cfg['wall_clock_seconds']
        con = DevelopmentConsole('2', 0., 0, cfg, clock=lambda: 0.)
        limits = con.mission()['limits']
        self.assertEqual(limits['energy_budget'], 10)
        self.assertEqual(limits['wall_clock_seconds'], 300.)

if __name__=='__main__':unittest.main()
