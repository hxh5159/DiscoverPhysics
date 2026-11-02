"""Synthetic transport/agent checks, never an actual language-model benchmark."""
import base64
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from development_worlds import catalog
from development_worlds.console import DevelopmentConsole
from development_worlds.service import DevelopmentSession, serve_session
from rift import Console
from rift.console import ConsoleError
from shore_agent.agent import DiscoveryAgent, parse_action, compact
from shore_agent.tools import candidate_tool

ROOT = Path(__file__).resolve().parents[1]
LAW = '''import numpy as np
LAW_API_VERSION = '1.0.0'
PARAMS = {'k': 0.02}
def acceleration(t, bodies, params):
    out = np.zeros((len(bodies),2), dtype=float)
    for i,b in enumerate(bodies):
        if b['kind'] == 'probe':
            out[i] = -params['k'] * b['pos']
    return out
'''
CLAIMS = {'dynamics_order': 2, 'hidden_matter': False, 'time_dependence': False}


class ScriptedModel:
    capability_label = 'synthetic_protocol_test_not_model_capability'

    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def complete(self, messages, *, max_tokens, timeout):
        self.calls.append((messages, max_tokens, timeout))
        return json.dumps(next(self.actions))


class RunnableBaselineTests(unittest.TestCase):
    def console(self, clock=None):
        kw = {} if clock is None else {'clock': clock}
        return DevelopmentConsole('2', catalog.noise_sigma('2'), 0, catalog.config('2'), **kw)

    def test_mock_agent_uses_unchanged_console_transport_and_atomic_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            session = DevelopmentSession(self.console(), directory/'session')
            with serve_session(session) as auth:
                client = Console(auth['endpoint'], auth['token'], timeout=5)
                request = catalog.first_light('2')
                model = ScriptedModel([
                    {'action':'observe','times':[0,.05,.1]},
                    {'action':'launch', **request},
                    {'action':'check','law':LAW},
                    {'action':'draft','law':LAW,'claims':CLAIMS},
                    {'action':'submit','law':LAW,'claims':CLAIMS}])
                outcome = DiscoveryAgent(client, model, directory/'agent', max_turns=8).run()
                self.assertEqual(outcome['status'], 'submitted')
                self.assertEqual(outcome['model_transport'], model.capability_label)
                self.assertEqual(len(model.calls), 5)
                self.assertTrue(all(0 < call[2] <= 300 for call in model.calls))
                self.assertTrue(session.closed)
                self.assertEqual(session.console.budget['energy_used'], 2)
                self.assertEqual(session.console.budget['observe_used'], 1)
                self.assertIsNotNone(session.latest)
                self.assertEqual((Path(session.latest['artifact_directory'])/'law.py').read_text(), LAW)
                self.assertTrue((directory/'agent/law.py').exists())
                with self.assertRaises(ConsoleError):
                    client.mission()

    def test_invalid_pair_and_late_pair_never_replace_accepted_draft(self):
        now = [0.]
        with tempfile.TemporaryDirectory() as folder:
            session = DevelopmentSession(self.console(clock=lambda: now[0]), Path(folder)/'session')
            session.dispatch('mission', {})
            pair = {'law_b64': base64.b64encode(LAW.encode()).decode(),
                    'claims_b64': base64.b64encode(json.dumps(CLAIMS).encode()).decode()}
            session.dispatch('save_draft', pair)
            previous = copy.deepcopy(session.latest)
            bad = {**pair, 'claims_b64': base64.b64encode(b'{}').decode()}
            with self.assertRaises(Exception):
                session.dispatch('submit', bad)
            self.assertFalse(session.closed)
            self.assertEqual(session.latest, previous)
            with patch('development_worlds.service.validate_source', side_effect=lambda source: now.__setitem__(0, 600.)):
                with self.assertRaises(TimeoutError):
                    session.dispatch('submit', pair)
            self.assertEqual(session.latest, previous)
            self.assertTrue(session.snapshot()['expired'])

    def test_parameter_fit_uses_fixed_candidate_integrator_on_supplied_data(self):
        console = self.console()
        mission = console.mission()
        record = copy.deepcopy(mission['first_light'])
        times = np.asarray(record['readout'])[:,None,None]
        pos = np.asarray([p['pos'] for p in record['probes']])[None,:,:]
        vel = np.asarray([p['vel'] for p in record['probes']])[None,:,:]
        omega = np.sqrt(.06)
        record['positions'] = (pos*np.cos(omega*times)+vel*np.sin(omega*times)/omega).tolist()
        before = candidate_tool(LAW, mission, [record], timeout=30)
        fitted = candidate_tool(LAW, mission, [record], fit=True, timeout=30, max_nfev=12)
        self.assertTrue(before['ok'], before)
        self.assertTrue(fitted['ok'], fitted)
        self.assertLess(fitted['observed_position_mse'], before['observed_position_mse']*1e-4)
        self.assertAlmostEqual(fitted['parameters']['k'], .06, places=5)
        global_law = LAW.replace("params['k']", "PARAMS['k']")
        global_fit = candidate_tool(global_law, mission, [record], fit=True, timeout=30, max_nfev=12)
        self.assertTrue(global_fit['ok'], global_fit)
        self.assertAlmostEqual(global_fit['parameters']['k'], .06, places=5)

    def test_late_model_result_is_rejected_without_artifacts(self):
        now = [0.]
        console = self.console(clock=lambda: now[0])

        class Client:
            def mission(self): return console.mission()

        class Late:
            def complete(self, *args, **kwargs):
                now[0] = 601.
                return json.dumps({'action':'submit','law':LAW,'claims':CLAIMS})

        with tempfile.TemporaryDirectory() as folder:
            result = DiscoveryAgent(Client(), Late(), folder, clock=lambda: now[0]).run()
            self.assertFalse(result['has_saved_pair'])
            self.assertFalse((Path(folder)/'law.py').exists())
            self.assertTrue(json.loads((Path(folder)/'events.json').read_text())[0]['late_response_rejected'])

    def test_parsing_never_executes_model_text(self):
        self.assertEqual(parse_action('<think>reason</think>```json\n{"action":"archive"}\n```'), {'action':'archive'})
        with self.assertRaises(ValueError):parse_action('')
        with self.assertRaises(ValueError):parse_action('{"action":"archive"}{"action":"archive"}')

    def test_context_sampling_and_full_record_pages_keep_time_axes_aligned(self):
        record = {'times': list(range(25)), 'readout': list(range(25)),
                  'positions': [[[i, 0]] for i in range(25)], 'velocities': [[[1,0]] for _ in range(25)]}
        shown = compact(record)
        self.assertEqual(shown['times'], shown['readout'])
        self.assertEqual([value[0][0] for value in shown['positions']], shown['times'])
        with tempfile.TemporaryDirectory() as folder:
            agent = DiscoveryAgent(object(), object(), folder, clock=lambda: 0.)
            agent.deadline = 600.
            agent.records = [record]
            page = agent._action({'action':'view_record','index':0,'start':10,'count':10})
            self.assertEqual(page['sample_indices'], list(range(10,20)))
            self.assertEqual(page['record']['times'], list(range(10,20)))
            self.assertEqual(page['record']['readout'], list(range(10,20)))
            self.assertEqual(page['full_time_count'], 25)


if __name__ == '__main__':unittest.main()
