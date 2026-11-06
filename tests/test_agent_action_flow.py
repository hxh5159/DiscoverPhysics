"""Exercise candidate ownership, bounded tool batches and actual handoff bytes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from development_worlds import catalog
from development_worlds.console import DevelopmentConsole
from development_worlds.service import DevelopmentSession, serve_session
from rift import Console
from agent.run import DiscoveryAgent, MAX_BATCH_ACTIONS
from agent.models import InferenceError

LAW = """import numpy as np
LAW_API_VERSION = '1.0.0'
PARAMS = {'k': 0.02}
def acceleration(t, bodies, params):
    return np.asarray([-params['k'] * body['pos'] if body['kind'] == 'probe'
                       else np.zeros(2) for body in bodies])
"""
FITTED = LAW.replace("0.02", "0.06")
OTHER = LAW.replace("0.02", "0.04")
CLAIMS = {'dynamics_order': 2, 'hidden_matter': False, 'time_dependence': False}
OTHER_CLAIMS = {**CLAIMS, 'time_dependence': True}


class Model:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def complete(self, messages, **limits):
        self.calls.append(messages)
        self.last_call_diagnostic = {'status': 'completed', 'finish_reason': 'stop',
                                     'usage': {'completion_tokens': 20}}
        return json.dumps(next(self.actions))


class AgentActionFlowTests(unittest.TestCase):
    def direct(self, root):
        console = Mock()
        console.save_draft.return_value = {'operation': 'save_draft'}
        console.submit.return_value = {'operation': 'submit'}
        agent = DiscoveryAgent(console, object(), root, clock=lambda: 0.)
        agent.deadline, agent.mission = 600., {}
        return agent, console

    def session(self, root):
        console = DevelopmentConsole('2', catalog.noise_sigma('2'), 0, catalog.config('2'))
        return DevelopmentSession(console, root/'session')

    def test_last_turn_fit_is_the_exact_candidate_automatically_submitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.session(root)
            model = Model([{'action': 'archive'},
                           {'action': 'draft', 'law': LAW, 'claims': CLAIMS},
                           {'action': 'fit', 'law_ref': 'saved'}])
            with serve_session(session) as auth, patch('agent.run.candidate_tool',
                    return_value={'ok': True, 'law': FITTED, 'parameters': {'k': .06}}) as fit:
                result = DiscoveryAgent(Console(auth['endpoint'], auth['token']), model, root/'agent').run()
            self.assertEqual(len(model.calls), 3)
            self.assertEqual(result['status'], 'submitted_after_optional_turn_limit')
            self.assertEqual(fit.call_args.args[0], LAW)
            artifact = Path(session.latest['artifact_directory'])
            self.assertEqual((artifact/'law.py').read_text(), FITTED)
            self.assertEqual(json.loads((artifact/'claims.json').read_text()), CLAIMS)
            events = json.loads((root/'agent/events.json').read_text())
            self.assertTrue(events[-1]['result']['saved_candidate'])
            self.assertEqual(events[-1]['generation']['finish_reason'], 'stop')

    def test_explicit_claims_take_precedence_for_matching_and_different_laws(self):
        for source in (LAW, OTHER):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                agent, console = self.direct(directory)
                agent._pair(LAW, CLAIMS)
                with patch('agent.run.candidate_tool', return_value={'ok': True, 'law': FITTED}):
                    result = agent._action({'action': 'fit', 'law': source, 'claims': OTHER_CLAIMS})
                self.assertTrue(result['saved_candidate'])
                self.assertEqual(agent.latest, {'law': FITTED, 'claims': OTHER_CLAIMS})
                self.assertEqual(console.save_draft.call_count, 2)

    def test_different_or_unsaved_law_without_claims_never_borrows_old_claims(self):
        for save_first in (False, True):
            with self.subTest(save_first=save_first), tempfile.TemporaryDirectory() as directory:
                agent, console = self.direct(directory)
                if save_first: agent._pair(LAW, CLAIMS)
                previous = copy.deepcopy(agent.latest)
                calls = console.save_draft.call_count
                with patch('agent.run.candidate_tool', return_value={'ok': True, 'law': FITTED}):
                    result = agent._action({'action': 'fit', 'law': OTHER})
                self.assertFalse(result['saved_candidate'])
                self.assertEqual(result['law'], FITTED)
                self.assertEqual(agent.latest, previous)
                self.assertEqual(console.save_draft.call_count, calls)

    def test_failed_fit_and_invalid_explicit_claims_keep_accepted_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, console = self.direct(directory)
            agent._pair(LAW, CLAIMS)
            with patch('agent.run.candidate_tool', return_value={'ok': False, 'error': 'fit failed'}):
                self.assertFalse(agent._action({'action': 'fit', 'law_ref': 'saved'})['ok'])
            with patch('agent.run.candidate_tool') as fit, self.assertRaises(ValueError):
                agent._action({'action': 'fit', 'law_ref': 'saved', 'claims': {}})
            fit.assert_not_called()
            self.assertEqual(agent.latest, {'law': LAW, 'claims': CLAIMS})
            self.assertEqual(console.save_draft.call_count, 1)

    def test_saved_reference_requires_a_pair_and_is_mutually_exclusive_with_source(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, _ = self.direct(directory)
            for action in ({'action': 'check', 'law_ref': 'saved'},
                           {'action': 'fit', 'law_ref': 'saved', 'law': LAW},
                           {'action': 'submit', 'law_ref': 'unknown'}):
                with self.subTest(action=action), patch('agent.run.candidate_tool') as tool:
                    with self.assertRaises(ValueError): agent._action(action)
                    tool.assert_not_called()

    def test_three_decisions_batch_real_experiments_fit_and_submit_without_extra_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.session(root)
            model = Model([
                {'action': 'batch', 'actions': [{'action': 'observe', 'times': [0., .05]},
                    {'action': 'launch', **catalog.first_light('2')}]},
                {'action': 'batch', 'actions': [{'action': 'draft', 'law': LAW, 'claims': CLAIMS},
                    {'action': 'fit', 'law_ref': 'saved'}]},
                {'action': 'batch', 'actions': [{'action': 'check', 'law_ref': 'saved'},
                    {'action': 'submit', 'law_ref': 'saved'}]}])
            with serve_session(session) as auth, patch('agent.run.candidate_tool',
                    return_value={'ok': True, 'law': FITTED}) as tools:
                result = DiscoveryAgent(Console(auth['endpoint'], auth['token']), model, root/'agent').run()
            self.assertEqual(result['status'], 'submitted')
            self.assertEqual(len(model.calls), 3)
            self.assertEqual(session.console.budget['energy_used'], 2)
            self.assertEqual(session.console.budget['observe_used'], 1)
            self.assertEqual([call.args[0] for call in tools.call_args_list], [LAW, FITTED])
            artifact = Path(session.latest['artifact_directory'])
            self.assertEqual((artifact/'law.py').read_text(), FITTED)
            candidates = sorted((root/'agent').glob('candidate-*.py'))
            self.assertEqual(len(candidates), 3)
            self.assertEqual(candidates[0].read_text(), LAW)
            self.assertEqual(candidates[1].read_text(), FITTED)

    def test_batch_error_stops_following_actions_and_keeps_latest_valid_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, console = self.direct(directory)
            with patch('agent.run.candidate_tool', return_value={'ok': True, 'law': FITTED}):
                result = agent._action({'action': 'batch', 'actions': [
                    {'action': 'draft', 'law': LAW, 'claims': CLAIMS},
                    {'action': 'fit', 'law_ref': 'saved'},
                    {'action': 'draft', 'law': 'invalid source', 'claims': CLAIMS},
                    {'action': 'submit', 'law_ref': 'saved'}]})
            self.assertFalse(result['ok'])
            self.assertEqual(result['completed_actions'], 2)
            self.assertEqual(result['skipped_actions'], 1)
            self.assertFalse(result['ends_exploration'])
            console.submit.assert_not_called()
            self.assertEqual(agent.latest, {'law': FITTED, 'claims': CLAIMS})

    def test_batch_tool_failure_stops_and_invalid_envelopes_do_not_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, console = self.direct(directory)
            invalid = [[], [{'action': 'archive'}]*(MAX_BATCH_ACTIONS+1),
                       [{'action': 'batch', 'actions': [{'action': 'archive'}]}],
                       [{'action': 'observe', 'times': [0.]},
                        {'action': 'submit', 'law': LAW, 'claims': CLAIMS}, {'action': 'archive'}]]
            for actions in invalid:
                with self.subTest(actions=actions), self.assertRaises(ValueError):
                    agent._action({'action': 'batch', 'actions': actions})
            console.observe.assert_not_called()
            with patch('agent.run.candidate_tool', return_value={'ok': False, 'error': 'not fitted'}):
                result = agent._action({'action': 'batch', 'actions': [
                    {'action': 'fit', 'law': LAW}, {'action': 'observe', 'times': [0.]}]})
            self.assertFalse(result['ok'])
            console.observe.assert_not_called()

    def test_batch_submission_retains_early_stop_and_first_light_is_already_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.session(root)
            model = Model([{'action': 'batch', 'actions': [
                {'action': 'draft', 'law': LAW, 'claims': CLAIMS},
                {'action': 'submit', 'law_ref': 'saved'}]}, {'action': 'archive'}])
            with serve_session(session) as auth:
                result = DiscoveryAgent(Console(auth['endpoint'], auth['token']), model, root/'agent').run()
            self.assertEqual(result['status'], 'submitted')
            self.assertEqual(len(model.calls), 1)
            mission = json.loads(next(x['content'].split('\n', 1)[1] for x in model.calls[0]
                                      if x['content'].startswith('MISSION:\n')))
            self.assertEqual(len(mission['first_light']['times']), 8)
            self.assertEqual(session.console.budget['energy_used'], 0)

    def test_model_failure_still_records_generation_diagnostic(self):
        class FailedModel:
            def complete(self, *args, **kwargs):
                self.last_call_diagnostic = {'status': 'transport_error', 'elapsed_seconds': .1}
                raise InferenceError('synthetic transport error')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with serve_session(self.session(root)) as auth:
                with self.assertRaises(InferenceError):
                    DiscoveryAgent(Console(auth['endpoint'], auth['token']), FailedModel(), root/'agent').run()
            events = json.loads((root/'agent/events.json').read_text())
            self.assertEqual(events[0]['generation']['status'], 'transport_error')
            self.assertIn('inference_error', events[0])


if __name__ == '__main__': unittest.main()
