"""Exercise baseline turn settings through the public command entry points."""
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from development_worlds import runner
from shore_agent import agent


class AgentTurnLimitEntryTests(unittest.TestCase):
    def test_agent_cli_defaults_to_three_and_preserves_explicit_zero(self):
        for arguments, maximum in (([], 3), (['--max-turns', '0'], 0)):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as folder:
                with patch('shore_agent.models.managed_model', return_value=nullcontext(object())), \
                        patch.object(agent, 'Console'), patch.object(agent, 'DiscoveryAgent') as discovery, \
                        redirect_stdout(StringIO()):
                    discovery.return_value.run.return_value = {'has_saved_pair': True}
                    result = agent.main(['--output-dir', folder, *arguments])
                self.assertEqual(result, 0)
                self.assertEqual(discovery.call_args.kwargs['max_turns'], maximum)

    def test_development_runner_forwards_baseline_limit_and_keeps_custom_command(self):
        for extra, maximum in (([], '3'), (['--max-turns', '0'], '0'),
                               (['--max-turns', '2'], '2')):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as folder:
                result = {'world': '1', 'has_valid_submission': True, 'score': {'total': 0.}}
                with patch.object(runner, 'run_episode', return_value=result) as episode, \
                        redirect_stdout(StringIO()):
                    status = runner.main(['--worlds', '1', '--output', str(Path(folder)/'run'), *extra])
                self.assertEqual(status, 0)
                command = episode.call_args.args[2]
                self.assertEqual(command[command.index('--max-turns')+1], maximum)
        with tempfile.TemporaryDirectory() as folder:
            custom = ['python', 'custom_agent.py', '--own-limit', '7']
            result = {'world': '1', 'has_valid_submission': True, 'score': {'total': 0.}}
            with patch.object(runner, 'run_episode', return_value=result) as episode, \
                    redirect_stdout(StringIO()):
                runner.main(['--worlds', '1', '--output', str(Path(folder)/'run'),
                             '--agent-command', *custom])
            self.assertEqual(episode.call_args.args[2], custom)


if __name__ == '__main__':
    unittest.main()
