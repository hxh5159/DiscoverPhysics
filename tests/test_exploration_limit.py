"""The public development session uses the same five-minute episode limit."""
import unittest

from development_worlds import catalog
from development_worlds.console import DevelopmentConsole


class ExplorationLimitTests(unittest.TestCase):
    def test_clock_starts_at_first_mission_and_never_resets(self):
        for world in catalog.WORLDS:
            with self.subTest(world=world):
                now = [100.]
                cfg = catalog.config(world)
                self.assertEqual(cfg['wall_clock_seconds'], 300.)
                console = DevelopmentConsole(world, catalog.noise_sigma(world), 1,
                                             cfg, clock=lambda: now[0])
                now[0] += 10000  # Loading/setup before mission is not charged.
                first = console.mission()
                self.assertEqual(first['limits']['wall_clock_seconds'], 300.)
                self.assertEqual(console.budget['wall_clock_remaining_seconds'], 300.)
                now[0] += 299
                self.assertEqual(console.mission()['budget']['wall_clock_remaining_seconds'], 1.)
                now[0] += 1
                with self.assertRaises(TimeoutError):
                    console.mission()
                self.assertEqual(console.budget['wall_clock_remaining_seconds'], 0.)


if __name__ == '__main__':
    unittest.main()
