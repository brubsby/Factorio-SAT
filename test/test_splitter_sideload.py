"""Regression: no underground end may sit crosswise in front of a splitter.

The layouts in test/regression/leak_*.txt are real Factorio-SAT outputs that were placed in Factorio 2.1.23 and
failed there: a splitter output that the network leaves unused (a dangling port) faced the side of an underground
entrance (leak_4x1) or exit (leak_4x2, which an earlier entrance-only fix still allowed). The game sideloads into
it, so the built network has an extra arc: the 4-1 TU lost 25 % throughput in 2 of 15 input/output subsets, the
4-2 TU 12 % in 1 of 45, with uneven outputs. ok_4x2 is a layout of the same 4-2 network that passed every in-game
check and must stay SAT.
"""
import os
import unittest

from factorio_sat import belt_balancer, blueprint
from factorio_sat.network import deduplicate_network, open_network

HERE = os.path.join(os.path.dirname(__file__), 'regression')


def pinned(net, bp_file, underground_length=4):
    network = deduplicate_network(open_network(os.path.join(HERE, net)))
    bp = open(os.path.join(HERE, bp_file)).read().strip()
    height, width = blueprint.convert_to_tiles(bp).shape
    grid = belt_balancer.create_balancer(network, width, height, underground_length)
    grid.prevent_intersection()
    belt_balancer.setup_balancer_ends(grid, network, False, False)
    grid.enforce_maximum_underground_length()
    belt_balancer.set_nonempty_tiles(grid, bp)
    return grid


class TestSplitterSideload(unittest.TestCase):
    def test_entrance_in_front_of_unused_output_rejected(self):
        self.assertIsNone(pinned('net_4x1_tu', 'leak_4x1_w7_l4.txt').solve())

    def test_exit_in_front_of_unused_output_rejected(self):
        self.assertIsNone(pinned('net_4x2_tu', 'leak_4x2_w6_l6_exit.txt').solve())

    def test_leak_free_layout_still_accepted(self):
        self.assertIsNotNone(pinned('net_4x2_tu', 'ok_4x2_w5_l7.txt').solve())


if __name__ == '__main__':
    unittest.main()
