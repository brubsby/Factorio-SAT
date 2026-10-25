"""Regression: the tile in front of a splitter's dangling output must be empty (Grid.forbid_dangling_fronts).

contact_1x3_w4_l8.txt is a layout of the 1-3 balancer network net_1x3_cbal_k5_11 (gensat complete.py, before the
rule) where splitter B's unused output half sits directly in front of splitter E's unused input half. In Factorio
2.1.23 the two are joined, so the built network has an 11th arc and reads 1/4, 1/4, 1/2 at full load. With the
rule it is UNSAT; ok_1x3_w3_l12.txt, a contact-free layout of the same network (1/3 each in game), stays SAT.
"""
import os
import unittest

from factorio_sat import belt_balancer, blueprint
from factorio_sat.network import deduplicate_network, open_network

HERE = os.path.join(os.path.dirname(__file__), 'regression')


def pinned(net, bp_file, underground_length=4, forbid=True):
    network = deduplicate_network(open_network(os.path.join(HERE, net)))
    bp = open(os.path.join(HERE, bp_file)).read().strip()
    height, width = blueprint.convert_to_tiles(bp).shape
    grid = belt_balancer.create_balancer(network, width, height, underground_length)
    grid.prevent_intersection(forbid_dangling_fronts=forbid)
    belt_balancer.setup_balancer_ends(grid, network, False, False)
    grid.enforce_maximum_underground_length()
    belt_balancer.set_nonempty_tiles(grid, bp)
    return grid


class TestDanglingFronts(unittest.TestCase):
    def test_contact_layout_rejected(self):
        self.assertIsNone(pinned('net_1x3_cbal_k5_11', 'contact_1x3_w4_l8.txt').solve())

    def test_contact_layout_was_accepted_without_rule(self):
        self.assertIsNotNone(pinned('net_1x3_cbal_k5_11', 'contact_1x3_w4_l8.txt', forbid=False).solve())

    def test_clean_layout_still_accepted(self):
        self.assertIsNotNone(pinned('net_1x3_cbal_k5_11', 'ok_1x3_w3_l12.txt').solve())

    def test_idempotent(self):
        network = deduplicate_network(open_network(os.path.join(HERE, 'net_1x3_cbal_k5_11')))
        grid = belt_balancer.create_balancer(network, 8, 4, 4)
        grid.prevent_intersection()
        n = len(grid.clauses)
        grid.forbid_dangling_fronts()
        self.assertEqual(n, len(grid.clauses))


if __name__ == '__main__':
    unittest.main()
