"""Branch `unified`: the lane encoder (lane_balancer) with belt-level networks, priorities and filters, mixed-tier
undergrounds and merges at underground ends. Each is off by default; with none of them the CNF is the one of branch
`lanes` (checked against stored digests in factorio-balancer-sat work/unified/equiv)."""
import os
import tempfile
import unittest

import numpy as np

from factorio_sat import lane_balancer as lb
from factorio_sat.belt_balancer import setup_balancer_input, setup_balancer_output
from factorio_sat.blueprint import convert_to_tiles
from factorio_sat.lane_debug import pinned
from factorio_sat.network import network_settings, open_network
from factorio_sat.rotate import Operation
from factorio_sat.tile import TransformableTile

HERE = os.path.join(os.path.dirname(__file__), 'regression')
NET_1X3 = '0 1 2 2 in=0\n-1 2 1 3\n-1 2 3 3\n'


def solve(grid, n, m, use_ends=False):
    grid.prevent_intersection()
    grid.enforce_maximum_underground_length()
    start = [grid.get_tile_instance(0, y) for y in range(grid.height)]
    end = [grid.get_tile_instance(grid.width - 1, y) for y in range(grid.height)]
    setup_balancer_input(grid, start, 0, n, not use_ends, rest_belts=False)
    setup_balancer_output(grid, end, 0, m, not use_ends)
    return next(iter(grid.itersolve(ignore_colour=True)), None)


def belt_net(text):
    with tempfile.NamedTemporaryFile('w', suffix='.net', delete=False) as f:
        f.write(text)
    net, sets = open_network(f.name, with_settings=True)
    os.unlink(f.name)
    return lb.belt_to_lane(net, network_settings(net, sets))


class TestUnified(unittest.TestCase):
    def test_belt_network_with_priority(self):
        nodes, settings = belt_net(NET_1X3)
        self.assertEqual(len(settings), 1)
        grid = lb.create_lane_balancer(nodes, 5, 4, 8, settings=settings)
        sol = solve(grid, *lb.lane_counts(nodes), use_ends=True)
        self.assertIsNotNone(sol)
        prios = [c['tile'].get('input_priority') for c in sol.flat if c['is_splitter']]
        self.assertEqual(sum(p is not None for p in prios), 2)       # both halves of the one prioritised splitter

    def test_belt_network_cnf_has_no_extras(self):
        # a belt network through belt_to_lane and the plain lane path build the same instance
        nodes, _ = belt_net('0 0 1 1\n')
        a = lb.create_lane_balancer(nodes, 3, 2)
        b = lb.create_lane_balancer(nodes, 3, 2, settings={}, tiers=None, ug_merges=False)
        self.assertEqual(a.clauses, b.clauses)

    def test_mixed_tiers(self):
        nodes, merges = lb.open_lane_network(os.path.join(HERE, 'lnet_2x2_lane.lnet'), with_merges=True)
        grid = lb.create_lane_balancer(nodes, 8, 3, merges=merges, tiers=['yellow', 'red'])
        self.assertTrue(grid.multi_tier)
        self.assertIsNotNone(solve(grid, *lb.lane_counts(nodes, merges)))

    def test_underground_exit_merge(self):
        # gen/4-2 TU W7 L5: a splitter's unused output side-feeds an underground exit (L9). The traced lane network
        # has a merge there; the layout is expressible only with ug_merges.
        path = os.path.join(HERE, 'lnet_4x2_tu_ugexit_merge.lnet')
        nodes, merges, settings, ugm = lb.open_lane_network(path, with_settings=True)
        self.assertTrue(ugm)
        with open(os.path.join(HERE, 'lnet_4x2_tu_ugexit_merge.bp')) as f:
            tiles = convert_to_tiles(f.read().strip())
        for flag, expect in ((True, True), (False, False)):
            grid = pinned(nodes, tiles, 3, merges, True, False, 8, None, settings, flag)
            self.assertEqual(next(iter(grid.itersolve(ignore_colour=True)), None) is not None, expect, flag)

    def test_lane_settings_parse(self):
        node = (((2, 4), (5, None)), ((2, 6), (7, None)))
        self.assertEqual(lb.parse_lane_settings(['in=4', 'filter'], node)['in'], 4)
        with self.assertRaises(ValueError):
            lb.parse_lane_settings(['in=2'], (((2, 2), (5, None)), ((2, 2), (7, None))))   # 2 is on both halves


if __name__ == '__main__':
    unittest.main()
