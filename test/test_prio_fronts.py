"""Dangling merger fronts held off by output priority (create_balancer(prio_fronts=...)).

leak_1x17_w17_l9_blue.txt is a free-ends layout of the 23-splitter 1-17 TU network net_1x17_tu1n made before
Grid.forbid_dangling_fronts: a 2->1 loop merger's dangling output faces something that takes items, so the game built a
2->2 splitter and the re-parsed network fails splnet (one output read 0.136 at full load in game). By default it is
now UNSAT. With prio_fronts=dangling_mergers(network) it is SAT again, and the export puts output priority on that
merger's used side: the user's hand fix (designs/1-17-tu-L9-blue-prio.tsv in factorio-balancer-sat), which passes
splnet's priority semantics and the game.
"""
import os
import unittest

from factorio_sat import belt_balancer, blueprint
from factorio_sat.network import deduplicate_network, open_network

HERE = os.path.join(os.path.dirname(__file__), 'regression')


def pinned(prio):
    network = deduplicate_network(open_network(os.path.join(HERE, 'net_1x17_tu1n')))
    bp = open(os.path.join(HERE, 'leak_1x17_w17_l9_blue.txt')).read().strip()
    height, width = blueprint.convert_to_tiles(bp).shape
    grid = belt_balancer.create_balancer(network, width, height, 8,
                                         prio_fronts=belt_balancer.dangling_mergers(network) if prio else None)
    grid.prevent_intersection()
    belt_balancer.setup_balancer_ends(grid, network, False, True, True)
    grid.enforce_maximum_underground_length()
    belt_balancer.set_nonempty_tiles(grid, bp)
    return grid


class TestPrioFronts(unittest.TestCase):
    def test_leaking_merger_rejected_by_default(self):
        self.assertIsNone(pinned(False).solve())

    def test_prio_mode_sets_output_priority_on_the_used_side(self):
        solution = pinned(True).solve()
        self.assertIsNotNone(solution)
        set_halves = [(x, y, cell['tile']['output_priority']) for y, row in enumerate(solution.tolist())
                      for x, cell in enumerate(row) if cell['is_splitter'] and cell['tile'].get('output_priority')]
        self.assertEqual(set_halves, [(1, 12, 'left'), (2, 12, 'left')])


if __name__ == '__main__':
    unittest.main()
