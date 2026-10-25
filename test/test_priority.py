"""Priority and filter splitters.

Priority is a per-splitter setting, so the layout encoding does not change: a network line may carry `in=X`
(input priority on colour X), `out=Y` (output priority on colour Y) or `filter` (its dangling output is blocked by a
filter), and the side is read from the solution's colours after solving (belt_balancer.apply_settings). The only
encoding change is for filters: the blocked half may have anything in front of it (allow_filtered_fronts).

rq_*.txt are Raynquist's 2025 book designs (Factorio 2.x blueprints): the 1-3 TU uses input priority on its
loop-back merge, the 4-3 TU input and output priorities, and the 5-2 a deconstruction-planner filter.
"""
import io
import os
import unittest

import numpy as np

from factorio_sat import belt_balancer, blueprint
from factorio_sat.direction import Direction
from factorio_sat.network import (deduplicate_network, get_input_output_colours, network_settings, open_network, parse_network, save_network,
                                  tidy_network)
from factorio_sat.tile import BaseTile, Splitter

HERE = os.path.join(os.path.dirname(__file__), 'regression')

NET_1X3 = '0 1 2 2 in=0\n-1 2 1 3\n-1 2 3 3\n'


def read(name):
    with open(os.path.join(HERE, name)) as f:
        return f.read().strip()


def sides(network):
    (_, n), (_, m) = get_input_output_colours(network)
    return n, m


def parse_blueprint(bp):
    tiles = blueprint.import_blueprint(blueprint.decode_blueprint(bp))
    network, settings = tidy_network(*parse_network(tiles, with_settings=True))
    out = io.StringIO()
    save_network(out, network, settings)
    return out.getvalue()


def solve_1x3():
    # the plain API: open_network / deduplicate_network / create_balancer carry the settings along by themselves
    network = deduplicate_network(open_network(io.StringIO(NET_1X3)))
    grid = belt_balancer.create_balancer(network, 5, 4, 8)
    grid.prevent_intersection()
    belt_balancer.setup_balancer_ends(grid, network, False, True)
    grid.enforce_maximum_underground_length()
    return grid.solve()


class TestFormat(unittest.TestCase):
    def test_round_trip(self):
        network, settings = open_network(io.StringIO(NET_1X3 + '0 3 -1 4 filter\n'), with_settings=True)
        self.assertEqual(settings, [{'in': 0}, None, None, {'filter': 'deconstruction-planner'}])
        out = io.StringIO()
        save_network(out, network, settings)
        self.assertEqual(out.getvalue(), NET_1X3 + '0 3 -1 4 filter\n')

    def test_plain_callers_unchanged(self):
        self.assertEqual(open_network(io.StringIO(NET_1X3)), [((0, 1), (2, 2)), ((None, 2), (1, 3)), ((None, 2), (3, 3))])
        out = io.StringIO()
        save_network(out, open_network(io.StringIO('0 0 1 1\n1 1 2 2\n')))
        self.assertEqual(out.getvalue(), '0 0 1 1\n1 1 2 2\n')

    def test_ambiguous_side_rejected(self):
        for line in ('0 0 1 1 in=0', '0 1 2 2 out=2', '0 1 2 3 in=5', '0 1 2 3 filter', '0 1 2 -1 filter out=2'):
            with self.assertRaises(ValueError, msg=line):
                open_network(io.StringIO(line + '\n'), with_settings=True)

    def test_identical_nodes_must_agree(self):
        network, settings = open_network(io.StringIO('0 1 2 2 in=0\n0 1 2 2\n'), with_settings=True)
        with self.assertRaises(ValueError):
            network_settings(network, settings)

    def test_splitter_tile(self):
        tile = Splitter(Direction.RIGHT, True, 'left', None, 'deconstruction-planner')
        self.assertEqual(BaseTile.read(tile.write()), tile)
        self.assertEqual(BaseTile.read(tile.write()).input_priority, 'left')
        self.assertEqual(tile, Splitter(Direction.RIGHT, True))  # settings take no part in equality
        self.assertEqual(Splitter(Direction.RIGHT, True).write(), {'type': 'splitter', 'direction': 0, 'is_head': True})
        self.assertEqual(tile.flip_x().input_priority, 'right')
        self.assertEqual(tile.rotate_90().input_priority, 'left')


class TestParse(unittest.TestCase):
    def test_input_priority_1x3(self):
        self.assertEqual(parse_blueprint(read('rq_1x3_tu.txt')), NET_1X3)

    def test_output_priority_4x3(self):
        text = parse_blueprint(read('rq_4x3_tu.txt'))
        self.assertIn('0 0 4 5 out=5\n', text)   # the two outputs of an output-priority splitter keep two colours
        self.assertEqual(text.count('in='), 3)

    def test_filter_5x2(self):
        text = parse_blueprint(read('rq_5x2_filter.txt'))
        self.assertIn('0 1 -1 4 filter\n', text)  # the blocked output is dangling, although an exit is in front

class TestLayout(unittest.TestCase):
    def test_priority_does_not_change_the_encoding(self):
        def clauses(text):
            network = deduplicate_network(open_network(io.StringIO(text)))
            grid = belt_balancer.create_balancer(network, 5, 4, 8)
            grid.prevent_intersection()
            return grid.clauses
        self.assertEqual(clauses(NET_1X3), clauses(NET_1X3.replace(' in=0', '')))

    def test_export_sets_the_side_of_the_colour(self):
        solution = solve_1x3()
        tiles = np.vectorize(lambda cell: BaseTile.read(cell['tile']), otypes=[object])(solution)
        bp = blueprint.make_blueprint(tiles)
        prioritised = [e for e in bp['blueprint']['entities'] if 'input_priority' in e]
        self.assertEqual(len(prioritised), 1)
        # the exported blueprint parses back to the 1-3 network (colours renumbered), priority on the network input
        parsed, settings = open_network(io.StringIO(parse_blueprint(blueprint.encode_blueprint(bp))), with_settings=True)
        self.assertEqual(sides(parsed), (1, 3))
        self.assertEqual([(node, entry) for node, entry in zip(parsed, settings) if entry], [(((0, 2), (1, 1)), {'in': 0})])


def pinned_leak(filtered):
    network = deduplicate_network(open_network(os.path.join(HERE, 'net_4x1_tu')))
    bp = read('leak_4x1_w7_l4.txt')
    height, width = blueprint.convert_to_tiles(bp).shape
    keys = {key for key in network if None in key[1]} if filtered else None
    grid = belt_balancer.create_balancer(network, width, height, 4, filtered=keys)
    grid.prevent_intersection()
    belt_balancer.setup_balancer_ends(grid, network, False, False)
    grid.enforce_maximum_underground_length()
    belt_balancer.set_nonempty_tiles(grid, bp)
    return grid, network


class TestFilter(unittest.TestCase):
    def test_filter_allows_an_underground_in_front(self):
        # leak_4x1 failed in game (entrance in front of an unused output): rejected plain, accepted with a filter
        self.assertIsNone(pinned_leak(False)[0].solve())
        grid, network = pinned_leak(True)
        solution = grid.solve()
        self.assertIsNotNone(solution)
        self.assertGreaterEqual(belt_balancer.apply_settings(solution, network, {}, 'deconstruction-planner'), 1)
        tiles = np.vectorize(lambda cell: BaseTile.read(cell['tile']), otypes=[object])(solution)
        filters = [e for e in blueprint.make_blueprint(tiles)['blueprint']['entities'] if 'filter' in e]
        self.assertTrue(filters and all(e['output_priority'] in ('left', 'right') for e in filters))
        # the filtered layout parses back to a 4-1 network of 3 splitters, filters where the exports put them
        text = parse_blueprint(blueprint.encode_blueprint(blueprint.make_blueprint(tiles)))
        parsed, settings = open_network(io.StringIO(text), with_settings=True)
        self.assertEqual(sides(parsed), (4, 1))
        self.assertEqual(len(parsed), 3)
        self.assertEqual(sum(bool(entry and 'filter' in entry) for entry in settings), len(filters))

if __name__ == '__main__':
    unittest.main()
