"""Mixed-tier undergrounds: different tiers may interleave on one line, the same tier may not.

The pinned layout (2 rows, width 8) is two lanes whose underground pairs interleave on row 0:
lane A goes under x2..x3 (entrance x1, exit x4), lane B under x4..x6 (entrance x3, exit x7). B's entrance sits on
A's segment and A's exit on B's. In Factorio 2.1.23 that works only when A and B are different tiers (a same-tier
end inside a same-tier span breaks the pairing), which is exactly what the encoding must say.

    row 0:  A>  A_in  >    B_in  A_out  v    .    B_out
    row 1:  B>  >     ^    .     .      >    >    >
"""
import os
import unittest

import numpy as np

from factorio_sat import belt_balancer, blueprint
from factorio_sat.direction import Direction
from factorio_sat.network import deduplicate_network, open_network
from factorio_sat.solver import Grid
from factorio_sat.tile import Belt, EmptyTile, UndergroundBelt

R, U, D = Direction.RIGHT, Direction.UP, Direction.DOWN
HERE = os.path.join(os.path.dirname(__file__), 'regression')


def interleaved(tier_a=None, tier_b=None):
    e = EmptyTile()
    return [
        [Belt(R, R), UndergroundBelt(R, True, tier_a), Belt(U, R), UndergroundBelt(R, True, tier_b), UndergroundBelt(R, False, tier_a),
         Belt(R, D), e, UndergroundBelt(R, False, tier_b)],
        [Belt(R, R), Belt(R, R), Belt(R, U), e, e, Belt(D, R), Belt(R, R), Belt(R, R)],
    ]


def pinned_grid(rows, **grid_args):
    grid = Grid(len(rows[0]), len(rows), None, **grid_args)
    grid.block_underground_through_edges()
    grid.prevent_bad_undergrounding()
    grid.enforce_maximum_underground_length()
    grid.prevent_intersection()
    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            grid.set_tile(x, y, tile)
    return grid


class TestMixedTiers(unittest.TestCase):
    def test_single_tier_rejects_interleaving(self):
        self.assertIsNone(pinned_grid(interleaved()).solve())
        self.assertIsNone(pinned_grid(interleaved(), underground_length=8).solve())

    def test_mixed_tiers_accept_interleaving(self):
        solution = pinned_grid(interleaved(), tiers=['yellow', 'red']).solve()
        self.assertIsNotNone(solution)
        tiles = np.vectorize(lambda cell: cell['tile'])(solution)
        self.assertNotEqual(tiles[0, 1].get('tier'), tiles[0, 3].get('tier'))
        self.assertEqual(tiles[0, 1]['tier'], tiles[0, 4]['tier'])  # a pair is one tier
        self.assertEqual(tiles[0, 3]['tier'], tiles[0, 7]['tier'])

    def test_same_tier_pinned_rejected(self):
        self.assertIsNone(pinned_grid(interleaved('normal', 'normal'), tiers=['yellow', 'red']).solve())
        self.assertIsNone(pinned_grid(interleaved('fast', 'fast'), tiers=['yellow', 'red']).solve())
        self.assertIsNotNone(pinned_grid(interleaved('fast', 'normal'), tiers=['yellow', 'red']).solve())

    def test_tier_cap(self):
        grid = pinned_grid(interleaved(), tiers=['yellow', 'red'])
        grid.limit_tier('red', 0)
        self.assertIsNone(grid.solve())
        grid = pinned_grid(interleaved(), tiers=['yellow', 'red'])
        grid.limit_tier('red', 1)
        self.assertIsNotNone(grid.solve())

    def test_tier_gap(self):
        # A tier only reaches its own gap: B's gap of 3 fits yellow, so yellow/yellow fails only on interleaving,
        # but a 5-gap pair needs red.
        e = EmptyTile()
        row = [Belt(R, R), UndergroundBelt(R, True), e, e, e, e, e, UndergroundBelt(R, False), Belt(R, R)]
        self.assertIsNone(pinned_grid([row]).solve())
        solution = pinned_grid([row], tiers=['yellow', 'red']).solve()
        self.assertEqual(solution[0, 1]['tile']['tier'], 'fast')

    def test_single_tier_cnf_unchanged(self):
        network = deduplicate_network(open_network(os.path.join(HERE, 'net_4x2_tu')))
        for name, length in (('yellow', 4), ('red', 6)):
            a = belt_balancer.create_balancer(network, 7, 5, length)
            b = belt_balancer.create_balancer(network, 7, 5, 99, tiers=[name])
            for grid in (a, b):
                grid.prevent_intersection()
                grid.enforce_maximum_underground_length()
            self.assertEqual(a.clauses, b.clauses)
            self.assertEqual(a.pool.top, b.pool.top)

    def test_blueprint_round_trip(self):
        tiles = np.empty((2, 8), dtype=object)
        tiles[:] = interleaved('normal', 'fast')
        bp = blueprint.make_blueprint(tiles)
        names = sorted({e['name'] for e in bp['blueprint']['entities']})
        self.assertEqual(names, ['fast-underground-belt', 'transport-belt', 'underground-belt'])
        back = blueprint.import_blueprint(bp)
        self.assertEqual(back[0, 3], UndergroundBelt(R, True, 'fast'))
        self.assertEqual(back[0, 4], UndergroundBelt(R, False, 'normal'))
        # A single-tier blueprint decodes exactly as before (no tier recorded)
        plain = blueprint.import_blueprint(blueprint.make_blueprint(np.vectorize(lambda t: t)(np.array(interleaved(), dtype=object)),
                                                                    level=blueprint.TransportBeltLevel.FAST))
        self.assertEqual(plain[0, 3], UndergroundBelt(R, True))

    def test_blueprint_2x_directions(self):
        # Factorio 2.x blueprints use 16-way directions (east = 4, south = 8)
        tiles = np.empty((2, 8), dtype=object)
        tiles[:] = interleaved('normal', 'fast')
        bp = blueprint.make_blueprint(tiles)
        bp['blueprint']['version'] = blueprint.encode_factorio_version(2, 0, 0, 0)
        for e in bp['blueprint']['entities']:
            if 'direction' in e:
                e['direction'] *= 2
        back = blueprint.import_blueprint(bp)
        self.assertEqual(back[0, 3], UndergroundBelt(R, True, 'fast'))
        self.assertEqual(back[0, 5].output_direction, D)


if __name__ == '__main__':
    unittest.main()
