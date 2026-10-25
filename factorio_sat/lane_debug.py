"""Debug helper for lane_balancer: pin a known layout (blueprint string) into the lane encoding and test SAT.

    python -m factorio_sat.lane_debug NETWORK BLUEPRINT_FILE [--rot K]
Rotates the layout by K quarter turns (np.rot90 convention) so the flow runs to +x, as the encoder expects."""
import argparse

import numpy as np

from .blueprint import convert_to_tiles
from .lane_balancer import create_lane_balancer, lane_counts, open_lane_network
from .belt_balancer import setup_balancer_input, setup_balancer_output
from .rotate import Operation
from .tile import EmptyTile, TransformableTile


def pinned(nodes, tiles, stage, merges=(), rest_empty=True, turn_90=False):
    h, w = tiles.shape
    n, m = lane_counts(nodes, merges)
    grid = create_lane_balancer(nodes, w, h, merges=merges)
    if stage >= 1:
        grid.prevent_intersection()
    if stage >= 2:
        grid.enforce_maximum_underground_length()
    if stage >= 3:
        start = [grid.get_tile_instance(0, y) for y in range(h)]
        end = [grid.get_tile_instance(x, h - 1) for x in range(w)] if turn_90 else \
            [grid.get_tile_instance(w - 1, y) for y in range(h)]
        setup_balancer_input(grid, start, 0, n, rest_empty, rest_belts=False)
        setup_balancer_output(grid, end, 3 if turn_90 else 0, m, rest_empty)
    for (y, x), t in np.ndenumerate(tiles):
        grid.set_tile(x, y, t)
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('network')
    ap.add_argument('bp')
    ap.add_argument('--rot', type=int, default=0)
    ap.add_argument('--use-ends', action='store_true')
    ap.add_argument('--90', dest='turn_90', action='store_true')
    a = ap.parse_args()
    nodes, merges = open_lane_network(a.network, with_merges=True)
    tiles = convert_to_tiles(open(a.bp).read().strip())
    op = [Operation.NO_OP, Operation.ROT_90, Operation.ROT_180, Operation.ROT_270][a.rot % 4]
    tiles = op.grid(tiles)
    tiles = np.vectorize(lambda t: op.tile(t) if isinstance(t, TransformableTile) else t, otypes=[object])(tiles)
    print(tiles.shape)
    for stage in range(4):
        grid = pinned(nodes, tiles, stage, merges, not a.use_ends, a.turn_90)
        sol = next(iter(grid.itersolve(ignore_colour=True)), None)
        print('stage', stage, 'SAT' if sol is not None else 'UNSAT')


if __name__ == '__main__':
    main()
