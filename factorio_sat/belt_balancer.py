import argparse
import json
from typing import Sequence

from pysat.card import EncType
import numpy as np

from .direction import Direction
from . import optimisations
from . import blueprint
from .cardinality import library_atleast, library_equals, quadratic_one
from .network import deduplicate_network, get_input_output_colours, network_settings, open_network, DEFAULT_FILTER
from .solver import Grid, TileTemplate
from .template import OneHotTemplate
from .tile import EmptyTile, Belt, Splitter
from .util import implies, invert_components, literals_different, set_all_false, set_number, set_numbers


def setup_balancer_ends_with_offsets(grid, network, start_offset: int, end_offset: int):
    (input_colour, input_count), (output_colour, output_count) = get_input_output_colours(network)

    assert start_offset + input_count <= grid.height and end_offset + output_count <= grid.height

    for y in range(start_offset):
        grid.set_tile(0, y, EmptyTile())

    for y in range(start_offset, start_offset + input_count):
        grid.set_tile(0, y, Belt(Direction.RIGHT, Direction.RIGHT))
        grid.set_colour(grid.width - 1, y, input_colour)

    for y in range(start_offset + input_count, grid.height):
        grid.set_tile(0, y, EmptyTile())

    for y in range(end_offset):
        grid.set_tile(grid.width - 1, y, EmptyTile())

    for y in range(end_offset, end_offset + output_count):
        grid.set_tile(grid.width - 1, y, Belt(Direction.RIGHT, Direction.RIGHT))
        grid.set_colour(grid.width - 1, y, output_colour)

    for y in range(end_offset + output_count, grid.height):
        grid.set_tile(grid.width - 1, y, EmptyTile())


def setup_balancer_input(grid: Grid, tiles: Sequence[TileTemplate], direction: int, count: int, rest_empty: bool = True,
                         rest_belts: bool = True):
    """rest_empty=False: the other tiles of the column may be used (nothing enters through the edge there); they are
    then belts unless rest_belts=False (empty tiles and splitters allowed too, as Raynquist's 1-n TU use)."""
    assert count <= len(tiles)
    offsets = [grid.allocate_variable() for _ in range(len(tiles) - count + 1)]

    if len(offsets) == 0:
        for tile in tiles:
            grid.clauses += [[tile.input_direction[direction]]]
    else:
        grid.clauses += quadratic_one(offsets)
        for di, literal in enumerate(offsets):
            consequences = []
            for i, tile in enumerate(tiles):
                if i in range(di, di + count):
                    consequences += [[tile.input_direction[direction]], [tile.is_belt]]
                elif rest_empty:
                    consequences += set_all_false(tile.all_direction)
                else:
                    consequences += [[-tile.input_direction[direction]]] + ([[tile.is_belt]] if rest_belts else [])
            grid.clauses += implies([literal], consequences)
    return offsets


def setup_balancer_output(grid: Grid, tiles: Sequence[TileTemplate], direction: int, count: int, rest_empty: bool = True):
    assert count <= len(tiles)
    offsets = [grid.allocate_variable() for _ in range(len(tiles) - count + 1)]

    if len(offsets) == 0:
        for tile in tiles:
            grid.clauses += [[tile.output_direction[direction]], [tile.is_belt]]
    else:
        grid.clauses += quadratic_one(offsets)
        for di, literal in enumerate(offsets):
            consequences = []
            for i, tile in enumerate(tiles):
                if i in range(di, di + count):
                    consequences += [[tile.output_direction[direction]], [tile.is_belt]]
                elif rest_empty:
                    consequences += set_all_false(tile.all_direction)
                else:
                    consequences += [[-tile.output_direction[direction]]]
            grid.clauses += implies([literal], consequences)
    return offsets


def setup_balancer_ends(grid: Grid, network, aligned: bool, use_ends: bool, free_ends: bool = False):
    """use_ends=False: closed edges (every non-port tile of the input and output columns empty), Factorio-SAT's
    original rule. use_ends=True: open edges (nothing enters or leaves through the edge but the ports; the Grid's
    is_input / is_output counts keep any other edge tile from leaking). free_ends (with use_ends): the input column's
    other tiles may also be empty or splitters, not only belts. With open edges a port tile may curve into the column;
    in the game a curve fed from off the grid is a different port map, so callers that need straight ports add that
    rule (factorio-balancer-sat tools/gensat/edges.py straight_io)."""
    (_, input_count), (_, output_count) = get_input_output_colours(network.elements())

    start_tiles = [grid.get_tile_instance(0, y) for y in range(grid.height)]
    end_tiles = [grid.get_tile_instance(grid.width - 1, y) for y in range(grid.height)]

    start_offsets = setup_balancer_input(grid, start_tiles, 0, input_count, not use_ends, not free_ends)
    end_offsets = setup_balancer_output(grid, end_tiles, 0, output_count, not use_ends)

    if aligned:
        if input_count >= output_count:
            for i, start_offset in enumerate(start_offsets):
                grid.clauses += implies([start_offset], [end_offsets[i:(i + 1 + input_count - output_count)]])
        else:
            for i, end_offset in enumerate(end_offsets):
                grid.clauses += implies([end_offset], [start_offsets[i:(i + 1 + output_count - input_count)]])


def setup_balancer_ends_90(grid: Grid, network, use_ends: bool, free_ends: bool = False):
    """Inputs enter column 0 heading east, outputs leave the bottom row heading south. use_ends / free_ends as in
    setup_balancer_ends (free_ends was missing here: with use_ends alone the input column's other tiles had to be
    belts)."""
    (_, input_count), (_, output_count) = get_input_output_colours(network.elements())

    start_tiles = [grid.get_tile_instance(0, y) for y in range(grid.height)]
    end_tiles = [grid.get_tile_instance(x, grid.height - 1) for x in range(grid.width)]
    setup_balancer_input(grid, start_tiles, 0, input_count, not use_ends, not free_ends)
    setup_balancer_output(grid, end_tiles, 3, output_count, not use_ends)


def setup_balancer_ends_180(grid: Grid, network):
    (_, input_count), (_, output_count) = get_input_output_colours(network.elements())

    tiles = [grid.get_tile_instance(0, y) for y in range(grid.height)]
    setup_balancer_input(grid, tiles, 0, input_count, rest_empty=False)
    setup_balancer_output(grid, tiles, 2, output_count, rest_empty=False)


def create_balancer(network, width: int, height: int, underground_length: int, tiers=None, filtered=None,
                    prio_fronts=None) -> Grid:
    """tiers: optional list of underground tiers (see solver.Grid); None keeps the single-tier encoding.
    filtered: optional node keys of `network` (splitters with one dangling output) whose dangling side gets a
    filter (see allow_filtered_fronts); by default those marked `filter` in network.settings (a deduplicated
    SetNetwork). Priorities need nothing in the encoding: every solution gets them written into its splitter tiles
    (apply_settings, via Grid.solution_hook).
    prio_fronts: optional node keys (splitters with one dangling output, e.g. dangling_mergers(network)) whose
    dangling side may have something in front of it, like a filtered splitter, but held off by OUTPUT PRIORITY toward
    the used side instead of a filter (set on export wherever that front is not empty). The game then sends that side
    only the overflow of the used one, so the built network differs from `network` by a priority arc: only a
    re-parse checked with splnet's priority semantics can accept such a layout."""
    assert width > 0 and height > 0

    all_colours = set()
    for input_colours, output_colours in network:
        for colour in input_colours + output_colours:
            all_colours.add(colour)
    all_colours.discard(None)

    grid = Grid(width, height, max(all_colours) + 1, underground_length, {'node': OneHotTemplate(len(network))}, tiers=tiers)
    for colour in range(max(all_colours) + 1):
        if colour in all_colours:
            continue
        grid.prevent_colour(colour)

    grid.block_underground_through_edges()
    grid.prevent_bad_undergrounding()
    grid.prevent_bad_colouring()

    checkerboard = [grid.get_tile_instance(x, y).is_splitter for x in range(grid.width) for y in range(grid.height) if (x + y) % 2]
    grid.clauses += library_equals(checkerboard, sum(network.values()), grid.pool)

    # There is exactly one splitter of each type
    for i, count in enumerate(network.values()):
        literals = [grid.get_tile_instance(x, y).node[i] for x in range(grid.width) for y in range(grid.height)]
        grid.clauses += library_equals(literals, count, grid.pool, EncType.kmtotalizer)

    # Each splitter has one type
    for x in range(grid.width):
        for y in range(grid.height):
            tile = grid.get_tile_instance(x, y)
            grid.clauses += quadratic_one([-tile.is_splitter_head, *tile.node])

    (input_colour, input_count), (output_colour, output_count) = get_input_output_colours(network.elements())

    is_inputs = [grid.get_tile_instance(x, y).is_input for x in range(grid.width) for y in range(grid.height)]
    grid.clauses += library_equals(is_inputs, input_count, grid.pool)

    is_outputs = [grid.get_tile_instance(x, y).is_output for x in range(grid.width) for y in range(grid.height)]
    grid.clauses += library_equals(is_outputs, output_count, grid.pool)

    for i, (input_colours, output_colours) in enumerate(network):
        assert sum(colour is None for colour in input_colours + output_colours) <= 1

        for x in range(grid.width):
            for y in range(grid.height):
                tile00 = grid.get_tile_instance(x, y)
                grid.clauses += implies([tile00.is_input], set_number(input_colour, tile00.colour))
                grid.clauses += implies([tile00.is_output], set_number(output_colour, tile00.colour))
                if any(colour is None for colour in input_colours):
                    assert not any(colour is None for colour in output_colours)
                    grid.clauses.append([-tile00.node[i], *tile00.output_direction])
                else:
                    grid.clauses.append([-tile00.node[i], *tile00.input_direction])

                for direction in Direction:
                    dx0, dy0 = direction.vec
                    dx1, dy1 = direction.next.vec

                    precondition = [
                        tile00.node[i],
                    ]
                    if any(colour is None for colour in input_colours):
                        assert not any(colour is None for colour in output_colours)
                        precondition.append(tile00.output_direction[direction])
                    else:
                        precondition.append(tile00.input_direction[direction])

                    tile10 = grid.get_tile_instance_offset(x, y, dx0, dy0)
                    tile01 = grid.get_tile_instance_offset(x, y, dx1, dy1)
                    tile11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1)

                    if any(tile is None for tile in (tile00, tile10, tile01, tile11)):
                        grid.clauses.append(invert_components(precondition))
                        continue

                    colour_a, colour_b = input_colours
                    if colour_a is None or colour_b is None:
                        colour = colour_a
                        if colour is None:
                            colour = colour_b
                            assert colour is not None
                        grid.clauses += implies(precondition, literals_different(tile00.input_direction[direction], tile01.input_direction[direction]))
                        grid.clauses += implies(precondition + [tile00.input_direction[direction]], set_number(colour, tile00.colour))
                        grid.clauses += implies(precondition + [tile01.input_direction[direction]], set_number(colour, tile01.colour))
                    else:
                        grid.clauses += implies(precondition, [[tile00.input_direction[direction]], [tile01.input_direction[direction]]])
                        grid.clauses += implies(precondition, set_numbers(*input_colours, tile00.colour, tile01.colour))

                    colour_a, colour_b = output_colours
                    if colour_a is None or colour_b is None:
                        colour = colour_a
                        if colour is None:
                            colour = colour_b
                            assert colour is not None

                        grid.clauses += implies(precondition, literals_different(tile00.output_direction[direction], tile01.output_direction[direction]))
                        grid.clauses += implies(precondition + [tile00.output_direction[direction]], set_number(colour, tile10.colour))
                        grid.clauses += implies(precondition + [tile01.output_direction[direction]], set_number(colour, tile11.colour))
                    else:
                        grid.clauses += implies(precondition, [[tile00.output_direction[direction]], [tile01.output_direction[direction]]])
                        grid.clauses += implies(precondition, set_numbers(*output_colours, tile10.colour, tile11.colour))
    settings = getattr(network, 'settings', None) or {}
    if filtered is None:
        filtered = {key for key, entry in settings.items() if 'filter' in entry}
    prio_fronts = set(prio_fronts or ()) - set(filtered)
    if filtered or prio_fronts:
        allow_filtered_fronts(grid, network, set(filtered) | prio_fronts)
    if settings or prio_fronts:
        grid.solution_hook = lambda solution: apply_settings(solution, network, settings, prio_fronts=prio_fronts)
    grid.prio_fronts = prio_fronts
    return grid


def dangling_mergers(network):
    """Node keys of the 2->1 splitters (both inputs used, exactly one output dangling): candidates for prio_fronts."""
    return {key for key in network if None not in key[0] and sum(colour is None for colour in key[1]) == 1}


def allow_filtered_fronts(grid: Grid, network, filtered):
    """Let the dangling-output half of the given splitters have anything in front of it (a crosswise belt or
    underground end, or a splitter half with no input), as Raynquist's filtered splitters do: in the game the filter
    (deconstruction planner, never on a belt) on that side's output priority stops it outputting there.

    Per tile, r -> the tile is the outputless half of a splitter whose node is in `filtered`; r then lifts
    prevent_intersection's in-front rules for that tile only (Grid.front_relax). Call before prevent_intersection."""
    keys = list(network.keys())
    indices = [keys.index(key) for key in filtered]
    for key in filtered:
        if sum(colour is None for colour in key[1]) != 1:
            raise ValueError(f'filtered splitter {key} needs exactly one dangling output')
    for x in range(grid.width):
        for y in range(grid.height):
            tile = grid.get_tile_instance(x, y)
            r = grid.allocate_variable()
            grid.front_relax[tile.is_splitter] = r
            grid.clauses.append([-r, tile.is_splitter])
            grid.clauses += [[-r, -lit] for lit in tile.output_direction]
            grid.clauses.append([-r, -tile.is_splitter_head, *[tile.node[i] for i in indices]])
            for direction in Direction:
                head = grid.get_tile_instance_offset(x, y, -direction.next.dx, -direction.next.dy)
                grid.clauses.append([-r, tile.is_splitter_head, -tile.input_direction[direction],
                                     *([] if head is None else [head.node[i] for i in indices])])


def apply_settings(solution, network, settings, filter_item=None, prio_fronts=()):
    """Write priorities and filters into a solved grid (cells as from Grid.solve / itersolve), in place.

    settings: {node key: {'in': X, 'out': Y, 'filter': item}} (network.network_settings). The side is read from the
    solution's colours: input priority goes on the half whose input has colour X, output priority on the half whose
    front tile has colour Y, and a filter (with output priority, as the game needs) on the half that outputs nothing.
    filter_item: also filter every other splitter with a dangling output that has something in front of it
    (--filter-dangling). prio_fronts: node keys whose outputless half has something in front of it get output priority
    on their other (used) half instead. Returns the number of splitters set."""
    keys = list(network.keys())
    height, width = solution.shape
    count = 0
    for y in range(height):
        for x in range(width):
            cell = solution[y, x]
            if not cell['is_splitter'] or not cell['is_splitter_head']:
                continue
            key = keys[cell['node']]
            entry = dict(settings.get(key) or {})
            head = Splitter.read(cell['tile'])
            d = head.direction
            halves = {'right': (x, y), 'left': (x + d.next.dx, y + d.next.dy)}
            fronts = {side: (hx + d.dx, hy + d.dy) for side, (hx, hy) in halves.items()}

            def colour_at(pos):
                px, py = pos
                return solution[py, px]['colour'] if 0 <= px < width and 0 <= py < height else None

            outless = [side for side, (hx, hy) in halves.items() if solution[hy, hx]['output_direction'] is None]
            if filter_item is not None and 'filter' not in entry and len(outless) == 1:
                fx, fy = fronts[outless[0]]
                if 0 <= fx < width and 0 <= fy < height and not solution[fy, fx]['is_empty']:
                    entry['filter'] = filter_item
            prio_used = None
            if key in prio_fronts and 'filter' not in entry and len(outless) == 1:
                fx, fy = fronts[outless[0]]
                if 0 <= fx < width and 0 <= fy < height and not solution[fy, fx]['is_empty']:
                    prio_used = next(side for side in halves if side != outless[0])
            if not entry and prio_used is None:
                continue
            values = {}
            if 'in' in entry:
                values['input_priority'] = next(side for side, pos in halves.items() if colour_at(pos) == entry['in'])
            if 'out' in entry:
                values['output_priority'] = next(side for side, pos in fronts.items() if colour_at(pos) == entry['out'])
            if prio_used is not None:
                values['output_priority'] = prio_used
            if 'filter' in entry:
                assert len(outless) == 1, (x, y, entry)
                values['output_priority'] = outless[0]
                values['filter'] = entry['filter']
            for hx, hy in halves.values():
                tile = solution[hy, hx]['tile']
                solution[hy, hx] = {**solution[hy, hx], 'tile': {**tile, **values}}
            count += 1
    return count


def enforce_edge_splitters(grid: Grid, network):
    (network_input_colour, _), (network_output_colour, _) = get_input_output_colours(network.elements())

    recirculate_input = 0
    recirculate_output = 0
    for inputs, outputs in network.elements():
        for colour in inputs:
            if colour == network_output_colour:
                recirculate_output += 1
        for colour in outputs:
            if colour == network_input_colour:
                recirculate_input += 1

    input_splitters = [(i, count) for i, ((input_colours, _), count) in enumerate(network.items())
                       if all(colour == network_input_colour for colour in input_colours)]
    if recirculate_input == 0:
        for i, count in input_splitters:
            literals = [grid.get_tile_instance(1, y).node[i] for y in range(grid.height)]
            grid.clauses += library_equals(literals, count, grid.pool, EncType.kmtotalizer)
            for y in range(grid.height):
                tile = grid.get_tile_instance(1, y)
                grid.clauses += implies([tile.node[i]], [[tile.input_direction[0], tile.output_direction[0]]])
    else:
        edge_splitter_min = sum(count for _, count in input_splitters) - recirculate_input
        if edge_splitter_min > 0:
            literals = [grid.get_tile_instance(1, y).node[i] for y in range(grid.height) for i, _ in input_splitters]
            grid.clauses += library_atleast(literals, edge_splitter_min, grid.pool)

    output_splitters = [(i, count) for i, ((_, output_colours), count) in enumerate(network.items())
                        if all(colour == network_output_colour for colour in output_colours)]
    if recirculate_output == 0:
        for i, count in output_splitters:
            literals = [grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height)]
            grid.clauses += library_equals(literals, count, grid.pool, EncType.kmtotalizer)

            # grid.clauses.append([grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height)])
            for y in range(grid.height):
                tile = grid.get_tile_instance(grid.width - 2, y)
                grid.clauses += implies([tile.node[i]], [[tile.input_direction[0], tile.output_direction[0]]])
    else:
        edge_splitter_min = sum(count for _, count in output_splitters) - recirculate_output
        if edge_splitter_min > 0:
            literals = [grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height) for i, _ in output_splitters]
            grid.clauses += library_atleast(literals, edge_splitter_min, grid.pool)


def prevent_double_edge_belts(grid: Grid):
    for x in (1, max(grid.width - 2, 1)):
        for y in range(grid.height - 1):
            tile_a = grid.get_tile_instance(x, y)
            tile_b = grid.get_tile_instance(x, y + 1)

            grid.clauses.append([
                -tile_a.input_direction[0], -tile_a.output_direction[0], tile_a.is_splitter,
                -tile_b.input_direction[0], -tile_b.output_direction[0], tile_b.is_splitter,
            ])


def set_nonempty_tiles(grid: Grid, blueprint_or_json: str):
    tiles = blueprint.convert_to_tiles(blueprint_or_json)
    for (row, col), tile in np.ndenumerate(tiles):
        if not isinstance(tile, EmptyTile):
            grid.set_tile(col, row, tile)


def main():
    parser = argparse.ArgumentParser(description='Creates a belt balancer from a splitter graph')
    parser.add_argument('network', type=argparse.FileType('r'), help='Splitter network')
    parser.add_argument('width', type=int, help='Belt balancer maximum width')
    parser.add_argument('height', type=int, help='Belt balancer maximum height')
    parser.add_argument('--90', action='store_true', dest='turn_90', help='Make 90 degree balancer')
    parser.add_argument('--180', action='store_true', dest='turn_180', help='Make 180 degree balancer')
    parser.add_argument('--custom', action='store_true', help="Don't set up balancer ends (you can set them up manually with --partial)")
    parser.add_argument('--edge-splitters', action='store_true',
                        help='Enforce that any splitter that has both connections to the input/output of the balancer must be placed on the edge')
    parser.add_argument('--edge-belts', action='store_true',
                        help='Prevents two side by side belts at the inputs/outputs of a balancer (weaker version of --edge-splitters)')
    parser.add_argument('--glue-splitters', action='store_true',
                        help='Prevents a configuration where a splitter has two straight input belts.'
                        ' Effectively pushing all splitters as far back as possible')
    parser.add_argument('--expand-underground', action='store_true', help='Ensures that underground belts are expanded if possible')
    parser.add_argument('--prevent-mergeable-underground', action='store_true', help='Prevent underground segments that can be merged')
    parser.add_argument('--prevent-bad-patterns', action='store_true',
                        help='Prevents patterns of belts that are not helpful, i.e. always substitutable for simpler configuration.')
    parser.add_argument('--break-symmetry', action='store_true',
                        help='Prevents vertical reflections being considered, sometimes unsound when applied with --prevent-bad-patterns')
    parser.add_argument('--use-ends', action='store_true', help='Allow ends to have non-empty tiles, asside from input/output belts')
    parser.add_argument('--free-ends', action='store_true',
                        help='With --use-ends, the input end\'s other tiles may also be empty or splitters (not only belts)')
    parser.add_argument('--fast', action='store_true', help='Enables all speed improving options')
    parser.add_argument('--aligned', action='store_true', help='Enforces balancer input aligns with output')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    parser.add_argument('--tiers', type=str,
                        help='Comma-separated underground tiers the layout may mix, e.g. yellow,red (gaps 4/6/8/10 for '
                        'yellow/red/blue/turbo). Tiers may nest and cross on one line, as in the game. Overrides '
                        '--underground-length; belts and splitters use the lowest tier listed')
    parser.add_argument('--tier-max', action='append', default=[], metavar='TIER=K',
                        help='At most K underground pairs of TIER (with --tiers), e.g. --tier-max red=2; repeatable')
    parser.add_argument('--all', action='store_true', help='Generate all belt balancers')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    parser.add_argument('--partial', type=argparse.FileType('r'), help='Partial balancer to base solution from')
    parser.add_argument('--filter-dangling', action='store_true',
                        help='Every splitter with a dangling output may have something in front of it there; the '
                        'output gets a filter (deconstruction planner) where it does. Network lines marked '
                        '`filter` get this individually')

    args = parser.parse_args()

    if args.underground_length == -1:
        args.underground_length = float('inf')

    if args.edge_splitters and args.edge_belts:
        raise RuntimeError('--edge-splitters and --edge-belts are mutually exclusive')

    if sum([args.aligned, args.turn_90, args.turn_180, args.custom]) >= 2:
        raise RuntimeError('--aligned, --90, --180 and --custom are mutually exclusive')

    if args.break_symmetry and args.turn_90:
        raise RuntimeError('--break-symmetry and --90 are mutually exclusive')

    network = open_network(args.network)
    args.network.close()

    network = deduplicate_network(network)
    settings = getattr(network, 'settings', None) or {}
    filtered = {key for key, entry in settings.items() if 'filter' in entry}
    if args.filter_dangling:
        filtered |= {key for key in network if None in key[1]}

    tiers = args.tiers.split(',') if args.tiers else None
    grid = create_balancer(network, args.width, args.height, args.underground_length, tiers, filtered)
    for cap in args.tier_max:
        name, k = cap.split('=')
        grid.limit_tier(name, int(k))
    grid.prevent_intersection()

    if args.edge_splitters or args.fast:
        enforce_edge_splitters(grid, network)
    if args.edge_belts:
        prevent_double_edge_belts(grid)
    if args.glue_splitters or args.fast:
        optimisations.glue_splitters(grid)
        optimisations.glue_partial_splitters(grid)
    if args.expand_underground or args.fast:
        optimisations.expand_underground(grid, min_x=1, max_x=grid.width - 2)
    if args.prevent_mergeable_underground or args.fast:
        optimisations.prevent_mergeable_underground(grid)
    if args.break_symmetry:
        optimisations.break_vertical_symmetry(grid)
    if args.prevent_bad_patterns or args.fast:
        optimisations.prevent_belt_hooks(grid)
        optimisations.prevent_semicircles(grid)
        optimisations.prevent_small_loops(grid)
        optimisations.prevent_underground_hook(grid)
        optimisations.prevent_zigzags(grid)
        optimisations.prevent_belt_parallel_splitter(grid)

    grid.enforce_maximum_underground_length()
    optimisations.prevent_empty_along_underground(grid)

    if args.partial is not None:
        with args.partial:
            set_nonempty_tiles(grid, args.partial.read())

    if args.turn_90:
        setup_balancer_ends_90(grid, network, args.use_ends, args.free_ends)
    elif args.turn_180:
        setup_balancer_ends_180(grid, network)
    elif args.custom:
        pass
    else:
        setup_balancer_ends(grid, network, args.aligned, args.use_ends, args.free_ends)

    for solution in grid.itersolve(solver=args.solver, ignore_colour=True):
        if args.filter_dangling:  # (the network's own settings are already in: Grid.solution_hook)
            apply_settings(solution, network, {}, DEFAULT_FILTER)
        print(json.dumps(solution.tolist()))
        if not args.all:
            break


if __name__ == '__main__':
    main()
