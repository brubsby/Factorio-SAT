import argparse
import collections
import copy
import json
import math
import sys
from os import path
from typing import Tuple

import numpy as np
from pysat.solvers import Solver

from .tile import Belt, EmptyTile, FillerTile, Splitter, UndergroundBelt
from .util import bin_length, get_popcount, read_number, set_not_number
from . import blueprint

try:
    from graphviz import Digraph
except ModuleNotFoundError:
    print('"graphviz" not installed: network rendering will not work', file=sys.stderr)


def create_benes_network(size):
    assert size >= 2
    if size == 2:
        return [((0, 0), (1, 1))]
    elif size % 2 == 1:
        inner = create_benes_network(size + 1)

        (network_input_colour, _), (network_output_colour, _) = get_input_output_colours(inner)

        network = []
        replaced = False
        for input_colours, output_colours in inner:
            if not replaced:
                if input_colours[0] == network_input_colour:
                    input_colours = network_output_colour, input_colours[1]
                    replaced = True
                elif input_colours[1] == network_input_colour:
                    input_colours = network_output_colour, input_colours[0]
                    replaced = True
            network.append((input_colours, output_colours))

        return network
    else:
        inner_size = size // 2
        inner = create_benes_network(inner_size)

        inner_output_colours = set()
        inner_input_colours = set()
        for inputs, outputs in inner:
            inner_input_colours |= set(inputs)
            inner_output_colours |= set(outputs)
        inner_all_colours = inner_input_colours | inner_output_colours

        (inner_input_colour, _), (inner_output_colour, _) = get_input_output_colours(inner)

        inner_a_mapping = {inner_input_colour: 1}
        inner_b_mapping = {inner_input_colour: 2}
        i = 3
        for colour in inner_all_colours - {inner_input_colour, inner_output_colour}:
            inner_a_mapping[colour] = i + 0
            inner_b_mapping[colour] = i + 1
            i += 2
        inner_a_mapping[inner_output_colour] = i
        inner_b_mapping[inner_output_colour] = i

        inner_a = remap_colours(inner, inner_a_mapping)
        inner_b = remap_colours(inner, inner_b_mapping)

        return [((0, 0), (1, 2))] * inner_size + inner_a + inner_b


def plot(network, filename=None, engine='dot'):
    g = Digraph(engine=engine, node_attr={'shape': 'rect', 'height': '0.5', 'width': '0.3'}, graph_attr={'rankdir': 'LR'})

    g.node('start', style='invis')
    g.node('end', style='invis')

    (network_input_colour, _), (network_output_colour, _) = get_input_output_colours(network)
    g.edge('start', 'c{}'.format(network_input_colour))
    g.edge('c{}'.format(network_output_colour), 'end')

    INPUT_SIDE = False
    OUTPUT_SIDE = True

    colour_occurences = collections.defaultdict(set)
    for i, (inputs, outputs) in enumerate(network):
        for colour in inputs:
            if colour is None:
                continue
            colour_occurences[colour].add((OUTPUT_SIDE, i))
        for colour in outputs:
            if colour is None:
                continue
            colour_occurences[colour].add((INPUT_SIDE, i))

    for i, _ in enumerate(network):
        g.node(str(i), label='')

    for colour, occurences in colour_occurences.items():
        output_count = sum(side == OUTPUT_SIDE for side, _ in occurences)
        input_count = sum(side == INPUT_SIDE for side, _ in occurences)
        if input_count == 1 and output_count == 1:
            node_a, node_b = [colour for _, colour in sorted(occurences)]
            g.edge(str(node_a), str(node_b), label=str(colour))
        else:
            gen_node = 'c{}'.format(colour)
            g.node(gen_node, shape='none', margin='0.01', width='0', height='0', label=str(colour))
            for side, node in occurences:
                if side == INPUT_SIDE:
                    if output_count != 0:
                        g.edge(str(node), gen_node, arrowhead='none')
                    else:
                        g.edge(str(node), gen_node)
                else:  # Output side
                    g.edge(gen_node, str(node))
    if filename is None:
        g.render('Network', format='png', view=True, cleanup=True)
    else:
        prefix, ext = path.splitext(filename)
        g.render(prefix, format=ext[1:], view=False, cleanup=True)


def remap_colours(network, colour_map):
    assert None not in colour_map

    result = []
    for ((in_a, in_b), (out_a, out_b)) in network:
        result.append(((colour_map.get(in_a, in_a), colour_map.get(in_b, in_b)), (colour_map.get(out_a, out_a), colour_map.get(out_b, out_b))))
    return result


def fix_colours(network):
    input_colours = set()
    output_colours = set()

    for inputs, outputs in network:
        input_colours |= set(inputs)
        output_colours |= set(outputs)

    input_colours.discard(None)
    output_colours.discard(None)

    all_colours = input_colours | output_colours
    remaining_colours = input_colours & output_colours
    input_colours, output_colours = input_colours - output_colours, output_colours - input_colours

    colour_map = {}
    for new_colour, old_colour in zip(range(0, len(all_colours)), input_colours):
        colour_map[old_colour] = new_colour

    for new_colour, old_colour in zip(range(len(input_colours) + len(remaining_colours), len(all_colours)), output_colours):
        colour_map[old_colour] = new_colour

    current = len(input_colours)
    for colour in remaining_colours:
        colour_map[colour] = current
        current += 1

    assert len(colour_map) == len(all_colours)

    return remap_colours(network, colour_map)


def simplify(network, allow_bottleneck=False):
    while True:
        for node in network:
            input_colours, output_colours = node

            if all(colour is None for colour in input_colours):
                network = copy.deepcopy(network)
                network.remove(node)

                colour_map = {}
                for colour in output_colours:
                    if colour is None:
                        continue
                    colour_map[colour] = None
                network = remap_colours(network, colour_map)

                break

            if all(colour is None for colour in output_colours):

                network = copy.deepcopy(network)
                network.remove(node)

                colour_map = {}
                for colour in input_colours:
                    if colour is None:
                        continue
                    colour_map[colour] = None
                network = remap_colours(network, colour_map)

                break

            if any(colour is None for colour in input_colours) and any(colour is None for colour in output_colours):
                network = copy.deepcopy(network)
                network.remove(node)

                input_colour = input_colours[0]
                if input_colour is None:
                    input_colour = input_colours[1]
                    assert input_colour is not None

                output_colour = output_colours[0]
                if output_colour is None:
                    output_colour = output_colours[1]
                    assert output_colour is not None

                network = remap_colours(network, {input_colour: output_colour})

                break
        else:
            colour_sources = {}
            colour_drains = {}
            for i, (input_colours, output_colours) in enumerate(network):
                for colour in output_colours:
                    colour_sources[colour] = i
                for colour in input_colours:
                    colour_drains[colour] = i

            for node_a in network:
                for node_b in network:
                    if node_a == node_b:
                        continue
                    input_a, output_a, input_b, output_b = [set(item) - set([None]) for item in node_a + node_b]

                    if set(output_a) == set(input_b):
                        assert len(output_a) != 1 and len(input_b) != 1
                        network = copy.deepcopy(network)
                        network.remove(node_a)

                        network[network.index(node_b)] = node_a[0], node_b[1]
                        break

                    if allow_bottleneck:
                        if len(output_a) == 1 and len(output_b) == 1 and len(input_a) == 2 and len(input_b) == 2:
                            input_set = set(colour_sources.get(colour, -1) for colour in input_a)
                            if -1 not in input_set and input_set == set(colour_sources.get(colour, -1) for colour in input_b):
                                network = copy.deepcopy(network)
                                network.remove(node_a)

                                network[network.index(node_b)] = input_a, (next(iter(output_a)), next(iter(output_b)))

                                network = remap_colours(network, dict((colour, None) for colour in input_b))
                                break

                        if len(input_a) == 1 and len(input_b) == 1 and len(output_a) == 2 and len(output_b) == 2:
                            output_set = set(colour_drains.get(colour, -1) for colour in output_a)
                            if -1 not in output_set and output_set == set(colour_drains.get(colour, -1) for colour in output_b):
                                network = copy.deepcopy(network)
                                network.remove(node_a)
                                network[network.index(node_b)] = (next(iter(input_a)), next(iter(input_b))), output_a

                                network = remap_colours(network, dict((colour, None) for colour in output_b))
                                break
                else:
                    continue
                break
            else:
                break
    # network = fix_colours(network)
    return network


def calculate_total_colours(network):
    colours = set()
    for inputs, outputs in network:
        colours |= set(inputs + outputs)

    colours.discard(None)

    return len(colours)


def get_input_output_colours(network) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    occurences = collections.defaultdict(lambda: 0)
    for inputs, outputs in network:
        for input in inputs:
            occurences[input] += 1
        for output in outputs:
            occurences[output] -= 1

    occurences.pop(None, None)

    input = None
    output = None

    for colour, count in occurences.items():
        if count > 0:
            assert input is None
            input = colour, count
        elif count < 0:
            assert output is None
            output = colour, -count

    assert input is not None
    assert output is not None

    return input, output


def get_exterior_colours(network):
    input_colours = set()
    output_colours = set()

    for inputs, outputs in network:
        input_colours |= set(inputs)
        output_colours |= set(outputs)

    input_colours.discard(None)
    output_colours.discard(None)

    return input_colours - output_colours, output_colours - input_colours


def calculate_network_size(network):
    input_colours = set()
    output_colours = set()

    for inputs, outputs in network:
        input_colours |= set(inputs)
        output_colours |= set(outputs)

    input_colours.discard(None)
    output_colours.discard(None)

    return len(input_colours - output_colours), len(output_colours - input_colours)


DEFAULT_FILTER = 'deconstruction-planner'


def parse_settings(tokens, node):
    """Per-splitter annotations after the four colours of a network line (all optional, any order):
        in=X      input priority on the input of colour X (the game's input_priority, on the side X enters)
        out=Y     output priority on the output of colour Y (filled first, the rest overflows to the other)
        filter    the dangling output (-1) is blocked by a filter (filter=ITEM names the item; default
                  deconstruction-planner, which never rides a belt), so the layout may put a belt, underground
                  end or splitter in front of it: the game's output_priority goes on that side
    Returns None for a plain (fair) splitter."""
    if not tokens:
        return None
    settings = {}
    for token in tokens:
        key, _, value = token.partition('=')
        if key in ('in', 'out'):
            settings[key] = int(value)
        elif key == 'filter':
            settings['filter'] = value or DEFAULT_FILTER
        else:
            raise ValueError(f'Unknown splitter setting {token!r}')
    check_settings(node, settings)
    return settings


def check_settings(node, settings):
    """A prioritised colour must differ from the other colour on its side, or the side would be ambiguous."""
    if not settings:
        return
    inputs, outputs = node
    for key, side in (('in', inputs), ('out', outputs)):
        if key in settings:
            colour = settings[key]
            if colour not in side or None in side or side[0] == side[1]:
                raise ValueError(f'{key}={colour} on {node}: needs two distinct colours on that side, one of them {colour}')
    if 'filter' in settings:
        if sum(colour is None for colour in outputs) != 1:
            raise ValueError(f'filter on {node}: needs exactly one dangling (-1) output')
        if 'out' in settings:
            raise ValueError(f'filter on {node}: a filtered splitter has one output, so no output priority')


def format_settings(settings):
    if not settings:
        return ''
    tokens = []
    if 'in' in settings:
        tokens.append(f'in={settings["in"]}')
    if 'out' in settings:
        tokens.append(f'out={settings["out"]}')
    if 'filter' in settings:
        tokens.append('filter' if settings['filter'] == DEFAULT_FILTER else f'filter={settings["filter"]}')
    return ' ' + ' '.join(tokens)


def open_network(file, with_settings=False):
    """with_settings: also return the per-splitter settings (a list parallel to the network, None = fair).
    Without it the result is the plain network (a list of nodes); if the file has settings, that list is a
    SetNetwork carrying them, so deduplicate_network / create_balancer / the solution keep them without the caller
    knowing (e.g. tools that build the instance themselves). A file without settings reads exactly as before."""
    require_close = False
    if isinstance(file, str):
        require_close = True
        file = open(file)

    try:
        network = []
        settings = []
        for line in file.readlines():
            line = line.strip()
            if len(line) == 0:
                continue

            if line.startswith('#'):
                continue

            tokens = line.split()
            if len(tokens) < 4:
                raise ValueError('Invalid file')
            colours = [int(colour) for colour in tokens[:4]]

            for i, colour in enumerate(colours):
                if colour == -1:
                    colours[i] = None

            node = (tuple(colours[:2]), tuple(colours[2:]))
            network.append(node)
            settings.append(parse_settings(tokens[4:], node))
    finally:
        if require_close:
            file.close()
    if with_settings:
        return network, settings
    if has_settings(settings):
        # Callers that only know the plain format still carry the settings along (deduplicate_network keeps them)
        network = SetNetwork(network)
        network.settings = settings
    return network


def save_network(file, network, settings=None):
    if isinstance(file, str):
        file = open(file, 'w')

    for i, (input_colours, output_colours) in enumerate(network):
        all_colours = list(input_colours + output_colours)
        for j, colour in enumerate(all_colours):
            if colour is None:
                all_colours[j] = -1

        file.write(' '.join(map(str, all_colours)) + (format_settings(settings[i]) if settings else '') + '\n')


def remap_settings(settings, colour_map):
    result = []
    for entry in settings:
        if entry:
            entry = dict(entry)
            for key in ('in', 'out'):
                if key in entry:
                    entry[key] = colour_map.get(entry[key], entry[key])
        result.append(entry)
    return result


def node_key(node):
    def key(colour):
        return -math.inf if colour is None else colour
    inputs, outputs = node
    return tuple(sorted(inputs, key=key)), tuple(sorted(outputs, key=key))


def network_settings(network, settings):
    """Settings by deduplicated node (the keys of deduplicate_network). Identical splitters are one node type
    in the layout encoding, so they must agree; give a prioritised splitter its own colours otherwise."""
    result = {}
    for node, entry in zip(network, settings or [None] * len(network)):
        k = node_key(node)
        check_settings(node, entry)
        if k in result and result[k] != entry:
            raise ValueError(f'Splitters {k} have different settings ({result[k]} vs {entry}): give them distinct colours')
        result[k] = entry
    return {k: v for k, v in result.items() if v}


def has_settings(settings):
    return any(settings or [])


def flip_network(network):
    return [node[::-1] for node in network]


def pop_count(value: int):
    assert value > 0
    result = 0
    while value != 0:
        result += value & 1
        value >>= 1
    return result


def calculate_cost(network):
    input_colours = set()
    output_colours = set()

    for colours_in, colours_out in network:
        input_colours.update(colours_in)
        output_colours.update(colours_out)
    input_colours.discard(None)
    output_colours.discard(None)

    middle_colours = input_colours & output_colours

    cost = 0
    for node in network:
        for side in node:
            if not any(colour is None for colour in side) and any(colour in middle_colours for colour in side):
                cost += pop_count(side[0] ^ side[1])
    return cost


def optimise_colours(network, solver='g3'):
    starting_cost = calculate_cost(network)

    input_colours = set()
    output_colours = set()
    for colours_in, colours_out in network:
        input_colours.update(colours_in)
        output_colours.update(colours_out)
    input_colours.discard(None)
    output_colours.discard(None)

    edge_colours = input_colours ^ output_colours
    all_colours = input_colours | output_colours

    length = bin_length(len(all_colours))

    clauses = []

    next_lit = 0

    def allocate():
        nonlocal next_lit
        next_lit += 1
        return next_lit
    labels = dict((colour, [allocate() for _ in range(length)]) for colour in all_colours)

    for colour in range(len(all_colours), 1 << length):
        for label in labels.values():
            clauses.append(set_not_number(colour, label))

    differences = {}
    for colour_a, label_a in labels.items():
        for colour_b, label_b in labels.items():
            if colour_a >= colour_b:
                continue

            difference = [allocate() for _ in range(length)]
            for in_a, in_b, out in zip(label_a, label_b, difference):
                clauses += [[in_a, in_b, -out], [-in_a, -in_b, -out], [in_a, -in_b, out], [-in_a, in_b, out]]
            clauses.append(difference)
            differences[colour_a, colour_b] = difference

    counted = []
    for node in network:
        for side in node:
            if any(colour is None for colour in side):
                continue

            if all(colour in edge_colours for colour in side):
                continue

            colour_a, colour_b = side
            if colour_a > colour_b:
                colour_a, colour_b = colour_b, colour_a
            difference = differences[colour_a, colour_b]

            counted += difference

    result_bits = bin_length(len(counted) + 1)

    result = [allocate() for _ in range(result_bits)]
    clauses += get_popcount(counted, result, allocate)

    for cost in range(starting_cost, 1 << result_bits):
        clauses.append(set_not_number(cost, result))

    model = None
    with Solver(name=solver, bootstrap_with=clauses) as s:
        for cost in reversed(range(1, starting_cost)):
            if s.solve():
                model = s.get_model()
            else:
                break

            s.add_clause(set_not_number(cost, result))

    if model is None:
        return network
    else:
        cost += 1

        values = {}
        for lit in model:
            values[abs(lit)] = lit > 0
        colour_map = {}
        for initial_colour, literals in labels.items():
            colour = read_number([values[lit] for lit in literals])
            colour_map[initial_colour] = colour

        result_network = remap_colours(network, colour_map)
        assert cost == calculate_cost(result_network)

        return result_network


def parse_network(tiles, assume_edge_splitter_are_connected=False, with_settings=False):
    """with_settings: also read the splitters' priorities and filters (see parse_settings) and return
    (network, settings). A filter blocks its output_priority side (that output becomes -1). The two outputs of an
    output-priority splitter get distinct colours (they carry different flows), and so do the two outputs of a
    splitter that feeds both inputs of an input-priority splitter."""
    tiles = tiles.T

    colour_mapping = np.full_like(tiles, None)
    splitters = set()
    current_colour = 0

    def next_tile(pos, is_forward):
        tile = tiles[pos]
        assert tile is not None

        if isinstance(tile, Belt):
            if is_forward:
                direction = tile.output_direction
            else:
                direction = tile.input_direction.reverse

            x, y = pos
            return x + direction.dx, y + direction.dy
        elif isinstance(tile, Splitter):
            assert False
        elif isinstance(tile, UndergroundBelt):
            if tile.is_input:
                if is_forward:
                    dx, dy = tile.direction.vec
                    while True:
                        pos = pos[0] + dx, pos[1] + dy
                        assert pos[0] >= 0 and pos[0] < tiles.shape[0] and pos[1] >= 0 and pos[1] < tiles.shape[1]

                        current_tile = tiles[pos]
                        assert current_tile != UndergroundBelt(tile.direction, True)

                        if current_tile == UndergroundBelt(tile.direction, False):
                            break
                    return pos
                else:
                    dx, dy = tile.input_direction.vec
                    x, y = pos
                    return x - dx, y - dy
            else:
                if is_forward:
                    dx, dy = tile.output_direction.vec
                    x, y = pos
                    return x + dx, y + dy
                else:
                    dx, dy = tile.direction.vec
                    while True:
                        pos = pos[0] - dx, pos[1] - dy
                        assert pos[0] >= 0 and pos[0] < tiles.shape[0] and pos[1] >= 0 and pos[1] < tiles.shape[1]

                        current_tile = tiles[pos]
                        assert current_tile != UndergroundBelt(tile.direction, False)

                        if current_tile == UndergroundBelt(tile.direction, True):
                            break
                    return pos
        else:
            raise RuntimeError(f'Unknown tile type: {tile}')

    def trace(colour: int, pos: Tuple[int, int], is_forward: bool):
        if pos[0] < 0 or pos[0] >= tiles.shape[0] or pos[1] < 0 or pos[1] >= tiles.shape[1]:
            return

        tile = tiles[pos]
        if tile == EmptyTile():
            return

        if isinstance(tile, Splitter):
            return

        if colour_mapping[pos] is not None:
            assert colour_mapping[pos] == colour
            return

        colour_mapping[pos] = colour

        trace(colour, next_tile(pos, is_forward), is_forward)

    for x, row in enumerate(tiles):
        for y, tile in enumerate(row):
            if isinstance(tile, Splitter) and tile.is_head:
                splitters.add((x, y))

            if colour_mapping[x, y] is not None:
                continue

            if tile == EmptyTile() or tile == FillerTile():
                colour_mapping[x, y] = -1
                continue

            if not isinstance(tile, Splitter):
                colour = current_colour
                current_colour += 1
                colour_mapping[x, y] = colour

                trace(colour, next_tile((x, y), True), True)
                trace(colour, next_tile((x, y), False), False)

    network = []
    sides = []  # per splitter: (input priority index, output priority index), index 0 = right half, 1 = left
    for x, y in splitters:
        tile = tiles[x, y]
        assert isinstance(tile, Splitter) and tile.is_head
        dx0, dy0 = tile.direction.vec
        dx1, dy1 = tile.direction.next.vec
        side_index = {'right': 0, 'left': 1, None: None}
        blocked = side_index[tile.output_priority] if tile.filter is not None else None
        sides.append((side_index[tile.input_priority], None if tile.filter is not None else side_index[tile.output_priority],
                      tile.filter))

        node = []
        for is_output, offset_a in enumerate([(-dx0, -dy0), (+dx0, +dy0)]):
            node.append([])
            for side, offset_b in enumerate([(0, 0), (dx1, dy1)]):
                if is_output and side == blocked:
                    node[-1].append(None)
                    continue
                pos = x + offset_a[0] + offset_b[0], y + offset_a[1] + offset_b[1]
                if pos[0] < 0 or pos[0] >= tiles.shape[0] or pos[1] < 0 or pos[1] >= tiles.shape[1]:
                    if assume_edge_splitter_are_connected:
                        node[-1].append(current_colour)
                        current_colour += 1
                    else:
                        node[-1].append(None)
                    continue

                offset_tile = tiles[pos]
                if offset_tile is None or isinstance(offset_tile, (EmptyTile, FillerTile)):
                    node[-1].append(None)
                    continue

                if isinstance(offset_tile, Splitter):
                    if not is_output and offset_tile.filter is not None and offset_tile.output_priority == offset_tile.side:
                        colour = None  # behind us is the blocked half of a filtered splitter
                    elif offset_tile.direction == tile.direction:
                        if is_output:
                            colour_pos = pos
                        else:
                            colour_pos = x + offset_b[0], y + offset_b[1]

                        colour = colour_mapping[colour_pos]
                        if colour is None:
                            colour = current_colour
                            current_colour += 1
                            colour_mapping[colour_pos] = colour
                    else:
                        colour = None
                else:
                    if is_output:
                        colour = colour_mapping[pos] if tile.direction == offset_tile.input_direction else None
                    else:
                        colour = colour_mapping[pos] if tile.direction == offset_tile.output_direction else None
                node[-1].append(colour)
            node[-1] = tuple(node[-1])
        network.append(tuple(node))

    network_input_colours, network_output_colours = get_exterior_colours(network)

    # Output pairs that must keep distinct colours (only with settings, so plain parses are unchanged)
    split = set()
    if with_settings:
        producer = {colour: j for j, (_, outputs) in enumerate(network) for colour in outputs if colour is not None}
        for j, (inputs, _) in enumerate(network):
            if sides[j][1] is not None:
                split.add(j)
            if sides[j][0] is not None and None not in inputs:
                a, b = (producer.get(colour) for colour in inputs)
                if a is not None and a == b:
                    split.add(a)

    mapping = {}
    for colour in network_input_colours:
        mapping[colour] = 0
    current_colour = 1
    for j, (_, output_colours) in enumerate(network):
        for colour in output_colours:
            mapping[colour] = current_colour
            if j in split and colour is not None:
                current_colour += 1
        if j not in split:
            current_colour += 1
    for colour in network_output_colours:
        mapping[colour] = current_colour

    mapping.pop(None, None)
    network = remap_colours(network, mapping)

    if not with_settings:
        return network

    settings = []
    for (inputs, outputs), (in_side, out_side, item) in zip(network, sides):
        entry = {}
        if in_side is not None and None not in inputs:  # input priority on a one-input splitter does nothing
            entry['in'] = inputs[in_side]
        if out_side is not None and None not in outputs:
            entry['out'] = outputs[out_side]
        if item is not None and None in outputs:
            entry['filter'] = item
        check_settings((inputs, outputs), entry)
        settings.append(entry or None)
    return network, settings


class SetNetwork(list):
    """A network (list of nodes) that remembers its per-splitter settings (open_network). Plain list otherwise."""
    settings = None


class SetCounter(collections.Counter):
    """deduplicate_network of a SetNetwork: the Counter of node types plus `settings` ({node key: entry}), which
    belt_balancer.create_balancer picks up (filters in the encoding, priorities written into each solution)."""
    settings = None


def deduplicate_network(network):
    counter = collections.Counter(node_key(node) for node in network)
    settings = getattr(network, 'settings', None)
    if settings and has_settings(settings):
        counter = SetCounter(counter)
        counter.settings = network_settings(network, settings)
    return counter


def tidy_network(network, settings=None):
    """settings: optional per-splitter settings, remapped and reordered with the network; then (network, settings)
    is returned."""
    fixed = fix_colours(network)
    if settings is not None:
        colour_map = {}
        for old_node, new_node in zip(network, fixed):
            for old_side, new_side in zip(old_node, new_node):
                for old, new in zip(old_side, new_side):
                    colour_map[old] = new
        settings = remap_settings(settings, colour_map)
    network = fixed

    input_colours, output_colours = get_exterior_colours(network)

    def none_key(v):
        return -1 if v is None else v

    for i, node in enumerate(network):
        network[i] = tuple(tuple(sorted(side, key=none_key)) for side in node)

    order = list(range(len(network)))
    order.sort(key=lambda i: none_key(network[i][0][0]))
    order.sort(key=lambda i: sum(colour in output_colours for colour in network[i][1]) - sum(colour in input_colours for colour in network[i][0]))
    network = [network[i] for i in order]

    if settings is None:
        return network
    return network, [settings[i] for i in order]


def main():
    parser = argparse.ArgumentParser(description='Manipulate belt balancer networks')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    create_parser = subparsers.add_parser('create', help='Create a belt balancer network')
    create_parser.add_argument('network', type=argparse.FileType('w'), help='Output network destination')
    create_parser.add_argument('size', type=int, help='Size of the generated balancer')

    # optimise_parser = subparsers.add_parser('optimise', help='Optimises a network colour layout (not guaranteed to reduce solver times)')
    # optimise_parser.add_argument('input_network', type=argparse.FileType('r'), help='Input network destination')
    # optimise_parser.add_argument('output_network', type=argparse.FileType('w'), help='Output network destination')

    flip_parser = subparsers.add_parser('flip', help='Flips a network around (e.g 3 to 4 network -> 4 to 3 network)')
    flip_parser.add_argument('input_network', type=argparse.FileType('r'), help='Input network destination')
    flip_parser.add_argument('output_network', type=argparse.FileType('w'), help='Output network destination')

    render_parser = subparsers.add_parser('render', help='Render a belt balancer network')
    render_parser.add_argument('network', type=argparse.FileType('r'), help='Network to render')
    render_parser.add_argument('output', nargs='?', type=str, help='File to render to')
    render_parser.add_argument('--engine', type=str, default='dot', help='Layout command for rendering')

    parse_parser = subparsers.add_parser('parse', help='Reads a belt balancer network from a belt balancer')
    parse_parser.add_argument('output', type=argparse.FileType('w'), help='Network output file')
    parse_parser.add_argument('--assume-valid-output', action='store_true', help='Assume splitters facing outside the balancer bounds are an input/output')

    args = parser.parse_args()

    if args.mode == 'create':
        if args.size <= 1:
            raise RuntimeError('Input size too small')

        network = create_benes_network(args.size)

        with args.network:
            save_network(args.network, network)
    elif args.mode == 'render':
        with args.network:
            network = open_network(args.network)

        plot(network, args.output, args.engine)
    elif args.mode == 'parse':
        tiles = np.array(json.loads(input()))
        for i, row in enumerate(tiles):
            for j, item in enumerate(row):
                tiles[i, j] = blueprint.read_tile(item)

        network, settings = parse_network(tiles, args.assume_valid_output, with_settings=True)
        network, settings = tidy_network(network, settings)

        with args.output:
            save_network(args.output, network, settings)
    else:
        with args.input_network:
            in_network = open_network(args.input_network)

        if args.mode == 'flip':
            out_network = flip_network(in_network)
            out_network = out_network[::-1]
        else:
            assert False

        with args.output_network:
            save_network(args.output_network, out_network)


if __name__ == '__main__':
    main()
