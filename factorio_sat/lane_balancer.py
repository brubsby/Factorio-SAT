"""Lane-level belt balancer layouts (branch `lanes`).

The belt-level encoder gives every tile one colour: the network edge its belt carries. Here every tile has two, one
per lane (`colour` = left lane, `colour_r` = right lane, relative to the tile's direction), and the network is a LANE
network (factorio-balancer-sat docs/lane-model.md): every physical splitter is two independent lane splitters.

Lane mechanics encoded (measured in game, docs/lane-model.md L1-L11):
  * belts, curves and undergrounds keep lanes (left stays left);
  * a splitter half carries a lane pair; each lane of the splitter is its own 2x2 splitter (the network gives the
    two lane nodes of every splitter);
  * an underground ENTRANCE may be fed from the side by a belt, an exit or a splitter half (tile state `side_in[d]`:
    the neighbour behind it along d outputs into it crosswise). Only the feeder's back lane (the one on the entrance's upstream side) enters, onto the near
    lane; the front lane is blocked. Such an entrance takes nothing from behind (`no_behind`), so no lane receives two
    inflows and no merge is needed. This is the lane-swap trick of Raynquist's lane balancers (L8, L10).
  * an entrance may be followed directly by its exit (gap 0): the belt-level encoder needs a segment tile between them.
Splitter halves carry their output lanes in colour_o / colour_or (their colour / colour_r are the input lanes), so a
splitter output can feed a side-fed entrance like any other feeder.
Belt sideloads are priority-merge nodes (merge_l / merge_r, network lines `M x0 x1 x2 y`); their feeder may be a belt,
an exit or a splitter half. Lone exits (no entrance) take a side feeder's front lane.
Branch `unified` adds, each off by default so that lane networks without them build the CNF of branch `lanes`:
  * merges at underground ends (`ug_merges`, auto-enabled by a merge line ending in `ug`): an entrance fed from behind
    AND from a side (near lane = merge(behind lane, side back lane); the side front lane is blocked, L8), and an exit
    with a live tunnel AND a side feeder (near lane = merge(tunnel lane, side front lane); the side back lane is
    blocked, L9). These are what a splitter's unused output does to a crosswise underground end in the game;
  * mixed-tier undergrounds with weaving (`tiers`, as solver.Grid): every tier has its own two-lane tunnel layer;
  * priority and filter splitters (`settings`, network annotations `in=X out=Y filter` after the 8 colours, X / Y a
    lane colour of that side): priorities are set after solving on the side where the colour landed (both lanes of a
    splitter share them, L3); a filtered splitter's outputless half may have anything in front of it;
  * belt-level networks (`belt_to_lane`, CLI --belt): each belt edge becomes a lane pair, so the belt encoder's
    networks (with their annotations) run through this lane-faithful encoder.
A belt sideload is a priority merge into one lane, so a layout with one is not a belt-level splitter network: check
its lane network (factorio-balancer-sat tools/lanes/lanecheck.py, tools/lanes/beltcheck.py for belt-level BAL / TU
on the lane network).

Reserved colours: 0 = EMPTY (a lane that never carries items: the unfed side of a side-fed entrance, or the absent lane
of a splitter half whose other lane is used), 1 = DEAD (a splitter lane output that the network leaves dangling while
the half's other lane is used; it may only end at the blocked lane of a side-feed), 2 = network input, 3 = network
output, 4.. = internal lane edges.

Lane network file: one physical splitter per line, `aL bL cL dL aR bR cR dR`: the left-lane node's two inputs and two
outputs, then the right-lane node's, -1 = no port. Within a node the first input/output is the splitter's LEFT half
(relative to its facing), the second its right half; the solver may mirror a splitter (swap halves) but always for both
lanes together.

    python -m factorio_sat.lane_balancer NETWORK WIDTH HEIGHT [--underground-length 4] [--all] [--blueprint]
        [--belt] [--tiers yellow,red] [--tier-max red=2] [--filter-dangling] [--contact] [--ug-merges] [--free-ends]
"""
import argparse
import collections
import json
import sys

from pysat.card import EncType

from . import blueprint
from .belt_balancer import setup_balancer_input, setup_balancer_output
from .cardinality import library_equals, quadratic_one
from .direction import Axis, Direction
from .solver import Grid
from .tile import UNDERGROUND_TIERS, parse_tier
from .template import ArrayTemplate, BoolTemplate, NumberTemplate, OneHotTemplate
from .cardinality import quadratic_amo
from .util import implies, invert_components, literals_different, set_maximum, set_number, set_numbers_equal

EMPTY, DEAD, INPUT, OUTPUT = 0, 1, 2, 3


def open_lane_network(path, with_merges=False, with_settings=False):
    """-> list of splitter nodes ((L_in, L_out), (R_in, R_out)), each a pair of 2-tuples with None for -1; with
    with_merges also the belt-sideload merges `M x0 x1 x2 y` (x0 = the rank-0 lane: straight / behind / tunnel,
    x1 = rank 1, x2 = rank 2, y = the merged lane; -1 = none; on a belt x1 is the side belt's back lane and x2 its
    front lane; a line ending in `ug` allows merges at underground ends, see LaneGrid). with_settings also returns
    {node: settings} (network.parse_settings tokens after the 8 colours: in=X out=Y filter[=ITEM]) and whether
    underground-end merges are on: (nodes, merges, settings, ug_merges)."""
    from .network import parse_settings
    nodes, merges, settings, ug = [], [], {}, False
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('M'):
            tok = line.split()
            merges.append(tuple(None if int(t) == -1 else int(t) for t in tok[1:5]))
            ug |= 'ug' in tok[5:]
            continue
        tok = line.split()
        c = [None if int(t) == -1 else int(t) for t in tok[:8]]
        node = ((tuple(c[0:2]), tuple(c[2:4])), (tuple(c[4:6]), tuple(c[6:8])))
        nodes.append(node)
        if len(tok) > 8:
            entry = parse_lane_settings(tok[8:], node)
            if node in settings and settings[node] != entry:
                raise ValueError(f'identical splitters {node} with different settings: give them distinct colours')
            settings[node] = entry
    if with_settings:
        return nodes, merges, settings, ug
    return (nodes, merges) if with_merges else nodes


def parse_lane_settings(tokens, node):
    """Annotations of one lane splitter line. in=X / out=Y name a colour on one input / output half (either lane);
    filter needs one output half with no lanes at all."""
    settings = {}
    for token in tokens:
        key, _, value = token.partition('=')
        if key in ('in', 'out'):
            settings[key] = int(value)
        elif key == 'filter':
            from .network import DEFAULT_FILTER
            settings['filter'] = value or DEFAULT_FILTER
        else:
            raise ValueError(f'Unknown splitter setting {token!r}')
    (Lin, Lout), (Rin, Rout) = node
    for key, (lc, rc) in (('in', (Lin, Rin)), ('out', (Lout, Rout))):
        if key in settings:
            halves = [{lc[j], rc[j]} - {None} for j in (0, 1)]
            hits = [j for j in (0, 1) if settings[key] in halves[j]]
            if len(hits) != 1:
                raise ValueError(f'{key}={settings[key]} on {node}: must name a colour of exactly one half')
    if 'filter' in settings:
        if sum(Lout[j] is None and Rout[j] is None for j in (0, 1)) != 1:
            raise ValueError(f'filter on {node}: needs exactly one output half with no lanes')
        if 'out' in settings:
            raise ValueError(f'filter on {node}: no output priority with a filter')
    return settings


def belt_to_lane(network, settings=None):
    """A belt-level Factorio-SAT network (list of ((a, b), (c, d)) with None) -> lane nodes and lane settings. Every
    belt edge becomes a lane pair: the network input -> (INPUT, INPUT), the output -> (OUTPUT, OUTPUT), any other
    colour k -> two fresh lane colours; in=X / out=Y move to X's / Y's left-lane colour. settings: {node: entry}
    for the original nodes (network.network_settings keys are node_key()-sorted, so pass a dict keyed either way)."""
    from .network import get_input_output_colours, node_key
    (ic, _), (oc, _) = get_input_output_colours(list(network))
    lane = {ic: (INPUT, INPUT), oc: (OUTPUT, OUTPUT)}
    nxt = OUTPUT + 1
    for (ins, outs) in network:
        for c in ins + outs:
            if c is not None and c not in lane:
                lane[c] = (nxt, nxt + 1)
                nxt += 2
    nodes, lsettings = [], {}
    settings = settings or {}
    for node in network:
        (a, b), (c, d) = node
        L = lambda x: None if x is None else lane[x][0]
        R = lambda x: None if x is None else lane[x][1]
        ln = (((L(a), L(b)), (L(c), L(d))), ((R(a), R(b)), (R(c), R(d))))
        nodes.append(ln)
        entry = settings.get(node) or settings.get(node_key(node))
        if entry:
            e = dict(entry)
            for key in ('in', 'out'):
                if key in e:
                    e[key] = lane[e[key]][0]
            lsettings[ln] = e
    return nodes, lsettings


def lane_counts(nodes, merges=()):
    n = sum(c == INPUT for node in nodes for lane in node for c in lane[0]) + sum(c == INPUT for mk in merges for c in mk[:3])
    m = sum(c == OUTPUT for node in nodes for lane in node for c in lane[1]) + sum(mk[3] == OUTPUT for mk in merges)
    assert n % 2 == 0 and m % 2 == 0, (n, m)
    return n // 2, m // 2


class LaneGrid(Grid):
    def __init__(self, width, height, colours, underground_length=4, extras=None, merges=(), tiers=None, ug_merges=False):
        """merges: the distinct belt-sideload merge nodes (x0, x1, x2, y) of the network (see open_lane_network).
        tiers: mixed-tier undergrounds (solver.Grid); ug_merges: merges may also sit at underground ends."""
        bits = (colours - 1).bit_length()
        self.merges = list(merges)
        self.ug_merges = bool(ug_merges)
        n_tiers = len({parse_tier(t) for t in tiers}) if tiers else 1
        ex = dict(extras or {})
        ex.update({
            'colour_r': NumberTemplate(bits),
            'colour_r_ux': NumberTemplate(bits),
            'colour_r_uy': NumberTemplate(bits),
            # a splitter half's OUTPUT lanes (its colour / colour_r are its input lanes)
            'colour_o': NumberTemplate(bits),
            'colour_or': NumberTemplate(bits),
            'side_in': ArrayTemplate(BoolTemplate(), (4,)),
            'no_behind': BoolTemplate(),
            # an underground exit with no entrance, fed only from the side (a one-tile lane filter)
            'lone': BoolTemplate(),
            # merge_l[j] / merge_r[j]: this belt's left / right lane is the output of belt-sideload merge j; mL / mR: any
            'merge_l': ArrayTemplate(BoolTemplate(), (max(len(self.merges), 1),)),
            'merge_r': ArrayTemplate(BoolTemplate(), (max(len(self.merges), 1),)),
            'mL': BoolTemplate(),
            'mR': BoolTemplate(),
        })
        for t in range(1, n_tiers):          # the right lane of every extra tier's tunnel layer
            ex[f'colour_r_ux_{t}'] = NumberTemplate(bits)
            ex[f'colour_r_uy_{t}'] = NumberTemplate(bits)
        super().__init__(width, height, colours, underground_length, ex, tiers=tiers)
        self.relax_edge_inputs()
        self.side_feed_rules()

    def lane_u(self, tile, t, axis, lane):
        """Colour of lane (0 = left, 1 = right) of the tier-t tunnel along axis at this tile."""
        if lane == 0:
            return self.colour_u(tile, t, axis)
        if t == 0:
            return tile.colour_r_ux if axis == Axis.HORIZONTAL else tile.colour_r_uy
        return getattr(tile, f'colour_r_ux_{t}' if axis == Axis.HORIZONTAL else f'colour_r_uy_{t}')

    @staticmethod
    def facing(tile, d):
        """'tile is a belt-like end facing d' literals per kind: belt / exit by their output, entrance by its input."""
        return {'belt': [tile.is_belt, tile.output_direction[d]], 'in': [tile.is_underground_in, tile.input_direction[d]],
                'out': [tile.is_underground_out, tile.output_direction[d]]}

    def relax_edge_inputs(self):
        """Grid makes every edge tile whose input points inward an input port (input_direction -> is_input). A side-fed
        entrance at the edge (no_behind) faces inward without being fed through the edge: weaken those clauses to
        input_direction -> is_input or no_behind (Raynquist's 1-1 TU lane corner balancer has one)."""
        edge = {}
        for x in range(self.width):
            for y in range(self.height):
                t = self.get_tile_instance(x, y)
                for d in Direction:
                    edge[(-t.input_direction[d], t.is_input)] = t.no_behind
        out = []
        for c in self.clauses:
            if len(c) == 2 and (c[0], c[1]) in edge:
                c = c + [edge[(c[0], c[1])]]
            out.append(c)
        self.clauses[:] = out

    def colour_ranges(self, tile):
        ranges = [tile.colour, tile.colour_ux, tile.colour_uy, tile.colour_r, tile.colour_r_ux, tile.colour_r_uy,
                  tile.colour_o, tile.colour_or]
        for t in range(1, self.tier_count):
            ranges += [self.lane_u(tile, t, ax, lane) for lane in (0, 1) for ax in (Axis.HORIZONTAL, Axis.VERTICAL)]
        return ranges

    # ------------------------------------------------------------------ geometry
    def side_feed_rules(self):
        for x in range(self.width):
            for y in range(self.height):
                b = self.get_tile_instance(x, y)
                for dd in Direction:
                    s = b.side_in[dd]
                    a = self.get_tile_instance_offset(x, y, -dd.dx, -dd.dy)     # the feeder, behind b along dd
                    if a is None:
                        self.clauses.append([-s])
                        continue
                    # b is a side-fed entrance (taking nothing from behind), a belt (sideload) or a lone exit; its own axis
                    # is across dd
                    kinds = [b.is_underground_in, b.is_belt, b.lone] + ([b.is_underground_out] if self.ug_merges else [])
                    self.clauses += implies([s], [kinds, [a.output_direction[dd]],
                                                  [-b.input_direction[dd]], [-b.input_direction[dd.reverse]],
                                                  [-b.output_direction[dd]], [-b.output_direction[dd.reverse]]])
                    if self.ug_merges:
                        # a side-fed entrance with a feeder behind, or a side-fed exit with a tunnel: a merge
                        self.clauses.append([-s, -b.is_underground_in, b.no_behind, b.mL, b.mR])
                        self.clauses.append([-s, -b.is_underground_out, b.lone, b.mL, b.mR])
                    else:
                        self.clauses.append([-s, -b.is_underground_in, b.no_behind])
                    # a sideloaded belt runs straight (a lone side feeder with nothing behind makes it a curve instead)
                    for d in (dd.next, dd.prev):
                        self.clauses.append([-s, -b.is_belt, -b.output_direction[d], b.input_direction[d]])
                self.clauses += implies([b.no_behind], [[b.is_underground_in, b.is_belt], [-b.is_input]])
                self.clauses.append([-b.no_behind, -b.is_underground_in, *b.side_in])
                for d in Direction:      # a belt fed from neither behind nor ... must be fed from both sides
                    self.clauses += implies([b.no_behind, b.is_belt, b.output_direction[d]],
                                            [[b.side_in[d.prev]], [b.side_in[d.next]]])
                self.clauses += implies([b.lone], [[b.is_underground_out], list(b.side_in)])
                # belt merges: a sideload onto the left (right) lane <=> exactly one merge_l (merge_r) node
                K = len(self.merges)
                if K == 0:
                    self.clauses += [[-b.mL], [-b.mR], [-b.merge_l[0]], [-b.merge_r[0]]]
                    for d in Direction:
                        self.clauses.append([-b.is_belt, -b.output_direction[d], -b.side_in[d.prev]])
                        self.clauses.append([-b.is_belt, -b.output_direction[d], -b.side_in[d.next]])
                    continue
                for ml, m_any, side in ((b.merge_l, b.mL, 'prev'), (b.merge_r, b.mR, 'next')):
                    self.clauses += quadratic_amo(list(ml))
                    self.clauses.append([-m_any, *ml])
                    host = [b.is_belt] + ([b.is_underground_in, b.is_underground_out] if self.ug_merges else [])
                    for j in range(K):
                        self.clauses += [[-ml[j], m_any], [-ml[j], *host]]
                    for d in Direction:
                        sd = b.side_in[getattr(d, side)]
                        self.clauses.append([-b.is_belt, -b.output_direction[d], -sd, m_any])
                        self.clauses.append([-m_any, -b.output_direction[d], sd])
                        if self.ug_merges:
                            self.clauses.append([-b.is_underground_in, -b.input_direction[d], -sd, b.no_behind, m_any])
                            self.clauses.append([-m_any, -b.is_underground_in, -b.input_direction[d], sd])
                            self.clauses.append([-b.is_underground_out, -b.output_direction[d], -sd, b.lone, m_any])
                    if self.ug_merges:
                        # a lone exit or a side-only entrance has a single inflow per lane: no merge
                        self.clauses += [[-m_any, -b.lone], [-m_any, -b.no_behind, b.is_belt]]

    def prevent_intersection(self, forbid_dangling_fronts: bool = True):
        """Grid.prevent_intersection with the one-to-one output/input rule relaxed for side-fed entrances."""
        if forbid_dangling_fronts:
            self.forbid_dangling_fronts()
        for direction in Direction:
            for tile_a, tile_b in self.iterate_tile_lines(direction.vec, 2):
                if tile_b is None:
                    continue
                out_a, in_b = tile_a.output_direction[direction], tile_b.input_direction[direction]
                self.clauses += [
                    [-out_a, in_b, tile_b.side_in[direction]],        # an output goes into b from behind or crosswise
                    [-in_b, tile_b.no_behind, out_a],                 # b takes from behind -> a feeds it
                    [-in_b, -tile_b.no_behind, -out_a],               # a side-fed entrance takes nothing from behind
                ]
                relax = self.front_relax.get(tile_a.is_splitter)
                relaxed = [] if relax is None else [-relax]
                # a belt crosswise in front of a splitter half is only allowed as a sideload target of that half
                self.clauses += implies([tile_a.input_direction[direction], tile_a.is_splitter, -tile_b.is_splitter, *relaxed], [
                    [-tile_b.input_direction[direction.next], -tile_b.output_direction[direction.next], tile_b.side_in[direction]],
                    [-tile_b.input_direction[direction.prev], -tile_b.output_direction[direction.prev], tile_b.side_in[direction]],
                    [-tile_b.input_direction[direction.next], -tile_b.output_direction[direction]],
                    [-tile_b.input_direction[direction.prev], -tile_b.output_direction[direction]],
                    [-tile_b.input_direction[direction.reverse], -tile_b.output_direction[direction.next]],
                    [-tile_b.input_direction[direction.reverse], -tile_b.output_direction[direction.prev]],
                ])
                # an entrance (or a lone exit) crosswise in front of a splitter half is side-fed by it (L11): that half
                # must output into it; other exits crosswise in front of a splitter stay forbidden
                fed = [tile_b.lone] + ([tile_b.mL, tile_b.mR] if self.ug_merges else [])
                for splitter_side in (tile_a.input_direction[direction], tile_a.output_direction[direction]):
                    self.clauses += implies([splitter_side, tile_a.is_splitter, -tile_b.is_splitter, *relaxed], [
                        [-tile_b.is_underground_in, -tile_b.input_direction[direction.next], out_a],
                        [-tile_b.is_underground_in, -tile_b.input_direction[direction.prev], out_a],
                        # a crosswise exit: only a lone exit (or, with ug_merges, a merging exit), side-fed by this half
                        [-tile_b.is_underground_out, -tile_b.output_direction[direction.next], *fed],
                        [-tile_b.is_underground_out, -tile_b.output_direction[direction.prev], *fed],
                        [-tile_b.is_underground_out, -tile_b.output_direction[direction.next], out_a],
                        [-tile_b.is_underground_out, -tile_b.output_direction[direction.prev], out_a],
                    ])

    def prevent_bad_undergrounding(self):
        """Grid.prevent_bad_undergrounding, but an entrance may be followed directly by its exit (gap 0).
        The belt-level encoder never needs that (a belt does the same), so it requires a segment tile in between; the
        lane trick does: a side-fed entrance with its exit right behind it is a one-tile lane filter (Raynquist's 2-2
        and 4-4 lane balancers).
        Per tier, as solver.Grid: an end only pairs through, and only conflicts with, its own tier's segments, so tiers
        nest and cross (weaving). With one tier this is the single-tier encoding of branch `lanes`, clause for clause."""
        for direction in Direction:
            reverse_dir = direction.reverse
            dx, dy = direction.vec
            for x in range(self.width):
                for y in range(self.height):
                    a = self.get_tile_instance(x, y)
                    nxt = self.get_tile_instance_offset(x, y, +dx, +dy)
                    prv = self.get_tile_instance_offset(x, y, -dx, -dy)
                    for t in range(self.tier_count):
                        ug_a = self.ug(a, t)
                        tier_a = self.tier_is(a, t)
                        self.clauses += implies([a.is_underground_in, a.input_direction[direction], *tier_a],
                                                [[-ug_a[direction]], [-ug_a[reverse_dir]]])
                        self.clauses += implies([a.is_underground_out, a.output_direction[direction], *tier_a],
                                                [[-ug_a[direction]], [-ug_a[reverse_dir]]])
                        if nxt is not None:
                            adj = self.allocate_variable()       # a = entrance, nxt = its exit, no gap
                            self.clauses += [[-adj, nxt.is_underground_out], [-adj, nxt.output_direction[direction]]]
                            self.clauses += [[-adj, lit] for lit in self.tier_is(nxt, t)]
                            self.clauses += implies([a.is_underground_in, a.input_direction[direction], *tier_a],
                                                    [[self.ug(nxt, t)[direction], adj]])
                        elif t == 0:
                            self.clauses += [[-a.is_underground_in, -a.input_direction[direction]]]
                        if prv is not None:
                            adj = self.allocate_variable()
                            self.clauses += [[-adj, prv.is_underground_in], [-adj, prv.input_direction[direction]]]
                            self.clauses += [[-adj, lit] for lit in self.tier_is(prv, t)]
                            self.clauses += implies([a.is_underground_out, a.output_direction[direction], *tier_a],
                                                    [[self.ug(prv, t)[direction], adj, a.lone]])
                            # a lone exit has no tunnel: no segment and no entrance of its tier right behind it
                            self.clauses += implies([a.lone, a.output_direction[direction], *tier_a],
                                                    [[-self.ug(prv, t)[direction]],
                                                     [-prv.is_underground_in, -prv.input_direction[direction],
                                                      *[-lit for lit in self.tier_is(prv, t)]]])
                        elif t == 0:
                            self.clauses += [[-a.is_underground_out, -a.output_direction[direction], a.lone]]
                        if nxt is not None:
                            self.clauses += implies([ug_a[direction], -self.ug(nxt, t)[direction]],
                                                    [[nxt.is_underground_out], [nxt.output_direction[direction]],
                                                     *([lit] for lit in self.tier_is(nxt, t))])
                            # gap 0: the exit takes the entrance's lanes directly (a merging exit's near lane is
                            # the merge output instead: merge_colours)
                            for qa, qb, mx in ((a.colour, nxt.colour, nxt.mL), (a.colour_r, nxt.colour_r, nxt.mR)):
                                self.clauses += implies([a.is_underground_in, a.input_direction[direction],
                                                         nxt.is_underground_out, nxt.output_direction[direction],
                                                         *tier_a, *self.tier_is(nxt, t),
                                                         *([-mx] if self.ug_merges else [])],
                                                        set_numbers_equal(qa, qb))
                        if prv is not None:
                            self.clauses += implies([ug_a[direction], -self.ug(prv, t)[direction]],
                                                    [[prv.is_underground_in], [prv.input_direction[direction]],
                                                     *([lit] for lit in self.tier_is(prv, t))])

    # ------------------------------------------------------------------ lane colours
    def prevent_bad_colouring(self):
        ugm = self.ug_merges
        for direction in Direction:
            dx, dy = direction.vec
            for x in range(self.width):
                for y in range(self.height):
                    a = self.get_tile_instance(x, y)
                    b = self.get_tile_instance_offset(x, y, dx, dy)
                    if b is None:
                        continue
                    out_a = a.output_direction[direction]
                    s = b.side_in[direction]
                    # a's outgoing lanes: its own (colour, colour_r), or a splitter half's output lanes
                    for kind, (aL, aR) in (([-a.is_splitter], (a.colour, a.colour_r)), ([a.is_splitter], (a.colour_o, a.colour_or))):
                        # from behind: lanes kept (belts, curves, into splitters / entrances, out of exits), except a
                        # lane that is a merge output (its behind lane is the merge's rank-0 input instead)
                        self.clauses += implies([out_a, *kind, b.input_direction[direction], -b.mL], set_numbers_equal(aL, b.colour))
                        self.clauses += implies([out_a, *kind, b.input_direction[direction], -b.mR], set_numbers_equal(aR, b.colour_r))
                        # crosswise into an entrance (b faces direction.next: a comes from b's left; .prev: its right):
                        # the feeder's back lane enters, its front lane is blocked (with ug_merges the entered lane
                        # may be a merge output instead: merge_colours)
                        mL = [-b.mL] if ugm else []
                        mR = [-b.mR] if ugm else []
                        self.clauses += implies([out_a, *kind, s, b.is_underground_in, b.input_direction[direction.next], *mL],
                                                set_numbers_equal(b.colour, aR) + set_maximum(DEAD, aL))
                        self.clauses += implies([out_a, *kind, s, b.is_underground_in, b.input_direction[direction.prev], *mR],
                                                set_numbers_equal(b.colour_r, aL) + set_maximum(DEAD, aR))
                        # crosswise into a lone exit: the feeder's FRONT lane enters, its back lane is blocked
                        self.clauses += implies([out_a, *kind, s, b.is_underground_out, b.output_direction[direction.next], *mL],
                                                set_numbers_equal(b.colour, aL) + set_maximum(DEAD, aR))
                        self.clauses += implies([out_a, *kind, s, b.is_underground_out, b.output_direction[direction.prev], *mR],
                                                set_numbers_equal(b.colour_r, aR) + set_maximum(DEAD, aL))
                    # underground segments and transitions, both lanes, per tier
                    axis = direction.axis
                    for lane, q, mx in ((0, lambda t: t.colour, b.mL), (1, lambda t: t.colour_r, b.mR)):
                        for t in range(self.tier_count):
                            qu = lambda tile: self.lane_u(tile, t, axis, lane)
                            self.clauses += implies([self.ug(a, t)[direction]], set_numbers_equal(qu(a), qu(b)))
                            self.clauses += implies([a.input_direction[direction], -a.is_splitter, *invert_components(a.output_direction),
                                                     *self.tier_is(a, t)],
                                                    set_numbers_equal(q(a), qu(b)))
                            self.clauses += implies([b.output_direction[direction], -b.is_splitter, -b.lone, *invert_components(b.input_direction),
                                                     *self.tier_is(b, t), *([-mx] if ugm else [])],
                                                    set_numbers_equal(qu(a), q(b)))
        for tile in self.iterate_tiles():
            for d in Direction:
                # the unfed side of a side-fed entrance or lone exit is EMPTY (left side: a feeder heading d.prev)
                self.clauses += implies([tile.is_underground_in, tile.input_direction[d], tile.no_behind, -tile.side_in[d.prev]],
                                        set_number(EMPTY, tile.colour))
                self.clauses += implies([tile.is_underground_in, tile.input_direction[d], tile.no_behind, -tile.side_in[d.next]],
                                        set_number(EMPTY, tile.colour_r))
                self.clauses += implies([tile.lone, tile.output_direction[d], -tile.side_in[d.prev]], set_number(EMPTY, tile.colour))
                self.clauses += implies([tile.lone, tile.output_direction[d], -tile.side_in[d.next]], set_number(EMPTY, tile.colour_r))
            # a belt or underground end carries at least one live lane
            self.clauses.append([tile.is_empty, tile.is_splitter, *tile.colour[1:], *tile.colour_r[1:]])
        self.merge_colours()

    def merge_colours(self):
        """Belt sideload merges (L4/L5): a belt facing d sideloaded from its left (a feeder heading d.prev) has its
        left lane = merge y of (x0: the behind tile's left lane, x1: the feeder's back lane = its right lane, x2: its front
        lane = its left lane); from the right (heading d.next): right lane, x1 = the feeder's left lane, x2 its right. A
        missing input (None) carries nothing: EMPTY, or no behind feeder at all."""
        def outgoing(t):
            return (([-t.is_splitter], (t.colour, t.colour_r)), ([t.is_splitter], (t.colour_o, t.colour_or)))

        def setc(c, q):
            return set_number(EMPTY if c is None else c, q)
        for x in range(self.width):
            for y in range(self.height):
                b = self.get_tile_instance(x, y)
                for j, (x0, x1, x2, yy) in enumerate(self.merges):
                    for d in Direction:
                        for ml, fd, own, back_i, lane in ((b.merge_l, d.prev, b.colour, 1, 0), (b.merge_r, d.next, b.colour_r, 0, 1)):
                            if self.ug_merges:
                                self.ug_merge_colours(x, y, b, j, d, ml, fd, own, back_i, lane, (x0, x1, x2, yy))
                            pre = [ml[j], b.output_direction[d]] + ([b.is_belt] if self.ug_merges else [])
                            a = self.get_tile_instance_offset(x, y, -fd.dx, -fd.dy)      # the side feeder
                            c = self.get_tile_instance_offset(x, y, -d.dx, -d.dy)        # behind
                            if a is None or (c is None and x0 is not None):
                                self.clauses.append(invert_components(pre))
                                continue
                            self.clauses += implies(pre, set_number(yy, own))
                            for kind, lanes in outgoing(a):
                                self.clauses += implies(pre + kind, setc(x1, lanes[back_i]) + setc(x2, lanes[1 - back_i]))
                            if c is not None:
                                if x0 is not None:
                                    self.clauses += implies(pre, [[c.output_direction[d]]])
                                for kind, lanes in outgoing(c):
                                    self.clauses += implies(pre + kind + [c.output_direction[d]], setc(x0, lanes[lane]))

    def ug_merge_colours(self, x, y, b, j, d, ml, fd, own, back_i, lane, mk):
        """Merge j at an underground end b facing d, side-fed by a feeder heading fd (L8 / L9):
        entrance: own lane = merge(x0 = the behind tile's lane, x1 = the feeder's back lane); its front lane blocked.
        exit:     own lane = merge(x0 = the tunnel's lane, x1 = the feeder's front lane); its back lane blocked.
        Only two ranks enter an underground end, so x2 must be absent."""
        x0, x1, x2, yy = mk
        a = self.get_tile_instance_offset(x, y, -fd.dx, -fd.dy)      # the side feeder
        c = self.get_tile_instance_offset(x, y, -d.dx, -d.dy)        # behind
        for end in ('in', 'out'):
            pre = [ml[j], *self.facing(b, d)[end]]
            if a is None or c is None or x2 is not None or x0 is None:
                self.clauses.append(invert_components(pre))
                continue
            self.clauses += implies(pre, set_number(yy, own))
            enter_i = back_i if end == 'in' else 1 - back_i
            for kind, lanes in (([-a.is_splitter], (a.colour, a.colour_r)), ([a.is_splitter], (a.colour_o, a.colour_or))):
                self.clauses += implies(pre + kind, set_number(x1, lanes[enter_i]) + set_maximum(DEAD, lanes[1 - enter_i]))
            if end == 'in':
                # fed from behind (prevent_intersection: no no_behind -> c outputs into b)
                self.clauses += implies(pre, [[c.output_direction[d]]])
                for kind, lanes in (([-c.is_splitter], (c.colour, c.colour_r)), ([c.is_splitter], (c.colour_o, c.colour_or))):
                    self.clauses += implies(pre + kind + [c.output_direction[d]], set_number(x0, lanes[lane]))
            else:
                # the tunnel: a tier-t segment at c, or (gap 0) a tier-t entrance at c facing d
                q = c.colour if lane == 0 else c.colour_r
                for t in range(self.tier_count):
                    tier_b = self.tier_is(b, t)
                    self.clauses += implies(pre + tier_b + [self.ug(c, t)[d]],
                                            set_number(x0, self.lane_u(c, t, d.axis, lane)))
                    self.clauses += implies(pre + tier_b + [-self.ug(c, t)[d], c.is_underground_in, c.input_direction[d]],
                                            set_number(x0, q))


def create_lane_balancer(nodes, width, height, underground_length=4, merges=(), tiers=None, settings=None,
                         ug_merges=False, filtered=None):
    """settings: {node: {'in': X, 'out': Y, 'filter': item}} (open_lane_network with_settings, belt_to_lane);
    priorities need no clauses (set after solving, Grid.solution_hook); filtered nodes (default: those marked filter)
    may have anything in front of their outputless half. tiers / ug_merges: see LaneGrid. With none of these the CNF
    is the one of branch `lanes`."""
    colours_used = {c for node in nodes for lane in node for port in lane for c in port if c is not None}
    colours_used |= {EMPTY, DEAD, INPUT, OUTPUT}
    n, m = lane_counts(nodes, merges)
    counts = collections.Counter(nodes)
    keys = list(counts)
    merge_counts = collections.Counter(merges)
    mkeys = list(merge_counts)
    colours_used |= {c for mk in mkeys for c in mk if c is not None}
    grid = LaneGrid(width, height, max(colours_used) + 1, underground_length, {'node': OneHotTemplate(len(keys))}, mkeys,
                    tiers=tiers, ug_merges=ug_merges)
    settings = {k: v for k, v in (settings or {}).items() if v}
    if filtered is None:
        filtered = {k for k, v in settings.items() if 'filter' in v}
    if filtered:
        allow_lane_filtered_fronts(grid, keys, filtered)
    if settings or filtered:
        grid.solution_hook = lambda solution: apply_lane_settings(solution, keys, settings, filtered)
    grid.lane_keys = keys
    for c in range(max(colours_used) + 1):
        if c not in colours_used:
            grid.prevent_colour(c)
    grid.block_underground_through_edges()
    grid.prevent_bad_undergrounding()
    grid.prevent_bad_colouring()

    checkerboard = [grid.get_tile_instance(x, y).is_splitter for x in range(width) for y in range(height) if (x + y) % 2]
    grid.clauses += library_equals(checkerboard, sum(counts.values()), grid.pool)
    for i, key in enumerate(keys):
        lits = [grid.get_tile_instance(x, y).node[i] for x in range(width) for y in range(height)]
        grid.clauses += library_equals(lits, counts[key], grid.pool, EncType.kmtotalizer)
    for tile in grid.iterate_tiles():
        grid.clauses += quadratic_one([-tile.is_splitter_head, *tile.node])
    for j, mk in enumerate(mkeys):
        lits = [lit for t in grid.iterate_tiles() for lit in (t.merge_l[j], t.merge_r[j])]
        grid.clauses += library_equals(lits, merge_counts[mk], grid.pool, EncType.kmtotalizer)
    grid.clauses += library_equals([t.is_input for t in grid.iterate_tiles()], n, grid.pool)
    grid.clauses += library_equals([t.is_output for t in grid.iterate_tiles()], m, grid.pool)
    for tile in grid.iterate_tiles():
        grid.clauses += implies([tile.is_input], set_number(INPUT, tile.colour) + set_number(INPUT, tile.colour_r))
        grid.clauses += implies([tile.is_output], set_number(OUTPUT, tile.colour) + set_number(OUTPUT, tile.colour_r))

    def pair(lanes, j, filler):
        """Port j's lane pair (L, R) of a node, or None if both lanes are absent; an absent lane next to a used one
        becomes `filler` (EMPTY on the input side, DEAD on the output side)."""
        cl, cr = lanes[0][j], lanes[1][j]
        if cl is None and cr is None:
            return None
        return (filler if cl is None else cl, filler if cr is None else cr)

    for i, ((Lin, Lout), (Rin, Rout)) in enumerate(keys):
        ins = [pair((Lin, Rin), j, EMPTY) for j in (0, 1)]      # j = 0: the left half, 1: the right half
        outs = [pair((Lout, Rout), j, DEAD) for j in (0, 1)]
        assert any(ins) and any(outs), keys[i]
        use_out = not all(ins)
        # a pass-through splitter (one input half, one output half: Raynquist's 8-6) has no side that is always set
        # on its head: its direction is the head's input or output direction (one literal per direction)
        one_one = use_out and not all(outs)
        for x in range(width):
            for y in range(height):
                t00 = grid.get_tile_instance(x, y)               # the head = the RIGHT half
                if one_one:
                    facing = [_either(grid, t00.input_direction[d], t00.output_direction[d]) for d in Direction]
                else:
                    facing = t00.output_direction if use_out else t00.input_direction
                grid.clauses.append([-t00.node[i], *facing])
                for direction in Direction:
                    dx0, dy0 = direction.vec
                    dx1, dy1 = direction.next.vec                # the LEFT half
                    t10 = grid.get_tile_instance_offset(x, y, dx0, dy0)
                    t01 = grid.get_tile_instance_offset(x, y, dx1, dy1)
                    t11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1)
                    pre = [t00.node[i], facing[direction]]
                    if any(t is None for t in (t00, t10, t01, t11)):
                        grid.clauses.append(invert_components(pre))
                        continue

                    def setp(p, t, out=False):
                        if out:
                            return set_number(p[0], t.colour_o) + set_number(p[1], t.colour_or)
                        return set_number(p[0], t.colour) + set_number(p[1], t.colour_r)

                    # halves: right = t00 (port 1 of the node), left = t01 (port 0). A fair splitter may swap its two
                    # inputs (both lanes together: one belt carries both) and, independently, its two outputs.
                    for side, side_ports in (('in', ins), ('out', outs)):
                        tiles, o = (t00, t01), side == 'out'
                        dlit = [h.input_direction[direction] if side == 'in' else h.output_direction[direction]
                                for h in (t00, t01)]
                        p_left, p_right = side_ports
                        if p_left is None or p_right is None:
                            live = p_left if p_right is None else p_right
                            grid.clauses += implies(pre, literals_different(dlit[0], dlit[1]))
                            grid.clauses += implies(pre + [dlit[0]], setp(live, tiles[0], o))
                            grid.clauses += implies(pre + [dlit[1]], setp(live, tiles[1], o))
                        else:
                            grid.clauses += implies(pre, [[dlit[0]], [dlit[1]]])
                            if p_left == p_right:
                                grid.clauses += implies(pre, setp(p_left, tiles[0], o) + setp(p_left, tiles[1], o))
                            else:
                                sel = swap_literal(grid, t00, i, side)
                                grid.clauses += implies(pre + [-sel], setp(p_right, tiles[0], o) + setp(p_left, tiles[1], o))
                                grid.clauses += implies(pre + [sel], setp(p_left, tiles[0], o) + setp(p_right, tiles[1], o))
    return grid


def _outless_half(node):
    (Lin, Lout), (Rin, Rout) = node
    halves = [j for j in (0, 1) if Lout[j] is None and Rout[j] is None]
    return halves[0] if len(halves) == 1 else None


def allow_lane_filtered_fronts(grid, keys, filtered):
    """belt_balancer.allow_filtered_fronts for lane networks: per tile, r -> the tile is the outputless half of a
    splitter whose node is filtered; r lifts the in-front rules for that tile (Grid.front_relax). Front_relax must be
    set before prevent_intersection / forbid_dangling_fronts."""
    indices = []
    for key in filtered:
        if _outless_half(key) is None:
            raise ValueError(f'filtered splitter {key} needs exactly one output half with no lanes')
        indices.append(keys.index(key))
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


def apply_lane_settings(solution, keys, settings, filtered=()):
    """Write priorities / filters into a solved lane grid, in place: input priority on the half whose input lanes
    (colour / colour_r) hold X, output priority on the half whose output lanes (colour_o / colour_or) hold Y, a
    filter (with output priority, as the game needs) on the outputless half of a filtered splitter."""
    from .network import DEFAULT_FILTER
    from .tile import Splitter
    height, width = solution.shape
    count = 0
    for y in range(height):
        for x in range(width):
            cell = solution[y, x]
            if not cell['is_splitter'] or not cell['is_splitter_head']:
                continue
            key = keys[cell['node']]
            entry = dict(settings.get(key) or {})
            if key in filtered and 'filter' not in entry:
                entry['filter'] = DEFAULT_FILTER
            if not entry:
                continue
            d = Splitter.read(cell['tile']).direction
            halves = {'right': (x, y), 'left': (x + d.next.dx, y + d.next.dy)}
            cells = {side: solution[hy, hx] for side, (hx, hy) in halves.items()}
            values = {}
            if 'in' in entry:
                values['input_priority'] = next(s for s, c in cells.items() if entry['in'] in (c['colour'], c['colour_r']))
            if 'out' in entry:
                values['output_priority'] = next(s for s, c in cells.items() if entry['out'] in (c['colour_o'], c['colour_or']))
            if 'filter' in entry:
                outless = [s for s, c in cells.items() if c['output_direction'] is None]
                assert len(outless) == 1, (x, y, entry)
                values['output_priority'] = outless[0]
                values['filter'] = entry['filter']
            for hx, hy in halves.values():
                solution[hy, hx] = {**solution[hy, hx], 'tile': {**solution[hy, hx]['tile'], **values}}
            count += 1
    return count


def _either(grid, a, b):
    """A literal equal to a or b (cached per pair)."""
    store = grid.__dict__.setdefault('_either', {})
    if (a, b) not in store:
        h = grid.allocate_variable()
        grid.clauses += [[-h, a, b], [-a, h], [-b, h]]
        store[(a, b)] = h
    return store[(a, b)]


def swap_literal(grid, head, i, side):
    """One literal per (head tile, node, side): the splitter placed there takes its two inputs ('in') or gives its two
    outputs ('out') the other way round."""
    store = grid.__dict__.setdefault('_lane_swaps', {})
    key = (head.is_splitter_head, i, side)
    if key not in store:
        store[key] = grid.allocate_variable()
    return store[key]


def main():
    ap = argparse.ArgumentParser(description='Lane-level belt balancer layout from a lane network')
    ap.add_argument('network')
    ap.add_argument('width', type=int)
    ap.add_argument('height', type=int)
    ap.add_argument('--underground-length', type=int, default=4)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--solver', default='Glucose3')
    ap.add_argument('--use-ends', action='store_true', help='other tiles of the end columns may be used (belts, '
                    'splitters or nothing), as long as nothing else enters or leaves through the edge')
    ap.add_argument('--blueprint', action='store_true', help='print blueprint strings instead of tile JSON')
    ap.add_argument('--level', default='express')
    ap.add_argument('--90', dest='turn_90', action='store_true', help='corner: outputs leave through the bottom edge')
    ap.add_argument('--in-offset', type=int, help='first input at this row of the input edge')
    ap.add_argument('--out-offset', type=int, help='first output at this row of the output edge')
    ap.add_argument('--belt', action='store_true', help='NETWORK is a belt-level Factorio-SAT network (4 colours a '
                    'line, with in=/out=/filter annotations): every edge becomes a lane pair')
    ap.add_argument('--tiers', help='mixed-tier undergrounds, e.g. yellow,red (overrides --underground-length)')
    ap.add_argument('--tier-max', action='append', default=[], metavar='TIER=K')
    ap.add_argument('--filter-dangling', action='store_true', help='every splitter with an outputless half may have '
                    'anything in front of it; it gets a filter on export')
    ap.add_argument('--contact', action='store_true', help="dangling_front_mode 'contact' instead of 'empty'")
    ap.add_argument('--ug-merges', action='store_true', help='merges may sit at underground ends (auto when a merge '
                    'line ends in `ug`)')
    a = ap.parse_args()
    if a.belt:
        from .network import open_network, network_settings
        net, sets = open_network(a.network, with_settings=True)
        nodes, settings = belt_to_lane(net, network_settings(net, sets))
        merges, ugm = [], False
    else:
        nodes, merges, settings, ugm = open_lane_network(a.network, with_settings=True)
    filtered = None
    if a.filter_dangling:
        filtered = {k for k in nodes if _outless_half(k) is not None} | {k for k, v in settings.items() if 'filter' in v}
    n, m = lane_counts(nodes, merges)
    tiers = a.tiers.split(',') if a.tiers else None
    grid = create_lane_balancer(nodes, a.width, a.height, a.underground_length, merges, tiers=tiers, settings=settings,
                                ug_merges=ugm or a.ug_merges, filtered=filtered)
    for cap in a.tier_max:
        name, k = cap.split('=')
        grid.limit_tier(name, int(k))
    if a.contact:
        grid.dangling_front_mode = 'contact'
    grid.prevent_intersection()
    grid.enforce_maximum_underground_length()
    start = [grid.get_tile_instance(0, y) for y in range(grid.height)]
    if a.turn_90:
        end = [grid.get_tile_instance(x, grid.height - 1) for x in range(grid.width)]
    else:
        end = [grid.get_tile_instance(grid.width - 1, y) for y in range(grid.height)]
    ino = setup_balancer_input(grid, start, 0, n, not a.use_ends, rest_belts=False)
    outo = setup_balancer_output(grid, end, 3 if a.turn_90 else 0, m, not a.use_ends)
    if a.in_offset is not None and ino:
        grid.clauses.append([ino[a.in_offset]])
    if a.out_offset is not None and outo:
        grid.clauses.append([outo[a.out_offset]])
    level = a.level
    if tiers:
        level = min((parse_tier(t) for t in tiers), key=UNDERGROUND_TIERS.get)    # belts and splitters: the lowest tier
    found = 0
    for solution in grid.itersolve(solver=a.solver, ignore_colour=True):
        found += 1
        if a.blueprint:
            import numpy as np
            tiles = np.vectorize(blueprint.read_tile)(np.array(solution.tolist()))
            print(blueprint.encode_blueprint(blueprint.make_blueprint(
                tiles, f'lane {n}-{m}', blueprint.TransportBeltLevel[level.upper()])), flush=True)
        else:
            print(json.dumps(solution.tolist()), flush=True)
        if not a.all:
            break
    if not found:
        print('UNSAT', file=sys.stderr)


if __name__ == '__main__':
    main()
