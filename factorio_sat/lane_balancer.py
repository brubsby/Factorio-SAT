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
Not encoded yet: sideloads onto belts and onto exits (they need priority-merge nodes in the colour model), lone exits
used as lane filters (Raynquist's 1-1 TU and 3-3 lane balancers), priority splitters, mixed tiers.

Reserved colours: 0 = EMPTY (a lane that never carries items: the unfed side of a side-fed entrance, or the absent lane
of a splitter half whose other lane is used), 1 = DEAD (a splitter lane output that the network leaves dangling while
the half's other lane is used; it may only end at the blocked lane of a side-feed), 2 = network input, 3 = network
output, 4.. = internal lane edges.

Lane network file: one physical splitter per line, `aL bL cL dL aR bR cR dR`: the left-lane node's two inputs and two
outputs, then the right-lane node's, -1 = no port. Within a node the first input/output is the splitter's LEFT half
(relative to its facing), the second its right half; the solver may mirror a splitter (swap halves) but always for both
lanes together.

    python -m factorio_sat.lane_balancer NETWORK WIDTH HEIGHT [--underground-length 4] [--all] [--blueprint]
"""
import argparse
import collections
import json
import sys

from pysat.card import EncType

from . import blueprint
from .belt_balancer import setup_balancer_input, setup_balancer_output
from .cardinality import library_equals, quadratic_one
from .direction import Direction
from .solver import Grid
from .template import ArrayTemplate, BoolTemplate, NumberTemplate, OneHotTemplate
from .cardinality import quadratic_amo
from .util import implies, invert_components, literals_different, set_maximum, set_number, set_numbers_equal

EMPTY, DEAD, INPUT, OUTPUT = 0, 1, 2, 3


def open_lane_network(path, with_merges=False):
    """-> list of splitter nodes ((L_in, L_out), (R_in, R_out)), each a pair of 2-tuples with None for -1; with
    with_merges also the belt-sideload merges `M x0 x1 x2 y` (x0 = the straight lane, x1 = the side belt's back lane,
    x2 = its front lane, y = the merged lane; -1 = none)."""
    nodes, merges = [], []
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('M'):
            merges.append(tuple(None if int(t) == -1 else int(t) for t in line.split()[1:5]))
            continue
        c = [None if int(t) == -1 else int(t) for t in line.split()[:8]]
        nodes.append(((tuple(c[0:2]), tuple(c[2:4])), (tuple(c[4:6]), tuple(c[6:8]))))
    return (nodes, merges) if with_merges else nodes


def lane_counts(nodes, merges=()):
    n = sum(c == INPUT for node in nodes for lane in node for c in lane[0]) + sum(c == INPUT for mk in merges for c in mk[:3])
    m = sum(c == OUTPUT for node in nodes for lane in node for c in lane[1]) + sum(mk[3] == OUTPUT for mk in merges)
    assert n % 2 == 0 and m % 2 == 0, (n, m)
    return n // 2, m // 2


class LaneGrid(Grid):
    def __init__(self, width, height, colours, underground_length=4, extras=None, merges=()):
        """merges: the distinct belt-sideload merge nodes (x0, x1, x2, y) of the network (see open_lane_network)."""
        bits = (colours - 1).bit_length()
        self.merges = list(merges)
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
        super().__init__(width, height, colours, underground_length, ex)
        self.side_feed_rules()

    def colour_ranges(self, tile):
        return [tile.colour, tile.colour_ux, tile.colour_uy, tile.colour_r, tile.colour_r_ux, tile.colour_r_uy,
                tile.colour_o, tile.colour_or]

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
                    self.clauses += implies([s], [[b.is_underground_in, b.is_belt, b.lone], [a.output_direction[dd]],
                                                  [-b.input_direction[dd]], [-b.input_direction[dd.reverse]],
                                                  [-b.output_direction[dd]], [-b.output_direction[dd.reverse]]])
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
                    for j in range(K):
                        self.clauses += [[-ml[j], m_any], [-ml[j], b.is_belt]]
                    for d in Direction:
                        sd = b.side_in[getattr(d, side)]
                        self.clauses.append([-b.is_belt, -b.output_direction[d], -sd, m_any])
                        self.clauses.append([-m_any, -b.output_direction[d], sd])

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
                self.clauses += implies([tile_a.input_direction[direction], tile_a.is_splitter, -tile_b.is_splitter, *relaxed], [
                    [-tile_b.input_direction[direction.next], -tile_b.output_direction[direction.next]],
                    [-tile_b.input_direction[direction.prev], -tile_b.output_direction[direction.prev]],
                    [-tile_b.input_direction[direction.next], -tile_b.output_direction[direction]],
                    [-tile_b.input_direction[direction.prev], -tile_b.output_direction[direction]],
                    [-tile_b.input_direction[direction.reverse], -tile_b.output_direction[direction.next]],
                    [-tile_b.input_direction[direction.reverse], -tile_b.output_direction[direction.prev]],
                ])
                # an entrance (or a lone exit) crosswise in front of a splitter half is side-fed by it (L11): that half
                # must output into it; other exits crosswise in front of a splitter stay forbidden
                for splitter_side in (tile_a.input_direction[direction], tile_a.output_direction[direction]):
                    self.clauses += implies([splitter_side, tile_a.is_splitter, -tile_b.is_splitter, *relaxed], [
                        [-tile_b.is_underground_in, -tile_b.input_direction[direction.next], out_a],
                        [-tile_b.is_underground_in, -tile_b.input_direction[direction.prev], out_a],
                        # a crosswise exit: only a lone exit, side-fed by this half
                        [-tile_b.is_underground_out, -tile_b.output_direction[direction.next], tile_b.lone],
                        [-tile_b.is_underground_out, -tile_b.output_direction[direction.prev], tile_b.lone],
                        [-tile_b.is_underground_out, -tile_b.output_direction[direction.next], out_a],
                        [-tile_b.is_underground_out, -tile_b.output_direction[direction.prev], out_a],
                    ])

    def prevent_bad_undergrounding(self):
        """Grid.prevent_bad_undergrounding, single tier, but an entrance may be followed directly by its exit (gap 0).
        The belt-level encoder never needs that (a belt does the same), so it requires a segment tile in between; the
        lane trick does: a side-fed entrance with its exit right behind it is a one-tile lane filter (Raynquist's 2-2
        and 4-4 lane balancers)."""
        assert not self.multi_tier, 'lanes: single tier only'
        for direction in Direction:
            reverse_dir = direction.reverse
            dx, dy = direction.vec
            for x in range(self.width):
                for y in range(self.height):
                    a = self.get_tile_instance(x, y)
                    nxt = self.get_tile_instance_offset(x, y, +dx, +dy)
                    prv = self.get_tile_instance_offset(x, y, -dx, -dy)
                    ug_a = self.ug(a, 0)
                    self.clauses += implies([a.is_underground_in, a.input_direction[direction]],
                                            [[-ug_a[direction]], [-ug_a[reverse_dir]]])
                    self.clauses += implies([a.is_underground_out, a.output_direction[direction]],
                                            [[-ug_a[direction]], [-ug_a[reverse_dir]]])
                    if nxt is not None:
                        adj = self.allocate_variable()       # a = entrance, nxt = its exit, no gap
                        self.clauses += [[-adj, nxt.is_underground_out], [-adj, nxt.output_direction[direction]]]
                        self.clauses += implies([a.is_underground_in, a.input_direction[direction]],
                                                [[self.ug(nxt, 0)[direction], adj]])
                    else:
                        self.clauses += [[-a.is_underground_in, -a.input_direction[direction]]]
                    if prv is not None:
                        adj = self.allocate_variable()
                        self.clauses += [[-adj, prv.is_underground_in], [-adj, prv.input_direction[direction]]]
                        self.clauses += implies([a.is_underground_out, a.output_direction[direction]],
                                                [[self.ug(prv, 0)[direction], adj, a.lone]])
                        # a lone exit has no tunnel: no segment and no entrance right behind it
                        self.clauses += implies([a.lone, a.output_direction[direction]],
                                                [[-self.ug(prv, 0)[direction]],
                                                 [-prv.is_underground_in, -prv.input_direction[direction]]])
                    else:
                        self.clauses += [[-a.is_underground_out, -a.output_direction[direction], a.lone]]
                    if nxt is not None:
                        self.clauses += implies([ug_a[direction], -self.ug(nxt, 0)[direction]],
                                                [[nxt.is_underground_out], [nxt.output_direction[direction]]])
                        # gap 0: the exit takes the entrance's lanes directly
                        for qa, qb in ((a.colour, nxt.colour), (a.colour_r, nxt.colour_r)):
                            self.clauses += implies([a.is_underground_in, a.input_direction[direction],
                                                     nxt.is_underground_out, nxt.output_direction[direction]],
                                                    set_numbers_equal(qa, qb))
                    if prv is not None:
                        self.clauses += implies([ug_a[direction], -self.ug(prv, 0)[direction]],
                                                [[prv.is_underground_in], [prv.input_direction[direction]]])

    # ------------------------------------------------------------------ lane colours
    def prevent_bad_colouring(self):
        assert not self.multi_tier, 'lanes: single tier only'
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
                        # the feeder's back lane enters, its front lane is blocked
                        self.clauses += implies([out_a, *kind, s, b.is_underground_in, b.input_direction[direction.next]],
                                                set_numbers_equal(b.colour, aR) + set_maximum(DEAD, aL))
                        self.clauses += implies([out_a, *kind, s, b.is_underground_in, b.input_direction[direction.prev]],
                                                set_numbers_equal(b.colour_r, aL) + set_maximum(DEAD, aR))
                        # crosswise into a lone exit: the feeder's FRONT lane enters, its back lane is blocked
                        self.clauses += implies([out_a, *kind, s, b.is_underground_out, b.output_direction[direction.next]],
                                                set_numbers_equal(b.colour, aL) + set_maximum(DEAD, aR))
                        self.clauses += implies([out_a, *kind, s, b.is_underground_out, b.output_direction[direction.prev]],
                                                set_numbers_equal(b.colour_r, aR) + set_maximum(DEAD, aL))
                    # underground segments and transitions, both lanes
                    ax = direction.axis.value == 0
                    for q, qux, quy in ((lambda t: t.colour, lambda t: t.colour_ux, lambda t: t.colour_uy),
                                        (lambda t: t.colour_r, lambda t: t.colour_r_ux, lambda t: t.colour_r_uy)):
                        qu = qux if ax else quy
                        self.clauses += implies([self.ug(a, 0)[direction]], set_numbers_equal(qu(a), qu(b)))
                        self.clauses += implies([a.input_direction[direction], -a.is_splitter, *invert_components(a.output_direction)],
                                                set_numbers_equal(q(a), qu(b)))
                        self.clauses += implies([b.output_direction[direction], -b.is_splitter, -b.lone, *invert_components(b.input_direction)],
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
                            pre = [ml[j], b.output_direction[d]]
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


def create_lane_balancer(nodes, width, height, underground_length=4, merges=()):
    colours_used = {c for node in nodes for lane in node for port in lane for c in port if c is not None}
    colours_used |= {EMPTY, DEAD, INPUT, OUTPUT}
    n, m = lane_counts(nodes, merges)
    counts = collections.Counter(nodes)
    keys = list(counts)
    merge_counts = collections.Counter(merges)
    mkeys = list(merge_counts)
    colours_used |= {c for mk in mkeys for c in mk if c is not None}
    grid = LaneGrid(width, height, max(colours_used) + 1, underground_length, {'node': OneHotTemplate(len(keys))}, mkeys)
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
        assert not (use_out and not all(outs)), f'splitter {keys[i]} has a single input and a single output'
        for x in range(width):
            for y in range(height):
                t00 = grid.get_tile_instance(x, y)               # the head = the RIGHT half
                grid.clauses.append([-t00.node[i], *(t00.output_direction if use_out else t00.input_direction)])
                for direction in Direction:
                    dx0, dy0 = direction.vec
                    dx1, dy1 = direction.next.vec                # the LEFT half
                    t10 = grid.get_tile_instance_offset(x, y, dx0, dy0)
                    t01 = grid.get_tile_instance_offset(x, y, dx1, dy1)
                    t11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1)
                    pre = [t00.node[i], (t00.output_direction if use_out else t00.input_direction)[direction]]
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
    ap.add_argument('--in-offset', type=int, help='first input at this row of the input edge')
    ap.add_argument('--out-offset', type=int, help='first output at this row of the output edge')
    a = ap.parse_args()
    nodes, merges = open_lane_network(a.network, with_merges=True)
    n, m = lane_counts(nodes, merges)
    grid = create_lane_balancer(nodes, a.width, a.height, a.underground_length, merges)
    grid.prevent_intersection()
    grid.enforce_maximum_underground_length()
    start = [grid.get_tile_instance(0, y) for y in range(grid.height)]
    end = [grid.get_tile_instance(grid.width - 1, y) for y in range(grid.height)]
    ino = setup_balancer_input(grid, start, 0, n, not a.use_ends, rest_belts=False)
    outo = setup_balancer_output(grid, end, 0, m, not a.use_ends)
    if a.in_offset is not None and ino:
        grid.clauses.append([ino[a.in_offset]])
    if a.out_offset is not None and outo:
        grid.clauses.append([outo[a.out_offset]])
    found = 0
    for solution in grid.itersolve(solver=a.solver, ignore_colour=True):
        found += 1
        if a.blueprint:
            import numpy as np
            tiles = np.vectorize(blueprint.read_tile)(np.array(solution.tolist()))
            print(blueprint.encode_blueprint(blueprint.make_blueprint(
                tiles, f'lane {n}-{m}', blueprint.TransportBeltLevel[a.level.upper()])), flush=True)
        else:
            print(json.dumps(solution.tolist()), flush=True)
        if not a.all:
            break
    if not found:
        print('UNSAT', file=sys.stderr)


if __name__ == '__main__':
    main()
