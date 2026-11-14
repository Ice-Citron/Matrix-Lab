"""MuJoCo bin sorting with round bowls: the bin-sorting cell with the three 3D-printed bowls of the real
SO-101 cell in place of the square bins.

Same UR5e, suction cup, cube size and placement box as sim.mj_sort_sim. The cube is cream, grey or black
(the seed picks the colour) and the three bowls are cream, grey and black, standing where the bins stood.
Each bowl is a round dish with a flat floor, BOWL_INNER_R inner radius (120 mm across) and BOWL_HEIGHT
(40 mm) walls. Success: the cube rests inside the bowl of its own colour with the suction off.
The wall is WALL_SEGMENTS overlapping boxes around the ring, so it is closed to contact and looks smooth.
"""
from __future__ import annotations

import math

from sim.mj_sort_sim import NOMINAL_BINS, SortSim, bin_offset, build_model, colour_of, mujoco

COLOURS = ("cream", "grey", "black")
RGBA = {"cream": "0.93 0.89 0.78 1", "grey": "0.55 0.55 0.57 1", "black": "0.12 0.12 0.12 1"}
NOMINAL_BOWLS = dict(zip(COLOURS, NOMINAL_BINS.values()))  # the three bin spots: (0.25, 0.30), (0.45, 0.30), (0.65, 0.30)
BOWL_INNER_R = 0.060   # 120 mm inner diameter, measured on the printed bowls
BOWL_WALL = 0.004
BOWL_HEIGHT = 0.040    # 40 mm walls, measured on the printed bowls
FLOOR = 0.003
WALL_SEGMENTS = 64
_OLD = dict(zip(COLOURS, ("red", "green", "blue")))  # bin_offset keys the phase-2 bin offsets by the bin colours


def _ring_mesh() -> str:
    """A closed ring (inner wall, outer wall, rim) as an inline MuJoCo mesh: the smooth look of the bowl."""
    r0, r1 = BOWL_INNER_R, BOWL_INNER_R + BOWL_WALL
    z0, z1 = FLOOR, BOWL_HEIGHT
    n = WALL_SEGMENTS
    verts, faces = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        c, s_ = math.cos(a), math.sin(a)
        verts += [(r0 * c, r0 * s_, z0), (r0 * c, r0 * s_, z1), (r1 * c, r1 * s_, z0), (r1 * c, r1 * s_, z1)]
    for i in range(n):
        j = (i + 1) % n
        i0, i1, i2, i3 = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        j0, j1, j2, j3 = 4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3
        faces += [(i0, j1, i1), (i0, j0, j1),      # inner wall, facing inwards
                  (i2, i3, j3), (i2, j3, j2),      # outer wall, facing outwards
                  (i1, j1, j3), (i1, j3, i3)]      # rim, facing up
    v = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts)
    f = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    return f'<asset><mesh name="bowl_ring" vertex="{v}" face="{f}"/></asset>'


def _bowl_xml(colour, xy):
    """Floor disc plus the visual ring mesh; the ring's contact is WALL_SEGMENTS boxes in group 3 (not drawn)."""
    r, w = BOWL_INNER_R, BOWL_WALL / 2
    h = (BOWL_HEIGHT - FLOOR) / 2
    chord = (r + w) * math.pi / WALL_SEGMENTS * 1.3  # half-length; 30 % overlap closes the ring to contact
    walls = []
    for i in range(WALL_SEGMENTS):
        a = 2 * math.pi * i / WALL_SEGMENTS
        walls.append(f'<geom type="box" pos="{(r + w) * math.cos(a):.5f} {(r + w) * math.sin(a):.5f} {FLOOR + h:.4f}" '
                     f'size="{w} {chord:.5f} {h:.4f}" euler="0 0 {math.degrees(a):.3f}" group="3" rgba="{RGBA[colour]}"/>')
    return f'''<body name="bowl_{colour}" pos="{xy[0]} {xy[1]} 0">
      <geom type="cylinder" pos="0 0 {FLOOR / 2}" size="{r + 2 * w:.4f} {FLOOR / 2}" rgba="{RGBA[colour]}"/>
      <geom type="mesh" mesh="bowl_ring" contype="0" conaffinity="0" rgba="{RGBA[colour]}"/>
      {chr(10).join(walls)}
    </body>'''


class BowlSim(SortSim):
    def colour_of(self, params: dict) -> str:
        return colour_of(params, COLOURS)

    def build(self, params: dict) -> mujoco.MjModel:
        bowls = "\n".join(_bowl_xml(c, NOMINAL_BOWLS[c] + bin_offset(params, _OLD[c])) for c in COLOURS)
        return build_model(params, fixtures=bowls, title="gaphunter bin sorting, round bowls", colours=COLOURS,
                           rgba=RGBA, assets=_ring_mesh())

    def bin_true(self, colour):
        return NOMINAL_BOWLS[colour] + bin_offset(self.params, _OLD[colour])

    def bin_of_cube(self):
        """Colour of the bowl the cube rests in, or None (round test instead of the bins' square one)."""
        x, y, z = self.cube_true()
        for c in COLOURS:
            bx, by = self.bin_true(c)
            if math.hypot(x - bx, y - by) < BOWL_INNER_R and z < BOWL_HEIGHT:
                return c
        return None

    def bowl_pose(self, colour):
        est_xy = self._estimate(self.bin_true(colour), NOMINAL_BOWLS[colour])
        result = ((float(est_xy[0]), float(est_xy[1]), BOWL_HEIGHT), 0.0)
        self._count("bowl_pose", {"colour": colour}, [list(result[0]), result[1]])
        return result
