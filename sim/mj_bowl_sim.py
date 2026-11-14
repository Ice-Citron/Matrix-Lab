"""MuJoCo cube-in-bowl: the bin-sorting cell with three round bowls instead of square bins.

Same UR5e, suction cup, cube colours and placement box as sim.mj_sort_sim. The three bowls are the
3D-printed ones of the real SO-101 cell: cream, grey and black, each a round dish of BOWL_INNER_R
inner radius (120 mm across) with 40 mm walls, standing where the bins stood. The cube's colour decides the
bowl: red -> cream, green -> grey, blue -> black. Success: the cube rests inside its bowl, suction off.
"""
from __future__ import annotations

import math

import numpy as np

from sim.mj_sort_sim import COLOURS, NOMINAL_BINS, SortSim, bin_offset, build_model, mujoco

BOWLS = ("cream", "grey", "black")
CUBE_TO_BOWL = dict(zip(COLOURS, BOWLS))  # red -> cream, green -> grey, blue -> black
BOWL_RGBA = {"cream": "0.93 0.89 0.78 1", "grey": "0.55 0.55 0.57 1", "black": "0.12 0.12 0.12 1"}
NOMINAL_BOWLS = dict(zip(BOWLS, (NOMINAL_BINS[c] for c in COLOURS)))  # same three spots as the bins
BOWL_INNER_R = 0.060   # 120 mm inner diameter, measured on the printed bowls
BOWL_WALL = 0.005
BOWL_HEIGHT = 0.040      # 40 mm walls, measured on the printed bowls
WALL_SEGMENTS = 16


def _bowl_xml(colour, xy):
    r, w, h = BOWL_INNER_R, BOWL_WALL / 2, BOWL_HEIGHT / 2
    seg = math.pi * (r + w) / WALL_SEGMENTS  # half-length of one wall segment along the ring
    walls = []
    for i in range(WALL_SEGMENTS):
        a = 2 * math.pi * i / WALL_SEGMENTS
        walls.append(f'<geom type="box" pos="{(r + w) * math.cos(a):.4f} {(r + w) * math.sin(a):.4f} {h}" '
                     f'size="{w} {seg:.4f} {h}" euler="0 0 {math.degrees(a):.2f}" rgba="{BOWL_RGBA[colour]}"/>')
    return f'''<body name="bowl_{colour}" pos="{xy[0]} {xy[1]} 0">
      <geom type="cylinder" pos="0 0 {w}" size="{r + 2 * w} {w}" rgba="{BOWL_RGBA[colour]}"/>
      {chr(10).join(walls)}
    </body>'''


class BowlSim(SortSim):
    def build(self, params: dict) -> mujoco.MjModel:
        bowls = "\n".join(_bowl_xml(b, NOMINAL_BOWLS[b] + bin_offset(params, c)) for c, b in CUBE_TO_BOWL.items())
        return build_model(params, fixtures=bowls, title="gaphunter cube in bowl")

    def bowl_true(self, bowl):
        cube_colour = next(c for c, b in CUBE_TO_BOWL.items() if b == bowl)
        return NOMINAL_BOWLS[bowl] + bin_offset(self.params, cube_colour)

    def bin_true(self, colour):  # keeps SortSim.bin_of_cube semantics by cube colour
        return self.bowl_true(CUBE_TO_BOWL[colour])

    def bin_of_cube(self):
        """Cube colour whose bowl the cube rests in, or None (round test instead of the bins' square one)."""
        x, y, z = self.cube_true()
        for c in COLOURS:
            bx, by = self.bin_true(c)
            if math.hypot(x - bx, y - by) < BOWL_INNER_R and z < BOWL_HEIGHT:
                return c
        return None

    def outcome(self) -> dict:
        o = super().outcome()
        o["bowl"] = CUBE_TO_BOWL.get(o["bin"]) if o["bin"] else None
        o["target_bowl"] = CUBE_TO_BOWL[self.colour]
        return o

    def bowl_pose(self, bowl):
        est_xy = self._estimate(self.bowl_true(bowl), NOMINAL_BOWLS[bowl])
        result = ((float(est_xy[0]), float(est_xy[1]), BOWL_HEIGHT), 0.0)
        self._count("bowl_pose", {"bowl": bowl}, [list(result[0]), result[1]])
        return result
