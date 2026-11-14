"""The five functions a sorting policy may call. A policy is `def run(api): ...` and receives this module.

Units: metres and degrees in the world frame; z is up and the table top is z = 0.
The robot has a suction cup pointing straight down. The centre of its 28 x 28 mm square pad is the controlled
point. One 40 mm cube lies on the table; its colour is cream, grey or black. Three round bowls stand on the
table, one per colour. Each bowl is a dish with a flat floor, 120 mm across inside, with 40 mm high walls.
The trial succeeds when the cube rests inside the bowl of its own colour with the suction off.
"""
from __future__ import annotations

_SIM = None  # set by the trial runner


def cube_pose():
    """Camera estimate of the cube: returns ((x, y, z_top_face), yaw_deg, colour).
    Yaw is the rotation of the cube about z, reported in [-45, 45) because a square repeats every 90 degrees.
    colour is "cream", "grey" or "black"."""
    return _SIM.cube_pose()


def bowl_pose(colour):
    """Camera estimate of the bowl for colour ("cream", "grey" or "black"): returns ((x, y, z_rim), yaw_deg).
    (x, y) is the centre of the bowl floor; z_rim is the height of the top of its wall."""
    return _SIM.bowl_pose(colour)


def move_to(xyz, yaw_deg):
    """Move the pad centre to xyz (metres) with the pad rotated yaw_deg about z, cup pointing down.
    Joint-space motion; blocks until the arm has settled. A held cube moves with the cup.
    The run is aborted as a collision if the contact force on the pad ever exceeds 150 N."""
    return _SIM.move_to(xyz, yaw_deg)


def suction(on):
    """Switch the vacuum on (True) or off (False), then wait for things to settle.
    The pad grips only what it touches. Returns True if the cup now holds the cube, else False.
    Switching off releases the cube, which drops from where it is."""
    return _SIM.suction(on)


def done():
    """True if the cube rests inside the bowl of its own colour (each bowl has a load cell)."""
    return _SIM.done()
