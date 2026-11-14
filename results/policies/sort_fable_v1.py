import math

CUBE = 0.040          # cube edge (m)
WALL = 0.060          # bin wall height (m)
MAX_CALLS = 300


class _Budget:
    """Counts API calls so we never exceed the per-trial limit."""

    def __init__(self, api):
        self.api = api
        self.n = 0

    def left(self):
        return MAX_CALLS - self.n

    def cube_pose(self):
        self.n += 1
        return self.api.cube_pose()

    def bin_pose(self, c):
        self.n += 1
        return self.api.bin_pose(c)

    def move_to(self, xyz, yaw):
        self.n += 1
        return self.api.move_to((float(xyz[0]), float(xyz[1]), float(xyz[2])), float(yaw))

    def suction(self, on):
        self.n += 1
        return self.api.suction(on)

    def done(self):
        self.n += 1
        return self.api.done()


def _wrap45(a):
    """Wrap an angle to [-45, 45): a square looks the same every 90 degrees."""
    return (a + 45.0) % 90.0 - 45.0


def _sense_cube(b, n=3):
    """Average several camera readings (circular mean for yaw, majority for colour)."""
    xs = ys = zs = 0.0
    s = c = 0.0
    votes = {}
    for _ in range(n):
        (x, y, z), yaw, col = b.cube_pose()
        xs += x
        ys += y
        zs += z
        s += math.sin(math.radians(4.0 * yaw))
        c += math.cos(math.radians(4.0 * yaw))
        votes[col] = votes.get(col, 0) + 1
    yaw = math.degrees(math.atan2(s, c)) / 4.0
    colour = max(votes, key=lambda k: votes[k])
    return (xs / n, ys / n, zs / n), _wrap45(yaw), colour


def _line(b, p0, p1, yaw, step=0.05):
    """Approximate a straight Cartesian line with short joint-space moves."""
    d = math.sqrt(sum((p1[i] - p0[i]) ** 2 for i in range(3)))
    k = max(1, int(math.ceil(d / step)))
    for i in range(1, k + 1):
        t = i / float(k)
        b.move_to([p0[j] + t * (p1[j] - p0[j]) for j in range(3)], yaw)
    return p1


def run(api):
    b = _Budget(api)
    try:
        for _attempt in range(4):
            if b.left() < 75:
                break

            (cx, cy, cz), cyaw, colour = _sense_cube(b)
            (bx, by, brim), byaw = b.bin_pose(colour)
            byaw = _wrap45(byaw)

            # Pad height at which a hanging cube clears every bin wall.
            z_safe = max(brim, WALL, cz) + CUBE + 0.08

            # --- approach above the cube, then descend vertically ---
            b.suction(False)
            p = (cx, cy, z_safe)
            b.move_to(p, cyaw)
            p = _line(b, p, (cx, cy, cz + 0.03), cyaw)
            p = _line(b, p, (cx, cy, cz + 0.004), cyaw, step=0.01)

            # --- gentle touch-down: 1 mm steps, try the vacuum at each one ---
            held = b.suction(True)
            dz = 0.004
            while not held and dz > -0.004:
                dz -= 0.001
                p = (cx, cy, cz + dz)
                b.move_to(p, cyaw)
                held = b.suction(True)

            if not held:
                b.suction(False)
                _line(b, p, (cx, cy, z_safe), cyaw)
                continue

            # --- lift straight up, verify grip ---
            p = _line(b, p, (cx, cy, cz + 0.02), cyaw, step=0.01)
            p = _line(b, p, (cx, cy, z_safe), cyaw)
            if not b.suction(True):
                b.suction(False)
                continue

            # --- carry over to the bin at safe height, align cube with bin ---
            p = _line(b, p, (bx, by, z_safe), cyaw)
            b.move_to(p, byaw)

            # --- lower inside the bin, release just above the floor ---
            floor = max(brim - WALL, 0.0)
            z_rel = floor + CUBE + 0.012
            p = _line(b, p, (bx, by, z_rel), byaw, step=0.04)
            b.suction(False)

            # --- retreat and check ---
            _line(b, p, (bx, by, z_safe), byaw)
            if b.done():
                return
    except Exception:
        pass
    try:
        if b.left() > 0:
            b.suction(False)
    except Exception:
        pass
