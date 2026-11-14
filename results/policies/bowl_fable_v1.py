import math

CUBE = 0.040  # cube edge, m


def _median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def _observe_cube(api, n=3):
    xs, ys, zs, yaws, cols = [], [], [], [], []
    for _ in range(n):
        (x, y, z), yaw, col = api.cube_pose()
        xs.append(x); ys.append(y); zs.append(z); yaws.append(yaw); cols.append(col)
    colour = max(set(cols), key=cols.count)
    return (_median(xs), _median(ys), _median(zs)), _median(yaws), colour


def _line(api, p0, p1, yaw, step):
    """Move from p0 to p1 in short Cartesian segments (move_to is joint-space)."""
    d = math.sqrt(sum((b - a) ** 2 for a, b in zip(p0, p1)))
    n = max(1, int(math.ceil(d / step)))
    for i in range(1, n + 1):
        t = i / float(n)
        api.move_to((p0[0] + (p1[0] - p0[0]) * t,
                     p0[1] + (p1[1] - p0[1]) * t,
                     p0[2] + (p1[2] - p0[2]) * t), yaw)
    return p1


def _attempt(api):
    (cx, cy, ctop), cyaw, colour = _observe_cube(api)
    (bx, by, brim), _ = api.bowl_pose(colour)

    rim_max = brim
    for c in ("cream", "grey", "black"):
        try:
            rim_max = max(rim_max, api.bowl_pose(c)[0][2])
        except Exception:
            pass

    # pad height at which a held cube clears every rim comfortably
    travel_z = max(rim_max, ctop) + CUBE + 0.07
    yaw = cyaw

    # ---- pick ----
    api.suction(False)
    pos = (cx, cy, travel_z)
    api.move_to(pos, yaw)
    pos = _line(api, pos, (cx, cy, ctop + 0.03), yaw, 0.05)
    held = False
    for dz in (0.002, 0.0005, -0.001, -0.002, -0.003):
        pos = (cx, cy, ctop + dz)
        api.move_to(pos, yaw)
        if api.suction(True):
            held = True
            break
    if not held:
        api.suction(False)
        _line(api, pos, (cx, cy, travel_z), yaw, 0.05)
        return False

    # ---- lift, carry, lower ----
    pos = _line(api, pos, (cx, cy, travel_z), yaw, 0.04)
    pos = _line(api, pos, (bx, by, travel_z), yaw, 0.05)

    floor_z = max(0.0, brim - 0.040)
    release_z = floor_z + CUBE + 0.008
    pos = _line(api, pos, (bx, by, release_z), yaw, 0.04)

    api.suction(False)
    _line(api, pos, (bx, by, travel_z), yaw, 0.06)
    return True


def run(api):
    for _ in range(4):
        try:
            if api.done():
                return
        except Exception:
            pass
        _attempt(api)
    try:
        api.done()
    except Exception:
        pass
