import math

MAX_CALLS = 290          # keep a margin under the 300-call limit
HOVER = 0.002            # m above the top surface for lateral moves
CONTACT_N = 5.0          # force that means "sitting on the block surface"
JAM_N = 30.0             # force that means "jammed while inserting"
XY_STEP = 0.0006         # m, search pitch (clearance is 0.5 mm per side)
YAW_STEP = 2.0           # deg, search pitch (a 20 mm peg in a 21 mm hole tolerates ~2.9 deg)


class _Budget(Exception):
    pass


def run(api):
    calls = [0]

    def use():
        calls[0] += 1
        if calls[0] > MAX_CALLS:
            raise _Budget()

    def hole_pose():
        use()
        return api.hole_pose()

    def move_to(p, yaw):
        use()
        return api.move_to(p, yaw)

    def step_down(mm):
        use()
        f = api.step_down(mm)
        return 0.0 if f is None else f

    def done():
        use()
        return bool(api.done())

    try:
        # Average a few camera readings (yaw averaged on the 90-degree circle).
        n = 3
        sx = sy = sz = 0.0
        sc = ss = 0.0
        for _ in range(n):
            (x, y, z), yaw = hole_pose()
            sx += x
            sy += y
            sz += z
            sc += math.cos(math.radians(yaw * 4.0))
            ss += math.sin(math.radians(yaw * 4.0))
        hx, hy, hz = sx / n, sy / n, sz / n
        hyaw = math.degrees(math.atan2(ss, sc)) / 4.0

        # Safe approach: high above the hole first, then straight down in short moves.
        move_to((hx, hy, hz + 0.10), hyaw)
        move_to((hx, hy, hz + 0.02), hyaw)

        # Candidate offsets (in the hole frame), nearest first, over x, y and yaw.
        cands = []
        R = 4
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                for k in range(-R, R + 1):
                    d = i * i + j * j + k * k
                    if d <= R * R:
                        cands.append((d, abs(k), i, j, k))
        cands.sort()

        c, s = math.cos(math.radians(hyaw)), math.sin(math.radians(hyaw))

        for _, _, i, j, k in cands:
            dx, dy = i * XY_STEP, j * XY_STEP
            px = hx + c * dx - s * dy
            py = hy + s * dx + c * dy
            yaw = hyaw + k * YAW_STEP
            move_to((px, py, hz + HOVER), yaw)

            # Probe: 0.5 mm below the surface, then 2.5 mm below.
            if step_down(2.5) > CONTACT_N:
                continue
            if step_down(2.0) > CONTACT_N:
                continue

            # We are in the hole: insert in steps, watching for jams.
            depth = 2.5
            jammed = False
            while depth < 36.0:
                stp = min(4.0, 36.0 - depth)
                f = step_down(stp)
                depth += stp
                if f > JAM_N:
                    if depth >= 30.0 and done():
                        return
                    jammed = True
                    break
            if not jammed:
                if done():
                    return
                # Slightly deeper in case the surface height estimate was off.
                for _ in range(3):
                    f = step_down(2.0)
                    depth += 2.0
                    if done():
                        return
                    if f > JAM_N:
                        break
                return
            # Jammed: back out fully, then try the next candidate.
            step_down(-(depth + HOVER * 1000.0))
    except _Budget:
        return
