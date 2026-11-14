import math

SAFE_Z = 0.22          # transport height for the pad centre
CUBE = 0.04
TARGET = {"red": "cream", "green": "grey", "blue": "black"}
MAX_CALLS = 300


class _Budget(Exception):
    pass


def run(api):
    state = {"n": 0, "pos": None}

    def tick(reserve=0):
        if state["n"] + 1 + reserve > MAX_CALLS:
            raise _Budget()
        state["n"] += 1

    def raw_move(p, yaw):
        tick(reserve=2)
        api.move_to((p[0], p[1], p[2]), yaw)
        state["pos"] = (p[0], p[1], p[2])

    def move(p, yaw, step=0.08):
        """Cartesian-ish move: split into short segments so the joint-space path stays near a line."""
        s = state["pos"]
        if s is None:
            raw_move(p, yaw)
            return
        d = math.sqrt(sum((p[i] - s[i]) ** 2 for i in range(3)))
        n = max(1, int(math.ceil(d / step)))
        for k in range(1, n + 1):
            t = k / float(n)
            raw_move(tuple(s[i] + (p[i] - s[i]) * t for i in range(3)), yaw)

    def suction(on):
        tick(reserve=1)
        return api.suction(on)

    def is_done():
        tick()
        return api.done()

    def attempt():
        tick()
        (cx, cy, ctop), cyaw, colour = api.cube_pose()
        bowl = TARGET.get(colour, "grey")
        tick()
        (bx, by, brim), _ = api.bowl_pose(bowl)

        yaw = cyaw
        # go above the cube at safe height (lift straight up first if we are low)
        if state["pos"] is not None and state["pos"][2] < SAFE_Z - 1e-4:
            move((state["pos"][0], state["pos"][1], SAFE_Z), yaw, 0.04)
        move((cx, cy, SAFE_Z), yaw)
        suction(False)
        # descend close to the top face
        move((cx, cy, ctop + 0.03), yaw, 0.04)
        held = False
        for dz in (0.004, 0.002, 0.001, 0.0, -0.001, -0.002):
            raw_move((cx, cy, ctop + dz), yaw)
            if suction(True):
                held = True
                break
        if not held:
            suction(False)
            move((cx, cy, SAFE_Z), yaw, 0.04)
            return False

        # lift, carry, lower into the bowl
        move((cx, cy, SAFE_Z), yaw, 0.03)
        move((bx, by, SAFE_Z), yaw, 0.07)
        floor = max(0.0, brim - 0.04)
        z_rel = floor + CUBE + 0.012
        move((bx, by, z_rel), yaw, 0.03)
        suction(False)
        move((bx, by, SAFE_Z), yaw, 0.04)
        return True

    try:
        for _ in range(6):
            if state["n"] > MAX_CALLS - 60:
                break
            attempt()
            if is_done():
                break
    except _Budget:
        pass
    try:
        if state["n"] < MAX_CALLS:
            api.suction(False)
    except Exception:
        pass
