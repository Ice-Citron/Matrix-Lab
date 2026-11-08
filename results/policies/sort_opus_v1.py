import math

SAFE_Z = 0.25        # transit height of the pad centre (held cube bottom stays far above 60 mm rims)
CUBE = 0.04          # cube edge length (m)
MAX_CALLS = 290      # stay below the 300-call budget


def run(api):
    calls = [0]

    def budget(n=1):
        return calls[0] + n <= MAX_CALLS

    def call(fn, *args):
        calls[0] += 1
        return fn(*args)

    def wrap_yaw(y):
        # Square pad on a square cube: any multiple of 90 degrees is equivalent.
        while y >= 45.0:
            y -= 90.0
        while y < -45.0:
            y += 90.0
        return y

    state = {"pos": None, "yaw": 0.0}

    def go(xyz, yaw, step=0.05):
        """Move in short segments so joint-space motion stays close to a straight line."""
        if state["pos"] is None:
            ok = call(api.move_to, tuple(xyz), yaw)
            state["pos"], state["yaw"] = list(xyz), yaw
            return ok
        p0, y0 = state["pos"], state["yaw"]
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, xyz)))
        n = max(1, int(math.ceil(d / step)))
        n = max(n, int(math.ceil(abs(yaw - y0) / 15.0)))
        ok = None
        for i in range(1, n + 1):
            t = float(i) / n
            p = [p0[k] + (xyz[k] - p0[k]) * t for k in range(3)]
            ok = call(api.move_to, tuple(p), y0 + (yaw - y0) * t)
        state["pos"], state["yaw"] = list(xyz), yaw
        return ok

    def lift():
        if state["pos"] is not None and state["pos"][2] < SAFE_Z - 1e-4:
            go([state["pos"][0], state["pos"][1], SAFE_Z], state["yaw"], step=0.05)

    # Start from a safe height.
    (cx, cy, cz), cyaw, colour = call(api.cube_pose)
    go([cx, cy, SAFE_Z], wrap_yaw(cyaw), step=0.10)

    for attempt in range(6):
        if not budget(30):
            break

        # ---------- locate cube ----------
        (cx, cy, cz), cyaw, colour = call(api.cube_pose)
        yaw = wrap_yaw(cyaw)
        go([cx, cy, SAFE_Z], yaw, step=0.10)
        # Re-measure while hovering directly above it.
        (cx, cy, cz), cyaw, colour = call(api.cube_pose)
        yaw = wrap_yaw(cyaw)
        go([cx, cy, SAFE_Z], yaw, step=0.10)

        # ---------- descend and grasp ----------
        go([cx, cy, cz + 0.02], yaw, step=0.05)
        held = False
        z = cz + 0.004
        while z >= cz - 0.008 and budget(4):
            go([cx, cy, z], yaw, step=0.01)
            if call(api.suction, True):
                held = True
                break
            z -= 0.002
        if not held:
            call(api.suction, False)
            lift()
            continue

        lift()

        # ---------- verify grip and measure grasp offset ----------
        (hx, hy, hz), hyaw, colour = call(api.cube_pose)
        px, py, pz = state["pos"]
        if abs(hz - pz) > 0.02:
            # Cube was lost during lift; release and retry.
            call(api.suction, False)
            continue
        offx, offy = hx - px, hy - py
        if math.hypot(offx, offy) > 0.03:
            offx, offy = 0.0, 0.0  # implausible estimate; ignore

        # ---------- transport to the correct bin ----------
        (bx, by, brim), byaw = call(api.bin_pose, colour)
        tx, ty = bx - offx, by - offy
        go([tx, ty, SAFE_Z], state["yaw"], step=0.10)

        # Lower so the cube bottom sits ~25 mm below the rim, well inside the bin.
        release_z = brim + 0.015
        go([tx, ty, release_z], state["yaw"], step=0.03)
        call(api.suction, False)
        lift()

        if call(api.done):
            return
        # Otherwise loop: re-locate the cube wherever it ended up and try again.

    # Make sure suction is off before returning.
    if budget(1):
        call(api.suction, False)
