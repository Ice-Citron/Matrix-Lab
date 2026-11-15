import math

SAFE_Z = 0.20      # pad height for transit, well above bowl rims (cube hangs 40 mm below)
CUBE = 0.04
STEPS = 6          # waypoints per lateral transit, keeps joint-space moves close to a straight line


def run(api):
    state = {"pos": None, "yaw": 0.0}

    def move(xyz, yaw):
        api.move_to(xyz, yaw)
        state["pos"] = (xyz[0], xyz[1], xyz[2])
        state["yaw"] = yaw

    def travel(xyz, yaw):
        """Rise vertically, move laterally at safe height in small steps, descend vertically."""
        if state["pos"] is None:
            move((xyz[0], xyz[1], SAFE_Z), yaw)
        else:
            x0, y0, z0 = state["pos"]
            yaw0 = state["yaw"]
            if z0 < SAFE_Z:
                move((x0, y0, SAFE_Z), yaw0)
            dist = math.hypot(xyz[0] - x0, xyz[1] - y0)
            n = max(1, int(math.ceil(dist / 0.06))) if dist > 1e-4 else 1
            n = min(n, STEPS)
            for i in range(1, n + 1):
                t = i / float(n)
                move((x0 + (xyz[0] - x0) * t, y0 + (xyz[1] - y0) * t, SAFE_Z),
                     yaw0 + (yaw - yaw0) * t)
        if xyz[2] < SAFE_Z:
            move((xyz[0], xyz[1], xyz[2] + 0.03 if xyz[2] + 0.03 < SAFE_Z else xyz[2]), yaw)
            move(tuple(xyz), yaw)

    def grasp():
        (x, y, zt), yaw, _ = api.cube_pose()
        travel((x, y, zt + 0.03), yaw)
        for dz in (0.004, 0.002, 0.0, -0.002, -0.004):
            move((x, y, zt + dz), yaw)
            if api.suction(True):
                return True
        api.suction(False)
        move((x, y, zt + 0.03), yaw)
        return False

    for attempt in range(5):
        if api.done():
            return
        if not grasp():
            continue
        # lift straight up
        px, py, _ = state["pos"]
        move((px, py, SAFE_Z), state["yaw"])

        # measure how the cube hangs relative to the pad
        (cx, cy, ctop), _, colour = api.cube_pose()
        px, py, pz = state["pos"]
        offx, offy = cx - px, cy - py
        hang = pz - ctop  # pad height above cube top (~0)
        if math.hypot(offx, offy) > 0.03:
            offx, offy = 0.0, 0.0

        (bx, by, zrim), _ = api.bowl_pose(colour)
        # cube bottom 20 mm below the rim, still above the floor
        target_z = zrim - 0.02 + CUBE + hang
        target_z = max(target_z, CUBE + 0.01)
        tx, ty = bx - offx, by - offy
        travel((tx, ty, min(target_z, SAFE_Z)), state["yaw"])
        api.suction(False)
        move((tx, ty, SAFE_Z), state["yaw"])
        if api.done():
            return
    return
