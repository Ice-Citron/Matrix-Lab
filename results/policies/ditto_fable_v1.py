import math


def run(api):
    SAFE_Z = 0.22          # travel height: sphere clears the 0.08 m rim easily
    MAX_CALLS = 60
    calls = [0]

    class _Budget(Exception):
        pass

    def call(fn, *args):
        # keep one call in reserve so we can always open the gripper at the end
        if calls[0] >= MAX_CALLS - 1:
            raise _Budget()
        calls[0] += 1
        return fn(*args)

    last = [None]  # last known gripper position

    def goto(x, y, z, correct=False):
        r = call(api.move_to, (x, y, z))
        try:
            rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
        except Exception:
            rx, ry, rz = x, y, z
        if correct:
            ex, ey = x - rx, y - ry
            if 0.002 < math.hypot(ex, ey) < 0.03:
                r2 = call(api.move_to, (x + ex, y + ey, z))
                try:
                    rx, ry, rz = float(r2[0]), float(r2[1]), float(r2[2])
                except Exception:
                    rx, ry, rz = x, y, z
        last[0] = (rx, ry, rz)
        return last[0]

    # grasp-height offsets relative to the sphere centre, varied between attempts
    z_offsets = [-0.006, 0.0, -0.012, -0.004]

    try:
        basket = None
        for attempt in range(4):
            call(api.set_gripper, False)
            if last[0] is not None:
                # retreat straight up before doing anything else
                goto(last[0][0], last[0][1], SAFE_Z)
            if call(api.done):
                return
            if basket is None:
                basket = call(api.basket_position)
            bx, by, bz = float(basket[0]), float(basket[1]), float(basket[2])

            o = call(api.object_position)
            ox, oy, oz = float(o[0]), float(o[1]), float(o[2])

            # sphere already resting in the basket: nothing more to pick
            if abs(ox - bx) < 0.09 and abs(oy - by) < 0.07 and oz < bz + 0.08:
                if last[0] is None:
                    return
                if call(api.done):
                    return
                return

            gz = max(oz + z_offsets[attempt], 0.012)

            # approach from above, then descend vertically around the sphere
            goto(ox, oy, SAFE_Z)
            g = goto(ox, oy, gz, correct=True)

            # make sure the sphere was not bumped during the approach
            o2 = call(api.object_position)
            if math.hypot(float(o2[0]) - g[0], float(o2[1]) - g[1]) > 0.006:
                continue

            call(api.set_gripper, True)
            goto(ox, oy, SAFE_Z)

            # verify the grasp held
            o3 = call(api.object_position)
            if float(o3[2]) < 0.10:
                continue

            # carry over the basket, lower inside the rim, release, retreat
            goto(bx, by, SAFE_Z)
            goto(bx, by, bz + 0.06)
            call(api.set_gripper, False)
            goto(bx, by, SAFE_Z)
            if call(api.done):
                return
    except _Budget:
        pass

    # out of attempts or budget: make sure the sphere is released
    if calls[0] < MAX_CALLS:
        try:
            api.set_gripper(False)
        except Exception:
            pass
