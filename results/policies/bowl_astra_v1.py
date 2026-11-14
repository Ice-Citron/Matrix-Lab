import math


def run(api):
    bowls = {"red": "cream", "green": "grey", "blue": "black"}
    commanded_position = None
    commanded_yaw = 0.0

    def move(target, yaw):
        """Use short Cartesian waypoints to limit joint-space path excursions."""
        nonlocal commanded_position, commanded_yaw
        target = tuple(target)

        if commanded_position is None:
            api.move_to(target, yaw)
        else:
            start = commanded_position
            start_yaw = commanded_yaw
            distance = math.sqrt(sum((target[i] - start[i]) ** 2
                                     for i in range(3)))
            steps = min(12, max(1, math.ceil(distance / 0.02)))
            for step in range(1, steps + 1):
                fraction = step / steps
                point = (
                    target if step == steps else
                    tuple(start[i] + fraction * (target[i] - start[i])
                          for i in range(3))
                )
                angle = start_yaw + fraction * (yaw - start_yaw)
                api.move_to(point, angle)

        commanded_position = target
        commanded_yaw = yaw

    api.suction(False)

    # Bounded retries; even with every waypoint count saturated, fewer
    # than 250 API calls are made.
    for _ in range(3):
        if api.done():
            return

        (cx, cy, top), yaw, colour = api.cube_pose()
        (bx, by, rim), _ = api.bowl_pose(bowls[colour])

        # A held cube extends 40 mm below the controlled pad centre.
        clearance = max(0.20, top + 0.12, rim + 0.12)

        move((cx, cy, clearance), yaw)
        move((cx, cy, top), yaw)

        if not api.suction(True):
            # Do not probe below the estimated face: there is no force
            # feedback with which to bound the resulting contact force.
            api.suction(False)
            move((cx, cy, clearance), yaw)
            continue

        move((cx, cy, clearance), yaw)
        move((bx, by, clearance), yaw)

        # Release centrally with the cube's bottom 20 mm above the rim.
        # This avoids driving the pad or held cube into the bowl floor.
        move((bx, by, rim + 0.04 + 0.02), yaw)
        api.suction(False)

        if api.done():
            return

        move((bx, by, clearance), yaw)
