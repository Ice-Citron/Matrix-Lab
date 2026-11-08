"""Colour sorting with aligned pickup and clearance-height transport."""

import math


def run(api):
    bins = {"red", "green", "blue"}

    def segment(start, end, yaw, spacing=0.01):
        """Use short waypoint moves to limit joint-space path excursions."""
        distance = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))
        count = max(1, int(math.ceil(distance / spacing)))
        for k in range(1, count + 1):
            t = k / count
            point = tuple(start[i] + t * (end[i] - start[i])
                          for i in range(3))
            api.move_to(point, yaw)
        return end

    api.suction(False)

    # Two bounded attempts leave ample room within the 300-call budget.
    for _ in range(2):
        (x, y, top), yaw, colour = api.cube_pose()
        if colour not in bins:
            return

        (bx, by, rim), _ = api.bin_pose(colour)
        clearance = max(0.20, rim + 0.12, top + 0.12)

        position = (x, y, clearance)
        api.move_to(position, yaw)
        position = segment(position, (x, y, top + 0.003), yaw)

        held = False
        # Align the square pad with the cube and approach in 0.25 mm steps.
        # Limit the nominal compression to 0.5 mm.
        for k in range(15):
            position = (x, y, top + 0.003 - 0.00025 * k)
            api.move_to(position, yaw)
            if api.suction(True):
                held = True
                break

        if not held:
            api.suction(False)
            segment(position, (x, y, clearance), yaw)
            continue

        # Lift before moving toward any bin walls.
        position = segment(position, (x, y, clearance), yaw)
        position = segment(position, (bx, by, clearance), yaw)

        # The cube's bottom remains 10 mm above the rim at release.
        # Dropping centrally avoids inserting the cup into the bin.
        position = segment(position, (bx, by, rim + 0.050), yaw)
        api.suction(False)
        api.done()
        return
