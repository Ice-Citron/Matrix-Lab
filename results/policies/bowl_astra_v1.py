import math


def run(api):
    (cx, cy, z_top), cube_yaw, colour = api.cube_pose()
    (bx, by, z_rim), _ = api.bowl_pose(colour)

    cube_size = 0.040
    wall_height = 0.040
    floor_z = z_rim - wall_height

    # Clearance accounts for the cube hanging below the controlled pad.
    travel_z = max(0.200, z_top + 0.120, z_rim + cube_size + 0.100)
    release_z = floor_z + cube_size + 0.003

    def move_segment(start, end):
        # Short joint-space moves approximate the desired Cartesian segment.
        # The cap also bounds the total API-call count.
        distance = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))
        steps = max(1, min(48, math.ceil(distance / 0.010)))
        for k in range(1, steps + 1):
            t = k / steps
            point = tuple(start[i] + t * (end[i] - start[i])
                          for i in range(3))
            api.move_to(point, cube_yaw)

    pick_high = (cx, cy, travel_z)
    pick_contact = (cx, cy, z_top)
    bowl_high = (bx, by, travel_z)
    release_point = (bx, by, release_z)

    api.suction(False)
    api.move_to(pick_high, cube_yaw)

    # Match the square pad to the cube and touch without commanded overtravel.
    move_segment(pick_high, pick_contact)
    if not api.suction(True):
        api.suction(False)
        move_segment(pick_contact, pick_high)
        return False

    # Lift before translating; preserve the grasp yaw throughout transport.
    move_segment(pick_contact, pick_high)
    move_segment(pick_high, bowl_high)
    move_segment(bowl_high, release_point)

    api.suction(False)
    move_segment(release_point, bowl_high)
    return api.done()
