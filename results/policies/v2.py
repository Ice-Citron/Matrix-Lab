import math


# Empirical camera calibration supplied with the failure reports:
# (camera_x, camera_y, measured_hole_x, measured_hole_y).
# These are calibration observations, not assumed trial locations.
_CALIBRATION = (
    (0.618,  0.001, 0.610,  0.000),
    (0.666,  0.128, 0.644,  0.116),
    (0.404,  0.116, 0.405,  0.112),
    (0.252,  0.087, 0.261,  0.080),
    (0.255, -0.216, 0.273, -0.184),
    (0.256, -0.167, 0.270, -0.148),
    (0.256, -0.058, 0.267, -0.055),
    (0.337, -0.043, 0.340, -0.041),
)


def _wrap(yaw):
    return (yaw + 45.0) % 90.0 - 45.0


def _median(values):
    values = sorted(values)
    n = len(values)
    return 0.5 * (values[(n - 1) // 2] + values[n // 2])


def _radial_inverse(x, y):
    """Empirical outward-distortion prior, not an API sensor model."""
    dx, dy = x - 0.45, y
    observed_radius = math.hypot(dx, dy)
    radius = observed_radius
    for _ in range(10):
        radius -= (
            radius + 2.0 * radius**3 - observed_radius
        ) / (1.0 + 6.0 * radius**2)
    scale = radius / observed_radius if observed_radius else 1.0
    return 0.45 + scale * dx, scale * dy


def _correct_camera(x, y):
    """Interpolate measured residuals around a smooth distortion prior."""
    bx, by = _radial_inverse(x, y)
    total_weight = 1.0
    residual_x = residual_y = 0.0

    for mx, my, tx, ty in _CALIBRATION:
        distance_squared = (x - mx)**2 + (y - my)**2
        if distance_squared < 1.0e-14:
            return tx, ty

        px, py = _radial_inverse(mx, my)
        weight = (0.045**2 / distance_squared)**2
        total_weight += weight
        residual_x += weight * (tx - px)
        residual_y += weight * (ty - py)

    return (
        bx + residual_x / total_weight,
        by + residual_y / total_weight,
    )


def run(api):
    """Calibrated alignment followed by bounded, force-monitored probing."""
    calls = 0
    seated = False
    target_z = None

    # The block height is specified independently of the camera.
    top = 0.050
    safe_z = top + 0.015
    hover_z = top + 0.001
    force_limit = 4.0

    class BudgetExhausted(Exception):
        pass

    def call(function, *args):
        nonlocal calls
        # Always reserve the final call for a vertical withdrawal.
        if calls >= 299:
            raise BudgetExhausted()
        calls += 1
        return function(*args)

    def lower(mm):
        nonlocal target_z
        force = call(api.step_down, mm)
        # Track commanded, not presumed achieved, position.
        target_z -= 0.001 * mm
        return force

    def clear(force):
        return math.isfinite(force) and abs(force) < force_limit

    def retract():
        nonlocal target_z
        if target_z is not None and target_z < safe_z:
            lower(1000.0 * (target_z - safe_z))

    try:
        readings = []
        for _ in range(17):
            (x, y, z), yaw = call(api.hole_pose)
            if not all(math.isfinite(v) for v in (x, y, z, yaw)):
                return
            readings.append((x, y, _wrap(yaw)))

            # Repeated identical observations provide no new information.
            if len(readings) >= 4 and all(
                reading == readings[0] for reading in readings
            ):
                break

        mx = _median([r[0] for r in readings])
        my = _median([r[1] for r in readings])

        # Average square orientations across the +/-45-degree boundary.
        sy = sum(math.sin(math.radians(4.0 * r[2])) for r in readings)
        cy = sum(math.cos(math.radians(4.0 * r[2])) for r in readings)
        if math.hypot(sy, cy) < 1.0e-8:
            return
        yaw = _wrap(math.degrees(math.atan2(sy, cy)) / 4.0)
        cx, cy = _correct_camera(mx, my)

        angle = math.radians(yaw)
        c, s = math.cos(angle), math.sin(angle)

        # Joint translation/yaw search, rather than exhausting one axis first.
        spacing = 0.00045
        candidates = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                for dyaw in (0.0, -1.25, 1.25, -2.5, 2.5):
                    score = i * i + j * j + 0.8 * (dyaw / 1.25)**2
                    candidates.append((score, i, j, dyaw))
        candidates.sort()

        call(api.move_to, (cx, cy, safe_z), yaw)
        target_z = safe_z

        for _, i, j, dyaw in candidates:
            # Enough for positioning, probing, a complete insertion, and exit.
            if calls + 55 > 299:
                break

            dx, dy = spacing * i, spacing * j
            x = cx + c * dx - s * dy
            y = cy + s * dx + c * dy
            trial_yaw = _wrap(yaw + dyaw)

            # All lateral and rotational changes occur above the surface.
            call(api.move_to, (x, y, safe_z), trial_yaw)
            target_z = safe_z
            call(api.move_to, (x, y, hover_z), trial_yaw)
            target_z = hover_z

            force = lower(1.0)
            blocked = not clear(force)

            # Reject contact early instead of accumulating spring deflection.
            if not blocked:
                for _ in range(6):
                    force = lower(0.2)
                    if not clear(force):
                        blocked = True
                        break

            if blocked:
                retract()
                continue

            # Remain slow through initial engagement. Once engaged, there
            # are no lateral/yaw commands and insertion remains force-checked.
            while target_z > top - 0.032:
                depth_mm = 1000.0 * (top - target_z)
                increment = 0.4 if depth_mm < 3.0 else 1.0
                increment = min(
                    increment, 1000.0 * (target_z - (top - 0.032))
                )
                force = lower(increment)

                if not clear(force):
                    if target_z <= top - 0.030:
                        seated = bool(call(api.done))
                    break

                if target_z <= top - 0.030:
                    seated = bool(call(api.done))
                    if seated:
                        return

            if seated:
                return
            retract()

    except BudgetExhausted:
        pass
    finally:
        if not seated and target_z is not None and target_z < safe_z:
            if calls < 300:
                calls += 1
                api.step_down(1000.0 * (target_z - safe_z))
