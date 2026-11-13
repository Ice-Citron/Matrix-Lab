import math


_TOP = 0.050
_SAFE_Z = 0.075
_HOVER_Z = 0.051
_FINAL_Z = 0.0165

# Empirical camera model retained from the supplied calibration.
_RADIAL_K = 2.0
_YAW_SCALE = 1.04

_FREE_FORCE = 0.8
_CONTACT_LIMIT = 12.0
_INSERT_RESERVE = 45


def _wrap(angle):
    return (angle + 45.0) % 90.0 - 45.0


def _trimmed_mean(values):
    values = sorted(values)
    trim = len(values) // 5
    if trim:
        values = values[trim:-trim]
    return sum(values) / len(values)


def _undistort(x, y):
    dx, dy = x - 0.45, y
    observed = math.hypot(dx, dy)
    if observed == 0.0:
        return x, y

    radius = observed
    for _ in range(12):
        radius -= (
            radius + _RADIAL_K * radius ** 3 - observed
        ) / (1.0 + 3.0 * _RADIAL_K * radius ** 2)

    scale = radius / observed
    return 0.45 + dx * scale, dy * scale


def run(api):
    """Camera-centred coverage search with settled, shallow contact tests."""
    calls = 0
    target_z = None
    seated = False

    class BudgetExhausted(Exception):
        pass

    class InvalidReading(Exception):
        pass

    def call(function, *args):
        nonlocal calls
        # Always leave one call available for vertical withdrawal.
        if calls >= 299:
            raise BudgetExhausted()
        calls += 1
        return function(*args)

    def move(x, y, z, yaw):
        nonlocal target_z
        call(api.move_to, (x, y, z), yaw)
        target_z = z

    def down(mm):
        nonlocal target_z
        force = call(api.step_down, mm)
        target_z -= mm * 0.001
        if not math.isfinite(force):
            raise InvalidReading()
        return abs(force)

    def withdraw():
        if target_z is not None and target_z < _SAFE_Z:
            down(1000.0 * (target_z - _SAFE_Z))

    def confirm():
        nonlocal seated
        seated = bool(call(api.done))
        return seated

    def insert():
        # Do not equate commanded depth with achieved depth.
        while target_z > _FINAL_Z + 1.0e-9:
            increment = 0.5 if target_z > _TOP - 0.002 else 1.0
            increment = min(
                increment, 1000.0 * (target_z - _FINAL_Z)
            )
            force = down(increment)

            if force > _CONTACT_LIMIT:
                return False

            if target_z <= _TOP - 0.031:
                if confirm():
                    return True

            # At first engagement, stop rather than accumulating target
            # penetration against a blocked peg.
            if target_z > _TOP - 0.003 and force > _FREE_FORCE:
                force = down(0.0)
                if force > _FREE_FORCE:
                    return False

        for _ in range(2):
            if down(0.0) > _CONTACT_LIMIT:
                return False
            if confirm():
                return True
        return False

    def probe(x, y, yaw):
        # Includes the ordinary probe, a complete insertion, and withdrawal.
        if calls + 7 + _INSERT_RESERVE > 299:
            raise BudgetExhausted()

        withdraw()

        # All lateral/yaw changes happen above the block. Descend vertically
        # at the candidate pose before using force-returning motion.
        move(x, y, _SAFE_Z, yaw)
        move(x, y, _HOVER_Z, yaw)

        force = down(1.5)  # Only 0.5 mm commanded penetration.
        if force > _CONTACT_LIMIT:
            return False

        # A transient zero force during servo settling is not clearance.
        force = down(0.0)
        if force > _FREE_FORCE:
            return False

        return insert()

    try:
        readings = []
        for _ in range(65):
            (x, y, z), raw_yaw = call(api.hole_pose)
            if not all(math.isfinite(v) for v in (x, y, z, raw_yaw)):
                raise InvalidReading()

            readings.append((x, y, raw_yaw))

            # Repeated identical estimates contain no additional information.
            if len(readings) == 5 and all(
                reading == readings[0] for reading in readings
            ):
                break

        mx = _trimmed_mean([r[0] for r in readings])
        my = _trimmed_mean([r[1] for r in readings])
        cx, cy = _undistort(mx, my)

        # Undo the empirical angle scale before applying square symmetry.
        angles = [_wrap(r[2] / _YAW_SCALE) for r in readings]
        ss = sum(math.sin(math.radians(4.0 * a)) for a in angles)
        cc = sum(math.cos(math.radians(4.0 * a)) for a in angles)
        if math.hypot(ss, cc) < 1.0e-8:
            raise InvalidReading()

        reference = math.degrees(math.atan2(ss, cc)) / 4.0
        yaw = _wrap(
            reference
            + _trimmed_mean([_wrap(a - reference) for a in angles])
        )

        theta = math.radians(yaw)
        cosine, sine = math.cos(theta), math.sin(theta)

        # Keep the camera estimate as the anchor. Scalar normal force on
        # the block is not a reliable lateral or angular error gradient.
        spacing = 0.00060
        visited = set()

        def attempt(i, j, rotation):
            key = (i, j, rotation)
            if key in visited:
                return False
            visited.add(key)

            u, v = spacing * i, spacing * j
            x = cx + cosine * u - sine * v
            y = cy + sine * u + cosine * v
            return probe(x, y, yaw + rotation)

        # First cover translation at the averaged camera orientation.
        # A 0.6 mm grid leaves at most 0.3 mm error along either grid axis.
        central = [
            (i * i + j * j, i, j)
            for i in range(-2, 3)
            for j in range(-2, 3)
        ]
        central.sort()

        for _, i, j in central:
            if attempt(i, j, 0.0):
                return

        # Then interleave wider coverage with modest orientation corrections.
        # These are explicit hypotheses, not moves toward lower block force.
        remaining = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                radius_squared = i * i + j * j
                for rotation in (0.0, -1.0, 1.0, -2.0, 2.0):
                    if (i, j, rotation) in visited:
                        continue
                    priority = radius_squared + 10.0 * rotation * rotation
                    remaining.append((priority, i, j, rotation))

        remaining.sort()
        for _, i, j, rotation in remaining:
            if attempt(i, j, rotation):
                return

    except (BudgetExhausted, InvalidReading):
        pass
    finally:
        if not seated and target_z is not None and target_z < _SAFE_Z:
            if calls < 300:
                calls += 1
                api.step_down(1000.0 * (target_z - _SAFE_Z))
