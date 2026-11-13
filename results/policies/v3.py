import math


# Smooth empirical camera corrections. Avoid interpolating individual noisy
# calibration observations: doing so can introduce several millimetres of bias.
_RADIAL_K = 2.0
_YAW_SCALE = 1.04

_TOP = 0.050
_SAFE_Z = 0.070
_HOVER_Z = 0.051
_PROBE_LIMIT = 18.0
_INSERT_LIMIT = 12.0
_FREE_FORCE = 0.8


def _wrap(yaw):
    return (yaw + 45.0) % 90.0 - 45.0


def _trimmed_mean(values):
    values = sorted(values)
    trim = len(values) // 5
    middle = values[trim:len(values) - trim] if trim else values
    return sum(middle) / len(middle)


def _undistort(x, y):
    dx, dy = x - 0.45, y
    observed = math.hypot(dx, dy)
    if observed == 0.0:
        return x, y

    radius = observed
    for _ in range(12):
        radius -= (
            radius + _RADIAL_K * radius**3 - observed
        ) / (1.0 + 3.0 * _RADIAL_K * radius**2)

    scale = radius / observed
    return 0.45 + scale * dx, scale * dy


def run(api):
    """Bounded camera estimation, comparable-depth probing, and insertion."""
    calls = 0
    target_z = None
    seated = False
    visited = set()

    class Stop(Exception):
        pass

    def call(function, *args):
        nonlocal calls
        # The last call is reserved for withdrawal.
        if calls >= 299:
            raise Stop()
        calls += 1
        return function(*args)

    def move(x, y, z, yaw):
        nonlocal target_z
        call(api.move_to, (x, y, z), _wrap(yaw))
        target_z = z

    def down(mm):
        nonlocal target_z
        force = call(api.step_down, mm)
        target_z -= mm * 0.001
        if not math.isfinite(force):
            raise Stop()
        return abs(force)

    def withdraw():
        if target_z is not None and target_z < _SAFE_Z:
            down(1000.0 * (target_z - _SAFE_Z))

    def insert():
        nonlocal seated
        # Commanded depth is not measured depth. Only done() confirms seating.
        while target_z > _TOP - 0.0335:
            depth = _TOP - target_z
            increment = 0.5 if depth < 0.003 else 1.0
            increment = min(
                increment, 1000.0 * (target_z - (_TOP - 0.0335))
            )
            force = down(increment)

            if target_z <= _TOP - 0.030:
                seated = bool(call(api.done))
                if seated:
                    return True

            if force > _INSERT_LIMIT:
                return False

        # Allow settling without commanding additional penetration.
        for _ in range(2):
            force = down(0.0)
            seated = bool(call(api.done))
            if seated:
                return True
            if force > _INSERT_LIMIT:
                break
        return False

    def key(pose):
        x, y, yaw = pose
        return (
            round(x, 6),
            round(y, 6),
            round(_wrap(yaw), 3),
        )

    def probe(pose):
        """Return a comparable shallow-contact score; insert if contact-free."""
        # Covers positioning, probing, full insertion, confirmation, withdrawal.
        if calls + 52 > 299:
            raise Stop()

        visited.add(key(pose))
        x, y, yaw = pose

        withdraw()
        move(x, y, _SAFE_Z, yaw)
        move(x, y, _HOVER_Z, yaw)

        force = down(1.0)  # Command the surface, never move_to below it.
        if force > _PROBE_LIMIT:
            withdraw()
            return 100.0 + force

        # Compare candidates at the SAME commanded penetration. Stopping every
        # candidate at its first small force makes the scores uninformative.
        for increment in (0.25, 0.25):
            force = down(increment)
            if force > _PROBE_LIMIT:
                withdraw()
                return 100.0 + force

        score = force
        if force <= _FREE_FORCE:
            if insert():
                return -1.0
            # Shallow clearance alone does not establish successful engagement.
            score = max(score, _INSERT_LIMIT)

        withdraw()
        return score

    try:
        readings = []
        for _ in range(25):
            (x, y, z), raw_yaw = call(api.hole_pose)
            if not all(math.isfinite(v) for v in (x, y, z, raw_yaw)):
                return

            readings.append((x, y, raw_yaw))
            if len(readings) >= 5 and all(
                r == readings[0] for r in readings
            ):
                break

        mx = _trimmed_mean([r[0] for r in readings])
        my = _trimmed_mean([r[1] for r in readings])
        cx, cy = _undistort(mx, my)

        # Correct raw angles before wrapping: the reports include estimates
        # slightly outside the camera's documented square-symmetry interval.
        angles = [_wrap(r[2] / _YAW_SCALE) for r in readings]
        ss = sum(math.sin(math.radians(4.0 * a)) for a in angles)
        cc = sum(math.cos(math.radians(4.0 * a)) for a in angles)
        if math.hypot(ss, cc) < 1.0e-8:
            return
        reference = math.degrees(math.atan2(ss, cc)) / 4.0
        yaw = _wrap(
            reference + _trimmed_mean([_wrap(a - reference) for a in angles])
        )

        # Search translation in the hole's approximate coordinate frame.
        theta = math.radians(yaw)
        cosine, sine = math.cos(theta), math.sin(theta)

        def offset(pose, u, v, rotation):
            return (
                pose[0] + cosine * u - sine * v,
                pose[1] + sine * u + cosine * v,
                _wrap(pose[2] + rotation),
            )

        best = (cx, cy, yaw)
        best_score = probe(best)
        if seated:
            return

        # A force-guided coarse-to-fine search can cover millimetre-scale
        # residual bias without exhaustively enumerating an XY/yaw product.
        for translation, rotation in (
            (0.0032, 2.0),
            (0.0016, 2.0),
            (0.0008, 1.0),
            (0.0004, 1.0),
            (0.0002, 0.5),
        ):
            for direction in (
                (translation, 0.0, 0.0),
                (0.0, translation, 0.0),
                (0.0, 0.0, rotation),
            ):
                origin = best
                for sign in (1.0, -1.0):
                    candidate = offset(
                        origin,
                        sign * direction[0],
                        sign * direction[1],
                        sign * direction[2],
                    )
                    if key(candidate) in visited:
                        continue

                    score = probe(candidate)
                    if seated:
                        return

                    # Do not chase tiny force fluctuations.
                    improvement = max(0.15, 0.025 * best_score)
                    if score < best_score - improvement:
                        best, best_score = candidate, score
                        break

        # Scalar force need not provide a useful gradient on a flat surface.
        # Spend remaining search calls on explicit nearby coverage instead.
        anchor = best
        candidates = []
        spacing = 0.00060
        for i in range(-6, 7):
            for j in range(-6, 7):
                for rotation in (0.0, -1.5, 1.5):
                    if i == 0 and j == 0 and rotation == 0.0:
                        continue
                    priority = i * i + j * j + 4.0 * (rotation / 1.5)**2
                    candidates.append((priority, i, j, rotation))
        candidates.sort()

        for _, i, j, rotation in candidates:
            candidate = offset(
                anchor, spacing * i, spacing * j, rotation
            )
            if key(candidate) in visited:
                continue
            probe(candidate)
            if seated:
                return

    except Stop:
        pass
    finally:
        if not seated and target_z is not None and target_z < _SAFE_Z:
            if calls < 300:
                calls += 1
                api.step_down(1000.0 * (target_z - _SAFE_Z))
