import math

MAX_CALLS = 294            # margin under the 300-call limit

# Camera model fitted from the failure log: the estimate is pushed radially away
# from the cell centre by k*r^3, and yaw is inflated by k_yaw*yaw^3 (wrapped yaw).
CAM_CX, CAM_CY = 0.45, 0.0
K_RADIAL = 1.68            # 1/m^2
K_RADIAL_SIGMA = 0.12      # assumed uncertainty of K_RADIAL
K_YAW = 3.1e-5             # 1/deg^2

HOVER_MM = 2.0             # hover height above the block for lateral moves
PITCH_MM = 0.65            # search pitch (hole frame)
N_MIN, N_MAX = 12, 80      # camera samples
INSERT_MM = 36.0           # commanded insertion depth


class _Budget(Exception):
    pass


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _trimmed_mean(vals):
    v = sorted(vals)
    n = len(v)
    cut = int(0.1 * n)
    v = v[cut:n - cut] if n - 2 * cut > 0 else v
    return sum(v) / len(v)


def _solve_cubic(e, k):
    """Solve t + k*t^3 = e for t."""
    t = e
    for _ in range(8):
        t -= (t + k * t * t * t - e) / (1.0 + 3.0 * k * t * t)
    return t


def run(api):
    calls = [0]

    def use():
        calls[0] += 1
        if calls[0] > MAX_CALLS:
            raise _Budget()

    def hole_pose():
        use()
        return api.hole_pose()

    def move_to(p, yaw):
        use()
        return api.move_to(p, yaw)

    def step_down(mm):
        use()
        f = api.step_down(mm)
        return 0.0 if f is None else float(f)

    def done():
        use()
        return bool(api.done())

    try:
        # ---------- 1. camera: many samples, robust mean, undo distortion ----------
        xs, ys, zs, yaws = [], [], [], []

        def sample(n):
            for _ in range(n):
                (x, y, z), yaw = hole_pose()
                xs.append(x)
                ys.append(y)
                zs.append(z)
                yaws.append(yaw)

        def spread_mm():
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
            vy = sum((y - my) ** 2 for y in ys) / (n - 1)
            return 1000.0 * math.sqrt(0.5 * (vx + vy))

        sample(N_MIN)
        s = spread_mm()
        n_total = int(_clamp(2.8 * s * s + 1.0, N_MIN, N_MAX))
        if n_total > len(xs):
            sample(n_total - len(xs))
        n = len(xs)

        mx, my, hz = _trimmed_mean(xs), _trimmed_mean(ys), _trimmed_mean(zs)
        ex, ey = mx - CAM_CX, my - CAM_CY
        rho_e = math.hypot(ex, ey)
        rho = _solve_cubic(rho_e, K_RADIAL)
        if rho_e > 1e-9:
            ux, uy = ex / rho_e, ey / rho_e
        else:
            ux, uy = 1.0, 0.0
        vx_, vy_ = -uy, ux
        hx, hy = CAM_CX + ux * rho, CAM_CY + uy * rho

        # yaw: undistort each sample, then average on the 90-degree circle
        sc = ss = 0.0
        for e in yaws:
            t = _solve_cubic(e, K_YAW)
            sc += math.cos(math.radians(4.0 * t))
            ss += math.sin(math.radians(4.0 * t))
        hyaw = math.degrees(math.atan2(ss, sc)) / 4.0

        # uncertainty of the position (mm), radial / tangential
        sr = st = 0.0
        for x, y in zip(xs, ys):
            dx, dy = x - mx, y - my
            sr += (dx * ux + dy * uy) ** 2
            st += (dx * vx_ + dy * vy_) ** 2
        sr = 1000.0 * math.sqrt(sr / (n - 1)) / math.sqrt(n) / (1.0 + 3.0 * K_RADIAL * rho * rho)
        st = 1000.0 * math.sqrt(st / (n - 1)) / math.sqrt(n) / (1.0 + K_RADIAL * rho * rho)
        sig_r = math.sqrt(sr * sr + (1000.0 * K_RADIAL_SIGMA * rho ** 3) ** 2 + 0.35 ** 2)
        sig_t = math.sqrt(st * st + 0.35 ** 2)

        # ---------- 2. force model (arm is much stiffer close to the base) ----------
        R = _clamp(math.hypot(hx, hy), 0.22, 0.85)
        R2 = R * R
        K = 2.8 / R2                                    # N per mm of blocked travel
        p1 = min(2.5, 0.7 + 14.4 * R2)                  # probe depth below surface (mm)
        f_model = (5.0 / R2) * (p1 - 0.7) / 1.8         # expected force on solid surface
        thr = _clamp(0.3 * f_model, 2.5, 8.0)           # below this: nothing under the peg
        dv = min(1.0, 7.2 * R2)                         # confirmation step (mm)
        s_ins = _clamp(25.0 * R2, 1.2, 5.0)             # insertion step (mm)
        s_first = _clamp(10.0 * R2, 0.5, 2.0)           # cautious first insertion steps

        def jam_limit(stp):
            return max(thr + 2.0, _clamp(0.35 * K * stp, 6.0, 30.0))

        # ---------- 3. candidate list: elliptical spiral in the hole frame ----------
        c, sn = math.cos(math.radians(hyaw)), math.sin(math.radians(hyaw))
        delta = PITCH_MM / min(sig_r, sig_t)
        cands = []
        M = 14
        for i in range(-M, M + 1):
            for j in range(-M, M + 1):
                dx, dy = i * PITCH_MM, j * PITCH_MM
                wx = c * dx - sn * dy
                wy = sn * dx + c * dy
                a = (wx * ux + wy * uy) / sig_r
                b = (wx * vx_ + wy * vy_) / sig_t
                q = math.sqrt(a * a + b * b)
                if q > 6.0:
                    continue
                ring = int(q / delta + 0.5)
                cands.append((ring, math.atan2(b, a), wx * 1e-3, wy * 1e-3))
        cands.sort()

        # ---------- 4. approach ----------
        move_to((hx, hy, hz + 0.10), hyaw)
        move_to((hx, hy, hz + 0.015), hyaw)

        def insert(depth):
            """Peg is in the hole at commanded depth (mm). Returns True when seated."""
            jam = False
            k_step = 0
            while depth < INSERT_MM:
                stp = min(s_first if k_step < 2 else s_ins, INSERT_MM - depth)
                k_step += 1
                f = step_down(stp)
                depth += stp
                if f > jam_limit(stp):
                    jam = True
                    break
            if not jam:
                if done():
                    return True
                for _ in range(4):
                    f = step_down(2.0)
                    depth += 2.0
                    if done():
                        return True
                    if f > jam_limit(2.0):
                        break
            elif depth >= 28.0 and done():
                return True
            step_down(-(depth + HOVER_MM))
            return False

        # ---------- 5. search: one move + one touch per candidate ----------
        first = True
        for _, _, wx, wy in cands:
            target = (hx + wx, hy + wy, hz + HOVER_MM * 1e-3)
            move_to(target, hyaw)
            if first:
                move_to(target, hyaw)      # let the long approach move converge
                first = False
            f = step_down(HOVER_MM + p1)
            if f >= thr:
                continue                   # resting on the block surface
            f = step_down(dv)              # confirm: a little deeper, more settle time
            if f >= thr:
                continue
            if insert(p1 + dv):
                return
    except _Budget:
        return
