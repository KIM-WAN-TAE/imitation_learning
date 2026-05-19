from __future__ import annotations


def clamp(x: float, lo: float, hi: float) -> float:
    """값을 [lo, hi] 범위로 제한합니다."""
    return max(lo, min(hi, float(x)))


def norm100_to_joint_raw_asymmetric(
    u: float,
    joint_min: float,
    joint_home: float,
    joint_max: float,
) -> float:
    """
    정규화된 joint-space 값(-100~100)을 joint raw 값으로 변환합니다.

    의미:
        u = -100 -> joint_min
        u = 0    -> joint_home
        u = 100  -> joint_max

    home 기준으로 +방향 범위와 -방향 범위가 달라도 처리할 수 있습니다.
    """
    u = clamp(u, -100.0, 100.0)

    if u >= 0.0:
        denom = joint_max - joint_home
        return joint_home + (u / 100.0) * denom

    denom = joint_home - joint_min
    return joint_home + (u / 100.0) * denom


def joint_raw_to_norm100_asymmetric(
    joint_raw: float,
    joint_min: float,
    joint_home: float,
    joint_max: float,
) -> float:
    """
    joint raw 값을 정규화된 joint-space 값(-100~100)으로 변환합니다.

    의미:
        joint_min  -> -100
        joint_home -> 0
        joint_max  -> 100
    """
    joint_raw = float(joint_raw)

    if joint_raw >= joint_home:
        denom = joint_max - joint_home
    else:
        denom = joint_home - joint_min

    if abs(denom) < 1e-9:
        return 0.0

    u = 100.0 * (joint_raw - joint_home) / denom
    return clamp(u, -100.0, 100.0)


def joint_raw_to_motor_raw(
    joint_raw: float,
    joint_home: float,
    motor_home: float,
    gear_ratio: float,
    direction: float,
) -> float:
    """
    joint raw 목표값을 motor raw 목표값으로 변환합니다.

    핵심:
        절대 위치값에 gear ratio를 곱하면 안 됩니다.
        home 기준 변위에만 gear ratio와 direction을 적용해야 합니다.

    수식:
        motor_raw = motor_home + (joint_raw - joint_home) * gear_ratio * direction
    """
    scale = float(gear_ratio) * float(direction)

    if abs(scale) < 1e-9:
        raise ValueError("gear_ratio * direction must not be zero")

    joint_delta = float(joint_raw) - float(joint_home)
    motor_delta = joint_delta * scale

    return float(motor_home) + motor_delta


def motor_raw_to_joint_raw(
    motor_raw: float,
    joint_home: float,
    motor_home: float,
    gear_ratio: float,
    direction: float,
) -> float:
    """
    motor raw 현재값을 joint raw 값으로 변환합니다.

    joint_raw_to_motor_raw()의 역변환입니다.

    수식:
        joint_raw = joint_home + (motor_raw - motor_home) / (gear_ratio * direction)
    """
    scale = float(gear_ratio) * float(direction)

    if abs(scale) < 1e-9:
        raise ValueError("gear_ratio * direction must not be zero")

    motor_delta = float(motor_raw) - float(motor_home)
    joint_delta = motor_delta / scale

    return float(joint_home) + joint_delta