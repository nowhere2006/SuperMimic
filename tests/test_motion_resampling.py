import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from supermimic_study.data.motion_conversion import (
    PREPROCESS_VERSION,
    _quat_slerp_wxyz,
    resample_motion_arrays,
)


def _rotation_error(a, b):
    """Sign-invariant geodesic error for wxyz quaternion pairs."""
    dot = np.abs(np.sum(a * b, axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(dot)


def test_slerp_matches_scipy_and_preserves_endpoints():
    starts = Rotation.from_euler("xyz", [[0.3, -0.2, 0.1], [-1.0, 0.4, 0.2]])
    ends = Rotation.from_euler("xyz", [[-0.5, 0.1, 1.2], [0.6, -0.7, -0.4]])
    fractions = np.array([0.0, 0.17, 0.5, 0.91, 1.0])
    for start, end in zip(starts, ends):
        # SciPy uses xyzw; helper uses wxyz.
        q0 = start.as_quat()[[3, 0, 1, 2]]
        q1 = end.as_quat()[[3, 0, 1, 2]]
        actual = _quat_slerp_wxyz(np.broadcast_to(q0, (len(fractions), 4)),
                                  np.broadcast_to(q1, (len(fractions), 4)), fractions)
        expected_xyzw = Slerp([0.0, 1.0], Rotation.from_quat([start.as_quat(), end.as_quat()]))(fractions).as_quat()
        expected = expected_xyzw[:, [3, 0, 1, 2]]
        assert np.max(_rotation_error(actual, expected)) < 2e-7
        assert np.max(_rotation_error(actual[[0, -1]], np.stack((q0, q1)))) < 2e-7


def test_antipodal_quaternions_follow_same_shortest_rotation():
    q = Rotation.from_euler("zyx", [0.4, -0.2, 0.7]).as_quat()[[3, 0, 1, 2]]
    result = _quat_slerp_wxyz(np.broadcast_to(q, (5, 4)), np.broadcast_to(-q, (5, 4)),
                              np.linspace(0.0, 1.0, 5))
    assert np.max(_rotation_error(result, np.broadcast_to(q, (5, 4)))) < 2e-7
    np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1.0, atol=1e-14)


def test_near_constant_rotation_is_interpolated_not_frozen():
    angle = 1e-5
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    result = _quat_slerp_wxyz(np.broadcast_to(q0, (3, 4)), np.broadcast_to(q1, (3, 4)),
                              np.array([0.0, 0.5, 1.0]))
    expected_mid = np.array([np.cos(angle / 4), 0.0, 0.0, np.sin(angle / 4)])
    assert _rotation_error(result[1], expected_mid) < 2e-8
    assert _rotation_error(result[1], q0) > 4.9e-6


def test_resampling_uses_float64_grid_and_constant_world_angular_velocity():
    input_fps, output_fps, frames, speed = 30, 50, 4, 0.8
    input_times = np.arange(frames, dtype=np.float64) / input_fps
    angles = input_times * speed
    quats = np.stack((np.cos(angles / 2), np.zeros(frames), np.zeros(frames),
                      np.sin(angles / 2)), axis=-1)
    pos = np.stack((input_times, 2 * input_times, -input_times), axis=-1)
    joints = np.stack((3 * input_times, -2 * input_times), axis=-1)
    times, p, q, j, pv, omega, jv = resample_motion_arrays(pos, quats, joints,
                                                             input_fps, output_fps)
    assert PREPROCESS_VERSION == 2
    assert len(times) == int(np.floor((frames - 1) / input_fps * output_fps + 1e-6)) + 1
    assert times.dtype == np.float64
    np.testing.assert_allclose(times, np.arange(len(times), dtype=np.float64) / output_fps)
    np.testing.assert_allclose(p[0], pos[0], atol=1e-14)
    np.testing.assert_allclose(q[0], quats[0], atol=1e-14)
    np.testing.assert_allclose(p[-1], pos[-1], atol=1e-14)
    np.testing.assert_allclose(q[-1], quats[-1], atol=1e-14)
    np.testing.assert_allclose(pv, np.tile([1.0, 2.0, -1.0], (len(times), 1)), atol=1e-12)
    np.testing.assert_allclose(jv, np.tile([3.0, -2.0], (len(times), 1)), atol=1e-12)
    np.testing.assert_allclose(omega[:, 2], speed, atol=2e-12)
    np.testing.assert_allclose(omega[:, :2], 0.0, atol=2e-12)
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-14)


def load_tests(loader, standard_tests, pattern):
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(check,description=check.__name__)
        for name,check in globals().items() if name.startswith("test_") and callable(check))

