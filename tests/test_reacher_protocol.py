import csv
import copy
import importlib
import io
import math
from dataclasses import FrozenInstanceError, astuple

import numpy as np
import pytest


def _protocol():
    return importlib.import_module("sdob.examples.reacher.reacher_protocol")


def test_seed_streams_and_protocol_constants_are_exact():
    protocol = _protocol()

    streams = protocol.seed_streams(1000)

    assert streams == protocol.ReacherSeedStreams(
        task_seed=1000,
        planner_seed=201000,
        sdob_observer_seed=301000,
        tube_observer_seed=311000,
        robust_observer_seed=321000,
        tube_feedback_seed=401000,
        robust_feedback_seed=501000,
    )
    assert protocol.ELIGIBILITY_THRESHOLD_RAD == 0.025
    assert protocol.ELIGIBILITY_TAIL_STEPS == 20
    assert protocol.EXPECTED_STEPS == 225
    assert protocol.CANDIDATE_TASK_SEEDS == tuple(range(1000, 1100))
    with pytest.raises(FrozenInstanceError):
        streams.task_seed = 0


def test_seed_streams_are_unique_and_advance_together():
    protocol = _protocol()
    streams = astuple(protocol.seed_streams(1000))
    next_streams = astuple(protocol.seed_streams(1001))

    assert len(set(streams)) == len(streams)
    assert next_streams == tuple(seed + 1 for seed in streams)


@pytest.mark.parametrize("task_seed", [1000, np.int32(1000), np.int64(1000)])
def test_seed_streams_accepts_integer_types_and_normalizes_to_python_int(task_seed):
    protocol = _protocol()

    streams = protocol.seed_streams(task_seed)

    assert streams == protocol.seed_streams(1000)
    assert all(type(seed) is int for seed in astuple(streams))


@pytest.mark.parametrize(
    "task_seed",
    [
        True,
        np.bool_(True),
        1000.0,
        "1000",
        np.array(1000),
        np.array([1000]),
        np.array(True),
        np.array(False),
        None,
    ],
)
def test_seed_streams_rejects_non_integer_inputs(task_seed):
    protocol = _protocol()

    with pytest.raises(ValueError):
        protocol.seed_streams(task_seed)


def test_reacher_final_error_is_raw_max_component_difference_without_wrapping():
    protocol = _protocol()

    error = protocol.reacher_final_error(
        [math.pi - 0.01, 0.2],
        [-math.pi + 0.01, 0.0],
    )

    assert error == pytest.approx(2.0 * math.pi - 0.02)
    assert error > 6.0


def test_reacher_final_error_uses_joint_one_when_it_has_the_largest_error():
    protocol = _protocol()

    assert protocol.reacher_final_error([0.1, -0.4], [0.0, 0.0]) == pytest.approx(
        0.4
    )


@pytest.mark.parametrize(
    ("qpos", "target_qpos"),
    [
        ([0.0], [0.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0]),
        ([[0.0, 0.0]], [0.0, 0.0]),
        ([0.0, np.nan], [0.0, 0.0]),
        ([0.0, np.inf], [0.0, 0.0]),
        (["bad", 0.0], [0.0, 0.0]),
        ([True, 0.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, np.bool_(False)]),
        ([np.array(True), 0.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, np.array(False)]),
        ([0.0, 0.0], [0.0]),
        ([0.0, 0.0], [0.0, np.nan]),
        ([0.0, 0.0], None),
    ],
)
def test_reacher_final_error_rejects_malformed_or_nonfinite_vectors(qpos, target_qpos):
    protocol = _protocol()

    with pytest.raises(ValueError):
        protocol.reacher_final_error(qpos, target_qpos)


def test_joint_boundary_distance_uses_nearest_raw_pi_boundary():
    protocol = _protocol()

    assert protocol.joint_boundary_distance(
        [math.pi - 0.1, -math.pi + 0.2]
    ) == pytest.approx(0.1)
    assert protocol.joint_boundary_distance([math.pi + 0.1, 0.0]) == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "qpos",
    [
        [],
        [0.0],
        [0.0, 0.0, 0.0],
        [[0.0, 0.0]],
        [0.0, np.nan],
        [np.inf, 0.0],
        ["bad", 0.0],
        [True, 0.0],
        [0.0, np.bool_(False)],
        [np.array(True), 0.0],
        [0.0, np.array(False)],
        None,
    ],
)
def test_joint_boundary_distance_rejects_malformed_or_nonfinite_vectors(qpos):
    protocol = _protocol()

    with pytest.raises(ValueError):
        protocol.joint_boundary_distance(qpos)


def _valid_episode(tail_errors=None):
    if tail_errors is None:
        tail_errors = [0.02] * 20
    assert len(tail_errors) == 20
    per_step_errors = [0.02] * 205 + list(tail_errors)
    summary = {
        "status": "ok",
        "final_error": tail_errors[-1],
        "tail_error": math.fsum(tail_errors) / 20,
        "target_qpos_0": 0.0,
        "target_qpos_1": 0.0,
    }
    trajectory_rows = [
        {
            "step": step,
            "error": error,
            "qpos_0": error,
            "qpos_1": 0.0,
            "target_qpos_0": 0.0,
            "target_qpos_1": 0.0,
            "u_applied_0": 0.1,
            "u_applied_1": -0.1,
            "planning_time_ms": 1.0,
        }
        for step, error in enumerate(per_step_errors)
    ]
    return summary, trajectory_rows


@pytest.mark.parametrize(
    ("tail_error", "expected"),
    [(0.025, True), (0.0250001, False)],
)
def test_nominal_episode_eligibility_uses_inclusive_raw_tail_mean_threshold(
    tail_error, expected
):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode([tail_error] * 20)

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is expected


def test_nominal_episode_eligibility_can_pass_with_large_final_step():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode([0.0] * 19 + [0.4])

    assert summary["tail_error"] == pytest.approx(0.02)
    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is True


def test_nominal_episode_eligibility_can_fail_with_tiny_final_step():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode([0.03] * 19 + [0.0])

    assert summary["tail_error"] > 0.025
    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_eligibility_accepts_csv_dict_reader_rows():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    csv_file = io.StringIO()
    writer = csv.DictWriter(csv_file, fieldnames=trajectory_rows[0].keys())
    writer.writeheader()
    writer.writerows(trajectory_rows)
    csv_file.seek(0)

    round_tripped_rows = list(csv.DictReader(csv_file))

    assert [row["step"] for row in round_tripped_rows] == [
        str(step) for step in range(225)
    ]
    assert protocol.eligible_nominal_episode(summary, round_tripped_rows) is True


def test_nominal_episode_eligibility_requires_exactly_225_rows():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()

    assert protocol.eligible_nominal_episode(summary, trajectory_rows[:-1]) is False


@pytest.mark.parametrize("status", ["failed", "OK", "ok ", None])
def test_nominal_episode_eligibility_requires_exact_ok_status(status):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    summary["status"] = status

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_eligibility_rejects_non_scalar_status_without_raising():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    summary["status"] = np.array(["ok", "ok"])

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize(
    "field",
    [
        "step",
        "error",
        "qpos_0",
        "qpos_1",
        "target_qpos_0",
        "target_qpos_1",
        "u_applied_0",
        "u_applied_1",
        "planning_time_ms",
    ],
)
def test_nominal_episode_eligibility_rejects_nan_in_any_required_row_field(field):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[100][field] = np.nan

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize(
    "field",
    [
        "step",
        "error",
        "qpos_0",
        "qpos_1",
        "target_qpos_0",
        "target_qpos_1",
        "u_applied_0",
        "u_applied_1",
        "planning_time_ms",
    ],
)
def test_nominal_episode_eligibility_rejects_zero_dimensional_boolean_row_fields(
    field,
):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    row_index = 1 if field == "step" else 100
    trajectory_rows[row_index][field] = np.array(True)

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize("step", [205, 214, 224])
def test_nominal_episode_eligibility_rejects_tail_row_error_mismatch(step):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[step]["error"] += 1e-6

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize("row_index", [100, 205, 224])
def test_nominal_episode_rejects_row_target_drift_even_when_row_error_is_zero(
    row_index,
):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode([0.0] * 20)
    trajectory_rows[row_index]["qpos_0"] = 0.5
    trajectory_rows[row_index]["target_qpos_0"] = 0.5
    trajectory_rows[row_index]["error"] = 0.0

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize("field", ["target_qpos_0", "target_qpos_1"])
@pytest.mark.parametrize("delta", [-1e-12, 1e-12])
def test_nominal_episode_target_binding_accepts_exact_absolute_tolerance(
    field,
    delta,
):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[100][field] = delta

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is True


@pytest.mark.parametrize("field", ["target_qpos_0", "target_qpos_1"])
@pytest.mark.parametrize(
    "delta",
    [
        np.nextafter(-1e-12, -np.inf),
        np.nextafter(1e-12, np.inf),
    ],
)
def test_nominal_episode_target_binding_rejects_just_over_absolute_tolerance(
    field,
    delta,
):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[100][field] = delta

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_target_binding_uses_zero_relative_tolerance():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode([0.0] * 20)
    summary["target_qpos_0"] = 1e9
    for row in trajectory_rows:
        row["target_qpos_0"] = 1e9
    for row in trajectory_rows[-20:]:
        row["qpos_0"] = 1e9
    trajectory_rows[100]["target_qpos_0"] += 1e-3
    trajectory_rows[100]["qpos_0"] = trajectory_rows[100]["target_qpos_0"]

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_eligibility_rejects_summary_tail_error_mismatch():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    summary["tail_error"] += 1e-6

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize("delta", [-1e-6, 1e-6])
def test_nominal_episode_eligibility_rejects_summary_final_error_mismatch(delta):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    summary["final_error"] += delta

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_eligibility_uses_absolute_1e_12_error_agreement():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[-1]["error"] += 0.5e-12
    summary["final_error"] -= 0.5e-12
    summary["tail_error"] += 0.5e-12

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is True


def test_nominal_episode_eligibility_rejects_tail_row_mismatch_just_over_atol():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[205]["error"] += 1.1e-12

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_eligibility_rejects_tail_summary_mismatch_just_over_atol():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    summary["tail_error"] += 1.1e-12

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_eligibility_rejects_final_summary_mismatch_just_over_atol():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    summary["final_error"] += 1.1e-12

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_eligibility_uses_zero_relative_tolerance_for_large_error():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode([0.4] + [0.0] * 19)
    trajectory_rows[205]["error"] += 1e-9

    assert summary["tail_error"] == pytest.approx(0.02)
    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize(
    "invalid_steps",
    ["duplicate", "out_of_range", "reordered"],
)
def test_nominal_episode_eligibility_rejects_duplicate_or_noncontiguous_steps(
    invalid_steps,
):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    if invalid_steps == "duplicate":
        trajectory_rows[100]["step"] = 99
    elif invalid_steps == "out_of_range":
        trajectory_rows[100]["step"] = 225
    else:
        trajectory_rows[0], trajectory_rows[1] = trajectory_rows[1], trajectory_rows[0]

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize(
    ("row_index", "step"),
    [
        (100, 100.0),
        (100, np.float64(100.0)),
        (1, True),
        (100, np.array(100)),
        (100, np.array([100])),
    ],
)
def test_nominal_episode_eligibility_rejects_non_integer_step_types(row_index, step):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[row_index]["step"] = step

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize(
    ("row_index", "step"),
    [
        (100, " 100"),
        (100, "100 "),
        (100, "+100"),
        (0, "-0"),
        (100, "100.0"),
        (100, "1e2"),
        (10, "010"),
        (0, "00"),
        (100, "not-a-step"),
    ],
)
def test_nominal_episode_eligibility_rejects_noncanonical_step_strings(
    row_index, step
):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[row_index]["step"] = step

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize("step", [np.int32(100), np.int64(100)])
def test_nominal_episode_eligibility_accepts_numpy_integer_steps(step):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[100]["step"] = step

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is True


@pytest.mark.parametrize(
    "field",
    [
        "step",
        "error",
        "qpos_0",
        "qpos_1",
        "target_qpos_0",
        "target_qpos_1",
        "u_applied_0",
        "u_applied_1",
        "planning_time_ms",
    ],
)
def test_nominal_episode_eligibility_rejects_missing_required_row_fields(field):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    del trajectory_rows[100][field]

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_eligibility_rejects_malformed_required_row_values():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[100]["u_applied_0"] = [0.1]

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize(
    "summary",
    [
        None,
        {},
        {"status": "ok"},
        {"status": "ok", "final_error": 0.02},
        {"status": "ok", "final_error": "bad", "tail_error": 0.02},
        {"status": "ok", "final_error": np.nan, "tail_error": 0.02},
        {"status": "ok", "final_error": 0.02, "tail_error": "bad"},
        {"status": "ok", "final_error": 0.02, "tail_error": np.nan},
    ],
)
def test_nominal_episode_eligibility_rejects_malformed_summary(summary):
    protocol = _protocol()
    _, trajectory_rows = _valid_episode()

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize("field", ["target_qpos_0", "target_qpos_1"])
def test_nominal_episode_eligibility_requires_summary_target_fields(field):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    del summary[field]

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_qpos_0", "bad"),
        ("target_qpos_0", np.nan),
        ("target_qpos_1", np.inf),
        ("target_qpos_1", np.array(True)),
    ],
)
def test_nominal_episode_eligibility_rejects_malformed_summary_targets(field, value):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    summary[field] = value

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize(
    ("field", "tail_errors"),
    [
        ("tail_error", [1.0] * 20),
        ("final_error", [0.0] * 19 + [1.0]),
    ],
)
def test_nominal_episode_eligibility_rejects_zero_dimensional_boolean_summary_fields(
    field, tail_errors, monkeypatch
):
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode(tail_errors)
    summary[field] = np.array(True)
    monkeypatch.setattr(protocol, "ELIGIBILITY_THRESHOLD_RAD", 2.0)

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


@pytest.mark.parametrize("trajectory_rows", [None, {}, "not rows"])
def test_nominal_episode_eligibility_rejects_malformed_trajectory_container(
    trajectory_rows,
):
    protocol = _protocol()
    summary, _ = _valid_episode()

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def test_nominal_episode_eligibility_rejects_non_mapping_trajectory_row():
    protocol = _protocol()
    summary, trajectory_rows = _valid_episode()
    trajectory_rows[100] = None

    assert protocol.eligible_nominal_episode(summary, trajectory_rows) is False


def _manifest_candidates():
    protocol = _protocol()
    candidates = []
    for task_seed in protocol.CANDIDATE_TASK_SEEDS:
        index = task_seed - protocol.CANDIDATE_TASK_SEEDS[0]
        initial_qpos = (index / 100.0, -0.25)
        target_qpos = (-index / 200.0, 0.4)
        eligible = task_seed < 1025
        tail_mean_error = 0.01 if eligible else 0.03
        candidates.append(
            {
                "task_seed": task_seed,
                "status": "ok",
                "row_count": protocol.EXPECTED_STEPS,
                "tail_mean_error": tail_mean_error,
                "final_error": tail_mean_error + 0.001,
                "eligible": eligible,
                "initial_qpos_0": initial_qpos[0],
                "initial_qpos_1": initial_qpos[1],
                "target_qpos_0": target_qpos[0],
                "target_qpos_1": target_qpos[1],
                "b_initial": protocol.joint_boundary_distance(initial_qpos),
                "b_target": protocol.joint_boundary_distance(target_qpos),
                "planner_seed": protocol.seed_streams(task_seed).planner_seed,
                "goal_image_sha256": f"{task_seed:064x}",
            }
        )
    return candidates


def _csv_dict_reader_rows(protocol, candidates):
    csv_file = io.StringIO()
    writer = csv.DictWriter(csv_file, fieldnames=protocol.SCREENING_RESULT_FIELDS)
    writer.writeheader()
    writer.writerows(candidates)
    csv_file.seek(0)
    return list(csv.DictReader(csv_file))


def test_screening_results_csv_deserializer_round_trips_typed_candidate_rows():
    protocol = _protocol()
    candidates = _manifest_candidates()

    rows = _csv_dict_reader_rows(protocol, candidates)
    expected = protocol.validate_screening_candidate_rows(candidates)
    actual = protocol.deserialize_screening_result_rows(rows)

    assert actual == expected == tuple(candidates)
    assert [
        {field: type(value) for field, value in row.items()} for row in actual
    ] == [
        {field: type(value) for field, value in row.items()} for row in expected
    ]


def test_screening_results_csv_deserializer_accepts_native_booleans():
    protocol = _protocol()
    candidates = _manifest_candidates()
    rows = _csv_dict_reader_rows(protocol, candidates)
    rows[0]["eligible"] = True
    rows[-1]["eligible"] = False

    assert protocol.deserialize_screening_result_rows(rows) == tuple(candidates)


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("task_seed", "01000"),
        ("task_seed", "1000 "),
        ("row_count", "225.0"),
        ("row_count", "0225"),
        ("planner_seed", "+201000"),
        ("tail_mean_error", "0.010"),
        ("final_error", "NaN"),
        ("final_error", "inf"),
        ("initial_qpos_0", " 0.0"),
        ("eligible", "true"),
        ("eligible", "False "),
        ("task_seed", 1000),
        ("tail_mean_error", 0.01),
    ],
)
def test_screening_results_csv_deserializer_rejects_noncanonical_scalars(
    field,
    malformed_value,
):
    protocol = _protocol()
    rows = _csv_dict_reader_rows(protocol, _manifest_candidates())
    rows[0][field] = malformed_value

    with pytest.raises(ValueError):
        protocol.deserialize_screening_result_rows(rows)


@pytest.mark.parametrize("schema_change", ["missing", "extra"])
def test_screening_results_csv_deserializer_rejects_wrong_fields(schema_change):
    protocol = _protocol()
    rows = _csv_dict_reader_rows(protocol, _manifest_candidates())
    if schema_change == "missing":
        del rows[0]["final_error"]
    else:
        rows[0]["unexpected"] = "value"

    with pytest.raises(ValueError, match="fields"):
        protocol.deserialize_screening_result_rows(rows)


def _reseal_manifest(protocol, manifest):
    manifest["manifest_sha256"] = protocol.selected_seed_manifest_sha256(manifest)
    return manifest


def test_selected_seed_manifest_selects_first_twenty_eligible_seeds_and_validates():
    protocol = _protocol()

    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())

    assert manifest["candidate_task_seeds"] == list(range(1000, 1100))
    assert manifest["eligibility_threshold_rad"] == 0.025
    assert manifest["eligibility_tail_steps"] == 20
    assert manifest["expected_steps"] == 225
    assert manifest["selection_count"] == 20
    assert manifest["selected_task_seeds"] == list(range(1000, 1020))
    assert manifest["rejected_task_seeds"] == list(range(1020, 1100))
    assert protocol.validate_selected_seed_manifest(manifest) == tuple(
        range(1000, 1020)
    )
    assert manifest["manifest_sha256"] == protocol.selected_seed_manifest_sha256(
        manifest
    )


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_manifest_rejects_noninteger_schema_version_even_when_equal_to_one(
    schema_version,
):
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    manifest["schema_version"] = schema_version
    _reseal_manifest(protocol, manifest)

    with pytest.raises(ValueError, match="schema_version"):
        protocol.validate_selected_seed_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "string_value"),
    [
        ("task_seed", "1000"),
        ("tail_mean_error", "0.01"),
        ("eligible", "True"),
    ],
)
def test_manifest_candidate_validation_does_not_accept_csv_scalar_strings(
    field,
    string_value,
):
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    manifest["candidates"][0][field] = string_value
    _reseal_manifest(protocol, manifest)

    with pytest.raises(ValueError):
        protocol.validate_selected_seed_manifest(manifest)


@pytest.mark.parametrize(
    "selected_task_seeds",
    [
        [1000, 1000, *range(1002, 1020)],
        [1001, 1000, *range(1002, 1020)],
        list(range(1000, 1019)),
        list(range(1001, 1021)),
    ],
    ids=["duplicate", "unsorted", "wrong-count", "not-first-twenty"],
)
def test_manifest_rejects_invalid_selected_seed_lists(selected_task_seeds):
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    manifest["selected_task_seeds"] = selected_task_seeds
    manifest["rejected_task_seeds"] = [
        seed
        for seed in protocol.CANDIDATE_TASK_SEEDS
        if seed not in selected_task_seeds
    ]
    _reseal_manifest(protocol, manifest)

    with pytest.raises(ValueError):
        protocol.validate_selected_seed_manifest(manifest)


def test_manifest_rejects_wrong_candidate_block_even_with_valid_hash():
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    manifest["candidate_task_seeds"][0] = 999
    _reseal_manifest(protocol, manifest)

    with pytest.raises(ValueError, match="candidate"):
        protocol.validate_selected_seed_manifest(manifest)


def test_manifest_rejects_wrong_threshold_even_with_valid_hash():
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    manifest["eligibility_threshold_rad"] = 0.0250001
    _reseal_manifest(protocol, manifest)

    with pytest.raises(ValueError, match="threshold"):
        protocol.validate_selected_seed_manifest(manifest)


def test_manifest_rejects_selected_seed_missing_from_candidate_rows():
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    manifest["candidates"] = manifest["candidates"][1:]
    _reseal_manifest(protocol, manifest)

    with pytest.raises(ValueError, match="candidate"):
        protocol.validate_selected_seed_manifest(manifest)


def test_manifest_rejects_noneligible_selected_candidate():
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    manifest["candidates"][0]["eligible"] = False
    manifest["candidates"][0]["tail_mean_error"] = 0.03
    _reseal_manifest(protocol, manifest)

    with pytest.raises(ValueError, match="eligible"):
        protocol.validate_selected_seed_manifest(manifest)


@pytest.mark.parametrize(
    "missing_field",
    [
        "status",
        "row_count",
        "tail_mean_error",
        "final_error",
        "eligible",
        "initial_qpos_0",
        "initial_qpos_1",
        "target_qpos_0",
        "target_qpos_1",
        "b_initial",
        "b_target",
        "planner_seed",
        "goal_image_sha256",
    ],
)
def test_manifest_rejects_missing_candidate_metadata(missing_field):
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    del manifest["candidates"][0][missing_field]
    _reseal_manifest(protocol, manifest)

    with pytest.raises(ValueError, match="candidate"):
        protocol.validate_selected_seed_manifest(manifest)


@pytest.mark.parametrize("boundary_field", ["b_initial", "b_target"])
def test_manifest_rejects_wrong_boundary_distance(boundary_field):
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    manifest["candidates"][0][boundary_field] += 1.1e-12
    _reseal_manifest(protocol, manifest)

    with pytest.raises(ValueError, match="boundary"):
        protocol.validate_selected_seed_manifest(manifest)


def test_manifest_rejects_payload_tampering_without_recomputed_hash():
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    original_hash = manifest["manifest_sha256"]
    manifest["candidates"][0]["final_error"] += 0.001

    assert manifest["manifest_sha256"] == original_hash
    with pytest.raises(ValueError, match="hash"):
        protocol.validate_selected_seed_manifest(manifest)


def test_manifest_rejects_modified_manifest_hash():
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    manifest["manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="hash"):
        protocol.validate_selected_seed_manifest(manifest)


def test_manifest_validation_does_not_mutate_input():
    protocol = _protocol()
    manifest = protocol.build_selected_seed_manifest(_manifest_candidates())
    original = copy.deepcopy(manifest)

    protocol.validate_selected_seed_manifest(manifest)

    assert manifest == original
