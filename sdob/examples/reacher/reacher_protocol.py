import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


ELIGIBILITY_THRESHOLD_RAD = 0.025
ELIGIBILITY_TAIL_STEPS = 20
EXPECTED_STEPS = 225
CANDIDATE_TASK_SEEDS = tuple(range(1000, 1100))
SELECTED_SEED_COUNT = 20
SCREENING_MANIFEST_SCHEMA_VERSION = 1
SCREENING_SELECTION_RULE = "first_20_eligible_ascending_task_seed"
FORMAL_SUCCESS_THRESHOLD_RAD = 0.05
FORMAL_REACHER_CASES = (
    ("MPPI without disturbance", "WorldModel-MPPI", False),
    ("MPPI", "WorldModel-MPPI", True),
    ("WorldModel+SDOB", "WorldModel+SDOB", True),
    ("Tube-MPPI", "Tube-MPPI", True),
    ("RMPPI", "Robust-MPPI", True),
)

_FORMAL_OBSERVER_CASE_NAMES = frozenset(
    {"WorldModel+SDOB", "Tube-MPPI", "RMPPI"}
)

SCREENING_RESULT_FIELDS = (
    "task_seed",
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
)

_SCREENING_RESULT_INTEGER_FIELDS = (
    "task_seed",
    "row_count",
    "planner_seed",
)

_SCREENING_RESULT_FLOAT_FIELDS = (
    "tail_mean_error",
    "final_error",
    "initial_qpos_0",
    "initial_qpos_1",
    "target_qpos_0",
    "target_qpos_1",
    "b_initial",
    "b_target",
)

_REQUIRED_FINITE_ROW_FIELDS = (
    "qpos_0",
    "qpos_1",
    "target_qpos_0",
    "target_qpos_1",
    "error",
    "u_applied_0",
    "u_applied_1",
    "planning_time_ms",
)


def _is_boolean_scalar(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    return (
        isinstance(value, np.ndarray)
        and value.shape == ()
        and isinstance(value.item(), (bool, np.bool_))
    )


@dataclass(frozen=True)
class ReacherSeedStreams:
    task_seed: int
    planner_seed: int
    sdob_observer_seed: int
    tube_observer_seed: int
    robust_observer_seed: int
    tube_feedback_seed: int
    robust_feedback_seed: int


@dataclass(frozen=True)
class ValidatedNominalEpisode:
    row_count: int
    tail_mean_error: float
    final_error: float
    eligible: bool


@dataclass(frozen=True)
class ValidatedFormalSeed:
    task_seed: int
    summary_row_count: int
    trajectory_row_count: int
    diagnostic_row_count: int


def seed_streams(task_seed: int) -> ReacherSeedStreams:
    if _is_boolean_scalar(task_seed) or not isinstance(task_seed, (int, np.integer)):
        raise ValueError("expected an integer task seed")
    task_seed = int(task_seed)
    return ReacherSeedStreams(
        task_seed=task_seed,
        planner_seed=200000 + task_seed,
        sdob_observer_seed=300000 + task_seed,
        tube_observer_seed=310000 + task_seed,
        robust_observer_seed=320000 + task_seed,
        tube_feedback_seed=400000 + task_seed,
        robust_feedback_seed=500000 + task_seed,
    )


def _two_joint_vector(value: object) -> np.ndarray:
    try:
        components = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("expected a finite two-joint vector") from exc
    if components.shape != (2,) or any(
        _is_boolean_scalar(component) for component in components
    ):
        raise ValueError("expected a finite two-joint vector")
    try:
        vector = components.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("expected a finite two-joint vector") from exc
    if not np.all(np.isfinite(vector)):
        raise ValueError("expected a finite two-joint vector")
    return vector


def reacher_final_error(qpos: object, target_qpos: object) -> float:
    qpos_vector = _two_joint_vector(qpos)
    target_vector = _two_joint_vector(target_qpos)
    return float(np.max(np.abs(qpos_vector - target_vector)))


def joint_boundary_distance(qpos: object) -> float:
    qpos_vector = _two_joint_vector(qpos)
    return float(np.min(np.pi - np.abs(qpos_vector)))


def _finite_scalar(value: object) -> float:
    if _is_boolean_scalar(value):
        raise ValueError("expected a finite scalar")
    try:
        value_array = np.asarray(value)
        if value_array.shape != ():
            raise ValueError("expected a finite scalar")
        scalar = float(value_array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("expected a finite scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError("expected a finite scalar")
    return scalar


def _integer_step(value: object) -> int:
    if not _is_boolean_scalar(value) and isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and (
        value == "0"
        or (
            value
            and value[0] in "123456789"
            and all(character in "0123456789" for character in value[1:])
        )
    ):
        return int(value)
    raise ValueError("expected an integer step")


def validate_nominal_episode(
    summary: object,
    trajectory_rows: object,
) -> ValidatedNominalEpisode:
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    status = summary.get("status")
    if not isinstance(status, str) or status != "ok":
        raise ValueError("summary status must be 'ok'")
    episode_target = (
        _finite_scalar(summary["target_qpos_0"]),
        _finite_scalar(summary["target_qpos_1"]),
    )
    if not isinstance(trajectory_rows, Sequence) or isinstance(
        trajectory_rows, (str, bytes)
    ):
        raise ValueError("trajectory rows must be a sequence")

    if len(trajectory_rows) != EXPECTED_STEPS:
        raise ValueError(f"expected exactly {EXPECTED_STEPS} trajectory rows")
    for expected_step, row in enumerate(trajectory_rows):
        if not isinstance(row, Mapping):
            raise ValueError("trajectory row must be a mapping")
        if _integer_step(row["step"]) != expected_step:
            raise ValueError("trajectory steps must be contiguous from 0")
        for field in _REQUIRED_FINITE_ROW_FIELDS:
            _finite_scalar(row[field])
        for field, target in zip(
            ("target_qpos_0", "target_qpos_1"), episode_target
        ):
            if not np.isclose(
                _finite_scalar(row[field]),
                target,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("trajectory row target disagrees with episode target")

    raw_tail_errors = []
    for row in trajectory_rows[-ELIGIBILITY_TAIL_STEPS:]:
        raw_error = reacher_final_error(
            [row["qpos_0"], row["qpos_1"]],
            [row["target_qpos_0"], row["target_qpos_1"]],
        )
        if not np.isclose(
            raw_error,
            _finite_scalar(row["error"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("tail row error disagrees with raw joint error")
        raw_tail_errors.append(raw_error)

    raw_tail_mean_error = math.fsum(raw_tail_errors) / ELIGIBILITY_TAIL_STEPS
    summary_tail_error = _finite_scalar(summary["tail_error"])
    summary_final_error = _finite_scalar(summary["final_error"])

    if not np.isclose(
        raw_tail_mean_error,
        summary_tail_error,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("summary tail_error disagrees with raw tail mean")
    if not np.isclose(
        raw_tail_errors[-1],
        summary_final_error,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("summary final_error disagrees with raw final error")
    return ValidatedNominalEpisode(
        row_count=len(trajectory_rows),
        tail_mean_error=raw_tail_mean_error,
        final_error=raw_tail_errors[-1],
        eligible=bool(raw_tail_mean_error <= ELIGIBILITY_THRESHOLD_RAD),
    )


def eligible_nominal_episode(summary: object, trajectory_rows: object) -> bool:
    try:
        return validate_nominal_episode(summary, trajectory_rows).eligible
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return False


def _manifest_integer(value: object, field: str) -> int:
    if _is_boolean_scalar(value) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _manifest_float(value: object, field: str) -> float:
    if _is_boolean_scalar(value) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _validated_candidate_rows(candidates: object) -> list[dict[str, object]]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be a sequence")
    if len(candidates) != len(CANDIDATE_TASK_SEEDS):
        raise ValueError("candidate rows must cover the exact candidate block")

    normalized = []
    for expected_seed, candidate in zip(CANDIDATE_TASK_SEEDS, candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate row must be a mapping")
        missing = [field for field in SCREENING_RESULT_FIELDS if field not in candidate]
        if missing:
            raise ValueError(f"candidate metadata missing: {missing[0]}")

        task_seed = _manifest_integer(candidate["task_seed"], "candidate task_seed")
        if task_seed != expected_seed:
            raise ValueError("candidate rows must cover the exact candidate block")
        if candidate["status"] != "ok":
            raise ValueError("candidate status must be 'ok'")
        row_count = _manifest_integer(candidate["row_count"], "candidate row_count")
        if row_count != EXPECTED_STEPS:
            raise ValueError("candidate row_count must equal expected_steps")

        tail_mean_error = _manifest_float(
            candidate["tail_mean_error"], "candidate tail_mean_error"
        )
        final_error = _manifest_float(candidate["final_error"], "candidate final_error")
        eligible = candidate["eligible"]
        if not isinstance(eligible, (bool, np.bool_)):
            raise ValueError("candidate eligible must be boolean")
        eligible = bool(eligible)
        if eligible != (tail_mean_error <= ELIGIBILITY_THRESHOLD_RAD):
            raise ValueError("candidate eligible flag disagrees with threshold")

        initial_qpos = (
            _manifest_float(candidate["initial_qpos_0"], "candidate initial_qpos_0"),
            _manifest_float(candidate["initial_qpos_1"], "candidate initial_qpos_1"),
        )
        target_qpos = (
            _manifest_float(candidate["target_qpos_0"], "candidate target_qpos_0"),
            _manifest_float(candidate["target_qpos_1"], "candidate target_qpos_1"),
        )
        b_initial = _manifest_float(candidate["b_initial"], "candidate b_initial")
        b_target = _manifest_float(candidate["b_target"], "candidate b_target")
        if not np.isclose(
            b_initial,
            joint_boundary_distance(initial_qpos),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("candidate initial boundary distance is invalid")
        if not np.isclose(
            b_target,
            joint_boundary_distance(target_qpos),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("candidate target boundary distance is invalid")

        planner_seed = _manifest_integer(
            candidate["planner_seed"], "candidate planner_seed"
        )
        if planner_seed != seed_streams(task_seed).planner_seed:
            raise ValueError("candidate planner_seed is invalid")
        goal_hash = candidate["goal_image_sha256"]
        if (
            not isinstance(goal_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", goal_hash) is None
        ):
            raise ValueError("candidate goal_image_sha256 is invalid")

        normalized.append(
            {
                "task_seed": task_seed,
                "status": "ok",
                "row_count": row_count,
                "tail_mean_error": tail_mean_error,
                "final_error": final_error,
                "eligible": eligible,
                "initial_qpos_0": initial_qpos[0],
                "initial_qpos_1": initial_qpos[1],
                "target_qpos_0": target_qpos[0],
                "target_qpos_1": target_qpos[1],
                "b_initial": b_initial,
                "b_target": b_target,
                "planner_seed": planner_seed,
                "goal_image_sha256": goal_hash,
            }
        )
    return normalized


def validate_screening_candidate_rows(
    candidates: object,
) -> tuple[dict[str, object], ...]:
    return tuple(_validated_candidate_rows(candidates))


def deserialize_screening_result_rows(
    rows: object,
) -> tuple[dict[str, object], ...]:
    if isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("screening result rows must be iterable mappings")
    try:
        rows = list(rows)
    except TypeError as exc:
        raise ValueError("screening result rows must be iterable mappings") from exc

    candidates = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or len(row) != len(SCREENING_RESULT_FIELDS)
            or any(field not in row for field in SCREENING_RESULT_FIELDS)
        ):
            raise ValueError("screening result row fields are invalid")
        candidate = dict(row)

        for field in _SCREENING_RESULT_INTEGER_FIELDS:
            value = row[field]
            if not isinstance(value, str) or not (
                value == "0"
                or (
                    value
                    and value[0] in "123456789"
                    and all(character in "0123456789" for character in value[1:])
                )
            ):
                raise ValueError(f"screening result {field} is not canonical")
            candidate[field] = int(value)

        for field in _SCREENING_RESULT_FLOAT_FIELDS:
            value = row[field]
            if not isinstance(value, str):
                raise ValueError(f"screening result {field} is not canonical")
            try:
                parsed = float(value)
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"screening result {field} is not canonical"
                ) from exc
            if not math.isfinite(parsed) or str(parsed) != value:
                raise ValueError(f"screening result {field} is not canonical")
            candidate[field] = parsed

        eligible = row["eligible"]
        if eligible == "True":
            candidate["eligible"] = True
        elif eligible == "False":
            candidate["eligible"] = False
        elif type(eligible) is not bool:
            raise ValueError("screening result eligible is not canonical")

        candidates.append(candidate)

    return validate_screening_candidate_rows(candidates)


def selected_seed_manifest_sha256(manifest: object) -> str:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    try:
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest is not canonical JSON data") from exc
    return hashlib.sha256(canonical).hexdigest()


def build_selected_seed_manifest(candidates: object) -> dict[str, object]:
    candidate_rows = _validated_candidate_rows(candidates)
    eligible_seeds = [
        int(candidate["task_seed"])
        for candidate in candidate_rows
        if candidate["eligible"]
    ]
    if len(eligible_seeds) < SELECTED_SEED_COUNT:
        raise ValueError(f"fewer than {SELECTED_SEED_COUNT} eligible candidates")
    selected_seeds = eligible_seeds[:SELECTED_SEED_COUNT]
    selected_set = set(selected_seeds)
    manifest: dict[str, object] = {
        "schema_version": SCREENING_MANIFEST_SCHEMA_VERSION,
        "candidate_task_seeds": list(CANDIDATE_TASK_SEEDS),
        "eligibility_threshold_rad": ELIGIBILITY_THRESHOLD_RAD,
        "eligibility_tail_steps": ELIGIBILITY_TAIL_STEPS,
        "expected_steps": EXPECTED_STEPS,
        "selection_count": SELECTED_SEED_COUNT,
        "selection_rule": SCREENING_SELECTION_RULE,
        "selected_task_seeds": selected_seeds,
        "rejected_task_seeds": [
            seed for seed in CANDIDATE_TASK_SEEDS if seed not in selected_set
        ],
        "candidates": candidate_rows,
    }
    manifest["manifest_sha256"] = selected_seed_manifest_sha256(manifest)
    return manifest


def validate_selected_seed_manifest(manifest: object) -> tuple[int, ...]:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    manifest_hash = manifest.get("manifest_sha256")
    if (
        not isinstance(manifest_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None
    ):
        raise ValueError("manifest hash is missing or malformed")
    if manifest_hash != selected_seed_manifest_sha256(manifest):
        raise ValueError("manifest hash mismatch")

    schema_version = _manifest_integer(
        manifest.get("schema_version"), "manifest schema_version"
    )
    if schema_version != SCREENING_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema_version")
    candidate_task_seeds = manifest.get("candidate_task_seeds")
    if (
        not isinstance(candidate_task_seeds, list)
        or any(
            _manifest_integer(seed, "candidate task seed") != expected
            for seed, expected in zip(candidate_task_seeds, CANDIDATE_TASK_SEEDS)
        )
        or len(candidate_task_seeds) != len(CANDIDATE_TASK_SEEDS)
    ):
        raise ValueError("manifest candidate block is invalid")
    threshold = _manifest_float(
        manifest.get("eligibility_threshold_rad"), "eligibility threshold"
    )
    if threshold != ELIGIBILITY_THRESHOLD_RAD:
        raise ValueError("manifest eligibility threshold is invalid")
    if (
        _manifest_integer(
            manifest.get("eligibility_tail_steps"), "eligibility tail steps"
        )
        != ELIGIBILITY_TAIL_STEPS
    ):
        raise ValueError("manifest eligibility tail steps are invalid")
    if (
        _manifest_integer(manifest.get("expected_steps"), "expected steps")
        != EXPECTED_STEPS
    ):
        raise ValueError("manifest expected steps are invalid")
    if (
        _manifest_integer(manifest.get("selection_count"), "selection count")
        != SELECTED_SEED_COUNT
    ):
        raise ValueError("manifest selection count is invalid")
    if manifest.get("selection_rule") != SCREENING_SELECTION_RULE:
        raise ValueError("manifest selection rule is invalid")

    candidate_rows = _validated_candidate_rows(manifest.get("candidates"))
    selected_values = manifest.get("selected_task_seeds")
    if not isinstance(selected_values, list):
        raise ValueError("selected task seeds must be a list")
    selected_seeds = [
        _manifest_integer(seed, "selected task seed") for seed in selected_values
    ]
    if len(selected_seeds) != SELECTED_SEED_COUNT:
        raise ValueError("selected task seed count is invalid")
    if selected_seeds != sorted(selected_seeds) or len(set(selected_seeds)) != len(
        selected_seeds
    ):
        raise ValueError("selected task seeds must be unique and sorted")

    expected_selected = [
        int(candidate["task_seed"])
        for candidate in candidate_rows
        if candidate["eligible"]
    ][:SELECTED_SEED_COUNT]
    if selected_seeds != expected_selected:
        raise ValueError("selected task seeds must be the first 20 eligible candidates")
    selected_set = set(selected_seeds)
    expected_rejected = [
        seed for seed in CANDIDATE_TASK_SEEDS if seed not in selected_set
    ]
    rejected_values = manifest.get("rejected_task_seeds")
    if not isinstance(rejected_values, list):
        raise ValueError("rejected task seeds must be a list")
    rejected_seeds = [
        _manifest_integer(seed, "rejected task seed") for seed in rejected_values
    ]
    if rejected_seeds != expected_rejected:
        raise ValueError("rejected task seeds must be the sorted selection complement")
    return tuple(selected_seeds)


def selected_seed_candidates(manifest: object) -> tuple[dict[str, object], ...]:
    selected_seeds = validate_selected_seed_manifest(manifest)
    candidates_by_seed = {
        int(candidate["task_seed"]): candidate
        for candidate in _validated_candidate_rows(manifest.get("candidates"))
    }
    return tuple(dict(candidates_by_seed[task_seed]) for task_seed in selected_seeds)


def _canonical_boolean(value: object, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field} must be a canonical boolean")


def _formal_rows(
    rows: object,
    *,
    name: str,
    expected_count: int,
) -> Sequence[Mapping[str, object]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{name} rows must be a sequence")
    if len(rows) != expected_count:
        raise ValueError(f"expected exactly {expected_count} {name} rows")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{name} row must be a mapping")
    return rows


def _formal_identity(
    row: Mapping[str, object],
    *,
    name: str,
) -> tuple[str, str]:
    case = row.get("case")
    method = row.get("method")
    if not isinstance(case, str) or not isinstance(method, str):
        raise ValueError(f"{name} case and method must be strings")
    return case, method


def _formal_candidate_identity(
    selected_candidate: object,
    task_seed: int,
) -> dict[str, object]:
    if not isinstance(selected_candidate, Mapping):
        raise ValueError("selected candidate must be a mapping")
    candidate_task_seed = _manifest_integer(
        selected_candidate.get("task_seed"),
        "selected candidate task_seed",
    )
    if candidate_task_seed != task_seed:
        raise ValueError("selected candidate task_seed mismatch")
    streams = seed_streams(candidate_task_seed)
    planner_seed = _manifest_integer(
        selected_candidate.get("planner_seed"),
        "selected candidate planner_seed",
    )
    if planner_seed != streams.planner_seed:
        raise ValueError("selected candidate planner_seed mismatch")
    initial_qpos = tuple(
        _manifest_float(
            selected_candidate.get(field),
            f"selected candidate {field}",
        )
        for field in ("initial_qpos_0", "initial_qpos_1")
    )
    target_qpos = tuple(
        _manifest_float(
            selected_candidate.get(field),
            f"selected candidate {field}",
        )
        for field in ("target_qpos_0", "target_qpos_1")
    )
    goal_hash = selected_candidate.get("goal_image_sha256")
    if (
        not isinstance(goal_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", goal_hash) is None
    ):
        raise ValueError("selected candidate goal_image_sha256 is invalid")
    return {
        "task_seed": candidate_task_seed,
        "planner_seed": planner_seed,
        "initial_qpos": initial_qpos,
        "target_qpos": target_qpos,
        "goal_image_sha256": goal_hash,
    }


def validate_formal_seed_rows(
    task_seed: int,
    summary_rows: object,
    trajectory_rows: object,
    diagnostic_rows: object,
    *,
    selected_candidate: object,
) -> ValidatedFormalSeed:
    task_seed = seed_streams(task_seed).task_seed
    streams = seed_streams(task_seed)
    candidate = _formal_candidate_identity(selected_candidate, task_seed)
    expected_cases = {
        (case, method): disturbed
        for case, method, disturbed in FORMAL_REACHER_CASES
    }
    expected_row_count = len(FORMAL_REACHER_CASES) * EXPECTED_STEPS
    summaries = _formal_rows(
        summary_rows,
        name="summary",
        expected_count=len(FORMAL_REACHER_CASES),
    )

    normalized_summaries: dict[tuple[str, str], dict[str, object]] = {}
    for row in summaries:
        identity = _formal_identity(row, name="summary")
        if identity not in expected_cases:
            raise ValueError("summary contains an unexpected case or method")
        if identity in normalized_summaries:
            raise ValueError("summary contains a duplicate case or method")
        expected_disturbed = expected_cases[identity]
        if _canonical_boolean(row.get("disturbed"), "summary disturbed") != expected_disturbed:
            raise ValueError("summary disturbed flag disagrees with canonical case")
        if row.get("status") != "ok":
            raise ValueError("summary status must be 'ok'")
        if _integer_step(row.get("steps")) != EXPECTED_STEPS:
            raise ValueError(f"summary steps must equal {EXPECTED_STEPS}")

        for field in streams.__dataclass_fields__:
            try:
                actual_seed = _integer_step(row.get(field))
            except ValueError as exc:
                raise ValueError(f"summary {field} is invalid") from exc
            if actual_seed != getattr(streams, field):
                raise ValueError(f"summary {field} mismatch")

        initial_qpos = (
            _finite_scalar(row.get("initial_qpos_0")),
            _finite_scalar(row.get("initial_qpos_1")),
        )
        target_qpos = (
            _finite_scalar(row.get("target_qpos_0")),
            _finite_scalar(row.get("target_qpos_1")),
        )
        for field, actual, expected in zip(
            ("initial_qpos_0", "initial_qpos_1"),
            initial_qpos,
            candidate["initial_qpos"],
        ):
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                raise ValueError(f"summary {field} mismatch")
        for field, actual, expected in zip(
            ("target_qpos_0", "target_qpos_1"),
            target_qpos,
            candidate["target_qpos"],
        ):
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                raise ValueError(f"summary {field} mismatch")
        goal_hash = row.get("goal_image_sha256")
        if (
            not isinstance(goal_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", goal_hash) is None
        ):
            raise ValueError("summary goal_image_sha256 is invalid")
        if goal_hash != candidate["goal_image_sha256"]:
            raise ValueError("summary goal_image_sha256 mismatch")

        tail_error = _finite_scalar(row.get("tail_error"))
        final_error = _finite_scalar(row.get("final_error"))
        success = _canonical_boolean(row.get("success"), "summary success")
        if success != (tail_error < FORMAL_SUCCESS_THRESHOLD_RAD):
            raise ValueError("summary success disagrees with tail_error")
        normalized_summaries[identity] = {
            "task_seed": task_seed,
            "initial_qpos": initial_qpos,
            "target_qpos": target_qpos,
            "goal_image_sha256": goal_hash,
            "tail_error": tail_error,
            "final_error": final_error,
            "success": success,
        }

    if set(normalized_summaries) != set(expected_cases):
        raise ValueError("summary case grid is incomplete")

    trajectories = _formal_rows(
        trajectory_rows,
        name="trajectory",
        expected_count=expected_row_count,
    )
    normalized_trajectories: dict[
        tuple[str, str, int], dict[str, object]
    ] = {}
    for row in trajectories:
        identity = _formal_identity(row, name="trajectory")
        if identity not in expected_cases:
            raise ValueError("trajectory contains an unexpected case or method")
        step = _integer_step(row.get("step"))
        if step < 0 or step >= EXPECTED_STEPS:
            raise ValueError("trajectory step is out of range")
        key = (*identity, step)
        if key in normalized_trajectories:
            raise ValueError("trajectory contains a duplicate case/method/step")

        finite_values = {
            field: _finite_scalar(row.get(field))
            for field in _REQUIRED_FINITE_ROW_FIELDS
        }
        summary_target = normalized_summaries[identity]["target_qpos"]
        for field, target in zip(
            ("target_qpos_0", "target_qpos_1"),
            summary_target,
        ):
            if not np.isclose(
                finite_values[field],
                target,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("trajectory target disagrees with method summary target")
        normalized_trajectories[key] = finite_values

    expected_keys = {
        (case, method, step)
        for case, method, _disturbed in FORMAL_REACHER_CASES
        for step in range(EXPECTED_STEPS)
    }
    if set(normalized_trajectories) != expected_keys:
        raise ValueError("trajectory case/method/step grid is incomplete")

    for case, method, _disturbed in FORMAL_REACHER_CASES:
        identity = (case, method)
        raw_tail_errors = []
        for step in range(EXPECTED_STEPS - ELIGIBILITY_TAIL_STEPS, EXPECTED_STEPS):
            values = normalized_trajectories[(*identity, step)]
            raw_error = reacher_final_error(
                (values["qpos_0"], values["qpos_1"]),
                (values["target_qpos_0"], values["target_qpos_1"]),
            )
            if not np.isclose(
                raw_error,
                values["error"],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("tail trajectory error disagrees with raw joint error")
            raw_tail_errors.append(raw_error)
        raw_tail_mean = math.fsum(raw_tail_errors) / ELIGIBILITY_TAIL_STEPS
        summary = normalized_summaries[identity]
        if not np.isclose(
            raw_tail_mean,
            summary["tail_error"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("summary tail_error disagrees with raw tail mean")
        if not np.isclose(
            raw_tail_errors[-1],
            summary["final_error"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("summary final_error disagrees with raw final error")
        if summary["success"] != (
            raw_tail_mean < FORMAL_SUCCESS_THRESHOLD_RAD
        ):
            raise ValueError("summary success disagrees with raw tail mean")

    diagnostics = _formal_rows(
        diagnostic_rows,
        name="diagnostic",
        expected_count=expected_row_count,
    )
    diagnostic_keys = set()
    for row in diagnostics:
        identity = _formal_identity(row, name="diagnostic")
        if identity not in expected_cases:
            raise ValueError("diagnostic contains an unexpected case or method")
        step = _integer_step(row.get("step"))
        if step < 0 or step >= EXPECTED_STEPS:
            raise ValueError("diagnostic step is out of range")
        key = (*identity, step)
        if key in diagnostic_keys:
            raise ValueError("diagnostic contains a duplicate case/method/step")
        if (
            _canonical_boolean(row.get("disturbed"), "diagnostic disturbed")
            != expected_cases[identity]
        ):
            raise ValueError("diagnostic disturbed flag disagrees with canonical case")
        summary = normalized_summaries[identity]
        try:
            diagnostic_task_seed = _integer_step(row.get("task_seed"))
        except ValueError as exc:
            raise ValueError("diagnostic task_seed is invalid") from exc
        if (
            diagnostic_task_seed != candidate["task_seed"]
            or diagnostic_task_seed != summary["task_seed"]
        ):
            raise ValueError("diagnostic task_seed mismatch")
        for fields, candidate_values, summary_values in (
            (
                ("initial_qpos_0", "initial_qpos_1"),
                candidate["initial_qpos"],
                summary["initial_qpos"],
            ),
            (
                ("target_qpos_0", "target_qpos_1"),
                candidate["target_qpos"],
                summary["target_qpos"],
            ),
        ):
            for field, expected_candidate, expected_summary in zip(
                fields,
                candidate_values,
                summary_values,
            ):
                actual = _finite_scalar(row.get(field))
                if not np.isclose(
                    actual,
                    expected_candidate,
                    rtol=0.0,
                    atol=1e-12,
                ) or not np.isclose(
                    actual,
                    expected_summary,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(f"diagnostic {field} mismatch")
        diagnostic_goal_hash = row.get("goal_image_sha256")
        if (
            diagnostic_goal_hash != candidate["goal_image_sha256"]
            or diagnostic_goal_hash != summary["goal_image_sha256"]
        ):
            raise ValueError("diagnostic goal_image_sha256 mismatch")
        for field in ("u_applied_0", "u_applied_1"):
            try:
                actual_control = _finite_scalar(row.get(field))
            except ValueError as exc:
                raise ValueError(f"diagnostic {field} must be finite") from exc
            if not np.isclose(
                actual_control,
                normalized_trajectories[key][field],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"diagnostic {field} mismatch")
        true_disturbance = {}
        for field in ("d_true_0", "d_true_1"):
            try:
                true_disturbance[field] = _finite_scalar(row.get(field))
            except ValueError as exc:
                raise ValueError(f"diagnostic {field} must be finite") from exc
        if not expected_cases[identity]:
            for field, value in true_disturbance.items():
                if value != 0.0:
                    raise ValueError(
                        f"diagnostic {field} must be exactly zero for nominal case"
                    )
        for field in ("d_hat_0", "d_hat_1"):
            if field not in row:
                raise ValueError(f"diagnostic {field} is missing")
            value = row[field]
            if identity[0] in _FORMAL_OBSERVER_CASE_NAMES or value != "":
                try:
                    _finite_scalar(value)
                except ValueError as exc:
                    raise ValueError(f"diagnostic {field} must be finite") from exc
        diagnostic_keys.add(key)

    if diagnostic_keys != expected_keys:
        raise ValueError("diagnostic grid is not synchronized with trajectories")
    return ValidatedFormalSeed(
        task_seed=task_seed,
        summary_row_count=len(summaries),
        trajectory_row_count=len(trajectories),
        diagnostic_row_count=len(diagnostics),
    )
