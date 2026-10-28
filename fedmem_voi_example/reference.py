"""Dependency-free, explicit-input reference primitives for the prospective design.

Callers must supply frozen, independently fitted critics and qualified histories.
This module cannot establish exchangeability, transport, independence or provenance.
"""
from dataclasses import dataclass
import math


def finite(x, name):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError(name + ' must be a finite number')
    return x


def nonnegative(x, name):
    finite(x, name)
    if x < 0:
        raise ValueError(name + ' must be nonnegative')
    return x


def _mask(mask):
    if not isinstance(mask, dict) or not mask or any(
        not isinstance(k, str) or not k or type(v) is not bool for k, v in mask.items()
    ) or not any(mask.values()):
        raise ValueError('nonempty named boolean mask with a feasible action required')


def uniform_logging(mask):
    """Known design probabilities; never infer propensities from observed counts."""
    _mask(mask)
    n = sum(mask.values())
    return {a: 1 / n if allowed else 0.0 for a, allowed in mask.items()}


def masked_softmax(logits, mask):
    """Temperature-one masked probabilities, no argmax/heuristic fallback."""
    _mask(mask)
    if not isinstance(logits, dict) or set(logits) != set(mask):
        raise ValueError('logits and mask action sets must match')
    for x in logits.values():
        finite(x, 'logit')
    top = max(logits[a] for a in mask if mask[a])
    weights = {a: math.exp(logits[a] - top) if mask[a] else 0.0 for a in mask}
    if any(mask[a] and weights[a] == 0 for a in mask):
        raise ValueError('floating-point underflow; exact feasible support not representable')
    total = math.fsum(weights.values())
    return {a: w / total for a, w in weights.items()}


def discovery_guard(worst_case_discovery, task_cap):
    return nonnegative(worst_case_discovery, 'discovery bound') <= nonnegative(task_cap, 'task cap')


def feasible_mask(permissions, remaining_bounds, sunk_cost, task_cap):
    """Caller supplies reply/status-specific worst-case remaining byte bounds."""
    if not isinstance(permissions, dict) or not isinstance(remaining_bounds, dict) or set(permissions) != set(remaining_bounds):
        raise ValueError('permission and bound action sets must match')
    nonnegative(sunk_cost, 'sunk cost'); nonnegative(task_cap, 'task cap')
    if sunk_cost > task_cap:
        raise ValueError('already exceeded task cap')
    for a, p in permissions.items():
        if type(p) is not bool:
            raise ValueError('boolean permissions required')
        nonnegative(remaining_bounds[a], 'remaining bound')
    result = {a: p and sunk_cost + remaining_bounds[a] <= task_cap for a, p in permissions.items()}
    _mask(result)
    return result


@dataclass(frozen=True)
class Outcome:
    status: str
    value: float | None
    reason: str | None

    def __post_init__(self):
        if self.status == 'observed':
            finite(self.value, 'observed outcome')
            if not 0 <= self.value <= 1 or self.reason is not None:
                raise ValueError('observed outcome requires [0,1] value and no missingness reason')
        elif self.status == 'Unknown':
            if self.value is not None or not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError('Unknown requires no numeric label and an explicit reason')
        else:
            raise ValueError('outcome status must be observed or Unknown')

    def training_label(self):
        if self.status != 'observed':
            raise ValueError('Unknown is not a training label')
        return self.value

    def endpoints(self):
        return (self.value, self.value) if self.status == 'observed' else (0.0, 1.0)


@dataclass(frozen=True)
class Stage:
    mask: dict
    action: str
    logged: dict
    target: dict
    critic: dict

    def __post_init__(self):
        expected = uniform_logging(self.mask)
        if self.action not in self.mask or not self.mask[self.action]:
            raise ValueError('recorded assignment must be feasible')
        for values in (self.logged, self.target, self.critic):
            if not isinstance(values, dict) or set(values) != set(self.mask):
                raise ValueError('all stage maps must match the full action set')
            for x in values.values():
                finite(x, 'stage value')
        # The public example supports only the prespecified uniform feasible logger.
        if self.logged != expected:
            raise ValueError('logged probabilities must equal the known uniform design')
        if any(p < 0 or (not self.mask[a] and p != 0) for a, p in self.target.items()):
            raise ValueError('target must respect the recorded mask')
        if not math.isclose(math.fsum(self.target.values()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError('target probabilities must sum to one')

    def ratio(self):
        return self.target[self.action] / self.logged[self.action]

    def average(self):
        return math.fsum(self.target[a] * self.critic[a] for a in self.mask)


@dataclass(frozen=True)
class Assignment:
    guard: bool
    outcome: Outcome
    candidates: tuple | None
    stage0: Stage
    stage1: Stage
    refinement_status: str
    delivery_status: str
    discovery_cost: float
    refinement_cost: float
    payload_cost: float

    def __post_init__(self):
        if type(self.guard) is not bool or not isinstance(self.outcome, Outcome):
            raise ValueError('explicit boolean guard and Outcome required')
        if not isinstance(self.stage0, Stage) or not isinstance(self.stage1, Stage):
            raise ValueError('both recorded stages required')
        for cost in (self.discovery_cost, self.refinement_cost, self.payload_cost):
            nonnegative(cost, 'acquisition cost')
        if self.refinement_status not in ('skip', 'reply', 'unresolved', 'transport_failure'):
            raise ValueError('explicit refinement status required')
        if self.delivery_status not in ('withhold', 'delivered', 'transport_failure'):
            raise ValueError('explicit delivery status required')
        if self.guard:
            if not isinstance(self.candidates, tuple):
                raise ValueError('candidate/version tuple required on acquisition branch')
            if any(not isinstance(c, tuple) or len(c) != 2 or
                   any(not isinstance(x, str) or not x for x in c) for c in self.candidates):
                raise ValueError('each candidate requires opaque identifier and version')
        else:
            if self.candidates is not None or any((self.discovery_cost, self.refinement_cost, self.payload_cost)):
                raise ValueError('forced Local requires null candidates and zero acquisition cost')
            for stage in (self.stage0, self.stage1):
                if stage.mask != {'dummy': True} or stage.action != 'dummy' or stage.target != {'dummy': 1.0}:
                    raise ValueError('forced Local requires probability-one dummy actions')
            if self.refinement_status != 'skip' or self.delivery_status != 'withhold':
                raise ValueError('forced Local cannot acquire')


def sequential_dr(row, cost_multiplier):
    """V8 Eq. (10), complete observed assignment only, branch evaluated first."""
    nonnegative(cost_multiplier, 'cost multiplier')
    y = row.outcome.training_label()
    if not row.guard:
        return y  # Genuine observed Local outcome, not an invented zero.
    s0, s1 = row.stage0, row.stage1
    w0, w1 = s0.ratio(), s1.ratio()
    return (-cost_multiplier * row.discovery_cost + s0.average()
            + w0 * (-cost_multiplier * row.refinement_cost + s1.average() - s0.critic[s0.action])
            + w0 * w1 * (y - cost_multiplier * row.payload_cost - s1.critic[s1.action]))


def cohort_decision(rows):
    """Reject entire supplied cohort if any outcome Unknown; never drop rows.

    Only a qualification decision. No fallback learner/policy is implemented.
    The caller must prove this is the entire prospectively allocated cohort.
    """
    if not isinstance(rows, (list, tuple)) or not rows or any(not isinstance(r, Assignment) for r in rows):
        raise ValueError('nonempty complete planned assignment cohort required')
    unknown = sum(r.outcome.status == 'Unknown' for r in rows)
    return {'eligible_for_complete_data_score': unknown == 0,
            'assigned_count': len(rows), 'unknown_count': unknown,
            'decision': 'eligible' if unknown == 0 else 'reject_whole_cohort'}


def paired_endpoints(full, comparator, client_weights):
    """Direct-evaluation paired identification bounds, not sampling intervals.

    Maps: client -> {prospectively assigned task ID: Outcome}. Missing tasks
    must be explicitly present as Unknown; no complete-case filtering or IPW.
    """
    if not all(isinstance(v, dict) for v in (full, comparator, client_weights)) or not client_weights or set(full) != set(comparator) or set(full) != set(client_weights):
        raise ValueError('same nonempty planned clients required')
    for w in client_weights.values():
        if nonnegative(w, 'client weight') == 0:
            raise ValueError('positive target client weights required')
    if not math.isclose(math.fsum(client_weights.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError('target client weights must sum to one')
    lower, upper = [], []
    for j, weight in client_weights.items():
        if not isinstance(full[j], dict) or not isinstance(comparator[j], dict) or not full[j] or set(full[j]) != set(comparator[j]):
            raise ValueError('identical nonempty planned paired task rosters required')
        for task in full[j]:
            f, c = full[j][task], comparator[j][task]
            if not isinstance(f, Outcome) or not isinstance(c, Outcome):
                raise ValueError('explicit Outcome for every assignment required')
            fl, fu = f.endpoints(); cl, cu = c.endpoints()
            lower.append(weight / len(full[j]) * (fl - cu))
            upper.append(weight / len(full[j]) * (fu - cl))
    return math.fsum(lower), math.fsum(upper)


@dataclass(frozen=True)
class CohortPlan:
    """Prospectively frozen (slot ID, client ID, whole-block role) registry."""
    slots: tuple
    completion_boundary: float
    methods: tuple

    def __post_init__(self):
        finite(self.completion_boundary, 'finite completion boundary')
        if not isinstance(self.slots, tuple) or not self.slots:
            raise ValueError('zero feedback requires a separately declared algorithm')
        for slot in self.slots:
            if not isinstance(slot, tuple) or len(slot) != 3 or any(
                    not isinstance(v, str) or not v.strip() for v in slot):
                raise ValueError('slot, client and whole-block role identifiers required')
        if len({s[0] for s in self.slots}) != len(self.slots):
            raise ValueError('unique planned slot IDs required')
        if not isinstance(self.methods, tuple) or not self.methods or any(
                not isinstance(m, str) or not m.strip() for m in self.methods
        ) or len(set(self.methods)) != len(self.methods):
            raise ValueError('unique nonempty matched learner names required')


def _validate_feedback(record, client, role):
    required = {'client', 'role', 'status', 'assignment', 'history0', 'history1'}
    if not isinstance(record, dict) or not required <= record.keys():
        raise ValueError('missing required feedback field')
    if record['client'] != client or record['role'] != role:
        raise ValueError('planned client/whole-block role mismatch')
    if record['status'] != 'complete':
        raise ValueError('assigned slot NotRun or unfinished')
    raw = record['assignment']
    if not isinstance(raw, dict):
        raise ValueError('explicit assignment field map required')
    fields = Assignment.__dataclass_fields__
    if set(raw) != set(fields):
        raise ValueError('missing or unexpected assignment field')
    raw = dict(raw)
    for stage_name in ('stage0', 'stage1'):
        values = raw[stage_name]
        if not isinstance(values, dict) or set(values) != {'mask', 'action', 'logged'}:
            raise ValueError('missing stage probability/mask/action field')
        if not isinstance(values['mask'], dict) or not isinstance(values['logged'], dict):
            raise ValueError('stage mask and logged probabilities must be mappings')
        # Validation-only placeholders: no fitted critic/target is required
        # before fit, and these temporary values are never scored or returned.
        raw[stage_name] = Stage(**values, target=values['logged'],
                                critic={a: 0.0 for a in values['mask']})
    if not isinstance(raw['outcome'], dict) or set(raw['outcome']) != set(Outcome.__dataclass_fields__):
        raise ValueError('explicit assigned outcome/status/reason required')
    raw['outcome'] = Outcome(**raw['outcome'])
    row = Assignment(**raw)
    row.outcome.training_label()
    if row.guard:
        if any(not isinstance(record[h], dict) or not record[h] for h in ('history0', 'history1')):
            raise ValueError('required actual acquisition histories missing')
        # Contents are opaque permitted snapshots: explicit unresolved and actual
        # failure/null symbols count as observed. No counterfactual is requested.
    elif record['history0'] is not None or record['history1'] is not None:
        raise ValueError('forced Local requires null acquisition histories')


def procedure_decision(plan, records, *, boundary_time, phase, technical_flags,
                       evaluation_started):
    """Pure prospective fit/fallback routing, never fitting or executing a policy.

    Call once at the predeclared collection boundary, then bind final method
    choices before evaluation. Retain input records/allocations and all ledgers
    externally. This function never mutates, subsets or fabricates observations.
    Technical validity is a predeclared method-specific predicate, not a score.
    """
    if not isinstance(plan, CohortPlan):
        raise ValueError('frozen CohortPlan required')
    finite(boundary_time, 'boundary time')
    if type(evaluation_started) is not bool or evaluation_started:
        raise ValueError('no routing or fallback after evaluation has started')
    if boundary_time < plan.completion_boundary:
        raise ValueError('collection boundary not reached; no premature routing')
    if phase not in ('before_fit', 'before_evaluation'):
        raise ValueError('explicit pre-fit or pre-evaluation phase required')
    if not isinstance(records, dict) or not isinstance(technical_flags, dict):
        raise ValueError('explicit record/technical flag mappings required')
    reasons = []
    expected = {s[0] for s in plan.slots}
    if set(records) != expected:
        reasons.append('returned_registry_mismatch')
    for slot, client, role in plan.slots:
        if slot in records:
            try:
                _validate_feedback(records[slot], client, role)
            except (ValueError, TypeError, KeyError):
                reasons.append('invalid_required_record:' + slot)
    eligible = not reasons
    if not eligible:
        if technical_flags:
            raise ValueError('ineligible cohort must never reach fitting')
        selections = {m: {'selection': 'Local', 'reason': 'shared_cohort_incomplete'} for m in plan.methods}
    elif phase == 'before_fit':
        if technical_flags:
            raise ValueError('technical fit flags cannot precede fitting')
        selections = {m: {'selection': 'fit_full_cohort', 'reason': 'shared_cohort_eligible'} for m in plan.methods}
    else:
        if set(technical_flags) != set(plan.methods) or any(
                f not in ('valid', 'valid_at_limit', 'technical_failure') for f in technical_flags.values()):
            raise ValueError('one predeclared technical completion flag per method required')
        selections = {m: {'selection': 'Local' if flag == 'technical_failure' else 'frozen_actor',
                          'reason': flag} for m, flag in technical_flags.items()}
    return {'cohort_eligible': eligible, 'planned_count': len(plan.slots),
            'returned_count': len(records), 'cohort_reasons': tuple(reasons),
            'methods': selections}
