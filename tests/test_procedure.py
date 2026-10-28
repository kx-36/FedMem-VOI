import copy
import unittest
from dataclasses import asdict
from fedmem_voi_example.reference import (
    Assignment, CohortPlan, Outcome, Stage, procedure_decision, sequential_dr,
)


def record(guard=False):
    dummy = Stage({'dummy': True}, 'dummy', {'dummy': 1.}, {'dummy': 1.}, {'dummy': 0.})
    row = Assignment(guard, Outcome('observed', 0., None), (('note', 'v1'),) if guard else None,
                     dummy, dummy, 'transport_failure' if guard else 'skip',
                     'transport_failure' if guard else 'withhold', 1 if guard else 0, 0, 0)
    raw = asdict(row)
    for key in ('stage0', 'stage1'):
        raw[key].pop('critic'); raw[key].pop('target')
    return {'client': 'c', 'role': 'receiver_learning_block', 'status': 'complete',
            'assignment': raw, 'history0': {'retrieval': 'observed'} if guard else None,
            'history1': {'source_symbol': 'unresolved', 'reply': 'transport_failure'} if guard else None}


class ProcedureTests(unittest.TestCase):
    def setUp(self):
        self.plan = CohortPlan((('s', 'c', 'receiver_learning_block'),), 10., ('Full', 'ACO'))

    def decide(self, records, **kwargs):
        arguments = dict(boundary_time=10., phase='before_fit', technical_flags={}, evaluation_started=False)
        arguments.update(kwargs)
        return procedure_decision(self.plan, records, **arguments)

    def test_complete_native_false_and_failure_unresolved_are_eligible(self):
        for guard in (False, True):
            self.assertEqual(self.decide({'s': record(guard)})['methods']['Full']['selection'], 'fit_full_cohort')

    def test_shared_gate_missing_components_and_absent_rows(self):
        variants = [{}]
        for path in (('status',), ('history1',), ('assignment', 'outcome'),
                     ('assignment', 'payload_cost'), ('assignment', 'stage0', 'logged'),
                     ('assignment', 'stage1', 'mask'), ('assignment', 'candidates')):
            row = record(True); target = row
            for key in path[:-1]: target = target[key]
            del target[path[-1]]
            variants.append({'s': row})
        for modification in ('Unknown', 'NotRun', 'missing_cost', 'missing_history'):
            row = record(True)
            if modification == 'Unknown':
                row['assignment']['outcome'] = {'status': 'Unknown', 'value': None, 'reason': 'missing'}
            elif modification == 'NotRun': row['status'] = 'NotRun'
            elif modification == 'missing_cost': row['assignment']['payload_cost'] = None
            else: row['history1'] = None
            variants.append({'s': row})
        for records in variants:
            preserved = copy.deepcopy(records)
            result = self.decide(records)
            self.assertFalse(result['cohort_eligible'])
            self.assertEqual(result['planned_count'], 1)
            self.assertTrue(all(v['selection'] == 'Local' for v in result['methods'].values()))
            self.assertEqual(records, preserved)

    def test_malformed_probability_containers_route_shared_local(self):
        for stage in ('stage0', 'stage1'):
            for field in ('logged', 'mask'):
                for malformed in (['dummy'], ('dummy',), None, 'dummy'):
                    row = record()
                    row['assignment'][stage][field] = malformed
                    result = self.decide({'s': row})
                    self.assertFalse(result['cohort_eligible'])
                    self.assertTrue(all(v['selection'] == 'Local' for v in result['methods'].values()))
        for field in ('logged', 'target', 'critic'):
            kwargs = dict(mask={'dummy': True}, action='dummy', logged={'dummy': 1.},
                          target={'dummy': 1.}, critic={'dummy': 0.})
            kwargs[field] = ['dummy']
            with self.assertRaises(ValueError):
                Stage(**kwargs)

    def test_postfit_flags_are_method_specific_and_limit_is_completion(self):
        result = self.decide({'s': record()}, phase='before_evaluation',
                             technical_flags={'Full': 'technical_failure', 'ACO': 'valid_at_limit'})
        self.assertEqual(result['methods']['Full']['selection'], 'Local')
        self.assertEqual(result['methods']['ACO']['selection'], 'frozen_actor')
        result = self.decide({'s': record()}, phase='before_evaluation',
                             technical_flags={'Full': 'valid', 'ACO': 'valid'})
        self.assertTrue(all(v['selection'] == 'frozen_actor' for v in result['methods'].values()))
        with self.assertRaises(ValueError):
            self.decide({}, technical_flags={'Full': 'technical_failure'})

    def test_fixed_boundary_zero_feedback_and_no_runtime_fallback(self):
        with self.assertRaises(ValueError): CohortPlan((), 10., ('Full',))
        with self.assertRaises(ValueError): self.decide({'s': record()}, boundary_time=9.)
        with self.assertRaises(ValueError): self.decide({'s': record()}, evaluation_started=True)
        with self.assertRaises(ValueError): self.decide({'s': record()}, phase='before_evaluation')
        # Explicit extra returned slots invalidate, never silently subset.
        self.assertFalse(self.decide({'s': record(), 'extra': record()})['cohort_eligible'])

    def test_acceptance_does_not_establish_conditional_dr_identity(self):
        # Fair A, Y=A, observed only for A=1. Eligible one-row cohorts contain
        # A=1 only. Exact original p=.5 still gives conditional score 2, not 1.
        s0 = Stage({'zero': True, 'one': True}, 'one', {'zero': .5, 'one': .5},
                   {'zero': 0., 'one': 1.}, {'zero': 0., 'one': 0.})
        s1 = Stage({'withhold': True}, 'withhold', {'withhold': 1.}, {'withhold': 1.}, {'withhold': 0.})
        row = Assignment(True, Outcome('observed', 1., None), (), s0, s1, 'skip', 'withhold', 0, 0, 0)
        self.assertEqual(sequential_dr(row, 0), 2.)
        self.assertNotEqual(sequential_dr(row, 0), 1.)
        # The original complete-data fair mixture still has correct expectation.
        self.assertEqual(.5 * sequential_dr(row, 0), 1.)


if __name__ == '__main__': unittest.main()
