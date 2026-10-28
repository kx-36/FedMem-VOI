import itertools
import math
import unittest
from dataclasses import replace
from fedmem_voi_example.reference import (
    Assignment, Outcome, Stage, cohort_decision, discovery_guard, feasible_mask,
    masked_softmax, paired_endpoints, sequential_dr, uniform_logging,
)


def stage(action='skip', target=None, critic=None):
    mask = {'skip': True, 'query': True}
    return Stage(mask, action, uniform_logging(mask), target or {'skip': .3, 'query': .7},
                 critic or {'skip': -.4, 'query': .9})


def local(outcome):
    dummy = Stage({'dummy': True}, 'dummy', {'dummy': 1.0}, {'dummy': 1.0}, {'dummy': 17.0})
    return Assignment(False, outcome, None, dummy, dummy, 'skip', 'withhold', 0, 0, 0)


class ReferenceTests(unittest.TestCase):
    def test_enumerated_sequential_dr_consistency(self):
        # Exhaust all b,z,a,y trajectories, including failed transports and native
        # zero outcomes. Nontrivial transitions, masks, costs, arbitrary critics.
        lam = .13
        expectation, target_value, total_mass = [], [], []
        for b, z in itertools.product(('skip', 'query'), (0, 1)):
            transition = (.2 if b == 'skip' else .7) if z else (.8 if b == 'skip' else .3)
            mask = {'withhold': True, 'payload': z == 1}
            p1 = uniform_logging(mask)
            pi1 = {'withhold': .4, 'payload': .6} if z else {'withhold': 1., 'payload': 0.}
            s0 = stage(b)
            for a in mask:
                if not mask[a]:
                    continue
                mu = .1 + .2 * (b == 'query') + .2 * z + .3 * (a == 'payload')
                for y in (0, 1):
                    py = mu if y else 1 - mu
                    s1 = Stage(mask, a, p1, pi1, {'withhold': -.2, 'payload': 1.7})
                    row = Assignment(True, Outcome('observed', y, None), (('note', 'v1'),),
                                     s0, s1, 'reply' if z else 'transport_failure',
                                     'transport_failure' if a == 'payload' else 'withhold',
                                     .2, .3 * (b == 'query'), .4 * (a == 'payload'))
                    mass = .5 * transition * p1[a] * py
                    total_mass.append(mass)
                    expectation.append(mass * sequential_dr(row, lam))
                    target_mass = s0.target[b] * transition * pi1[a] * py
                    target_value.append(target_mass * (y - lam * (.2 + row.refinement_cost + row.payload_cost)))
        self.assertAlmostEqual(math.fsum(total_mass), 1)
        self.assertAlmostEqual(math.fsum(expectation), math.fsum(target_value), places=12)
        # Full guarded population combines forced-Local and acquisition units.
        self.assertAlmostEqual(.4 * sequential_dr(local(Outcome('observed', .8, None)), lam)
                               + .6 * math.fsum(expectation), .4 * .8 + .6 * math.fsum(target_value))

    def test_guard_and_probability_one_local(self):
        self.assertTrue(discovery_guard(4, 4))
        self.assertFalse(discovery_guard(5, 4))
        self.assertEqual(sequential_dr(local(Outcome('observed', .7, None)), 9), .7)
        with self.assertRaises(ValueError):
            replace(local(Outcome('observed', 0, None)), discovery_cost=1)
        with self.assertRaises(ValueError):
            replace(local(Outcome('observed', 0, None)), candidates=())

    def test_mask_and_softmax(self):
        mask = feasible_mask({'skip': True, 'query': True, 'forbidden': False},
                             {'skip': 0, 'query': 5, 'forbidden': 0}, 2, 6)
        self.assertEqual(mask, {'skip': True, 'query': False, 'forbidden': False})
        self.assertEqual(masked_softmax({'skip': -5, 'query': 100, 'forbidden': 100}, mask),
                         {'skip': 1., 'query': 0., 'forbidden': 0.})
        self.assertEqual(uniform_logging({'a': True, 'b': True}), {'a': .5, 'b': .5})
        with self.assertRaises(ValueError):
            masked_softmax({'a': 0, 'b': -10000}, {'a': True, 'b': True})
        with self.assertRaises(ValueError):
            feasible_mask({'skip': True}, {'skip': 0}, 7, 6)

    def test_unknown_never_training_label(self):
        unknown = Outcome('Unknown', None, 'evaluator unavailable')
        with self.assertRaises(ValueError):
            unknown.training_label()
        with self.assertRaises(ValueError):
            sequential_dr(local(unknown), 0)
        with self.assertRaises(ValueError):
            Outcome('Unknown', 0, 'missing record')
        self.assertEqual(Outcome('observed', 0, None).training_label(), 0)
        result = cohort_decision([local(Outcome('observed', 0, None)), local(unknown)])
        self.assertEqual(result['assigned_count'], 2)
        self.assertEqual(result['decision'], 'reject_whole_cohort')

    def test_paired_unknown_bounds_and_denominators(self):
        yes, no = Outcome('observed', 1, None), Outcome('observed', 0, None)
        unknown = Outcome('Unknown', None, 'missing terminal record')
        full = {'a': {'1': yes, '2': unknown}, 'b': {'3': no}}
        comp = {'a': {'1': no, '2': yes}, 'b': {'3': unknown}}
        self.assertEqual(paired_endpoints(full, comp, {'a': .5, 'b': .5}), (-.5, .25))
        with self.assertRaises(ValueError):
            paired_endpoints(full, {'a': {'1': no}, 'b': {'3': unknown}}, {'a': .5, 'b': .5})

    def test_transport_failure_not_unknown_or_dropped(self):
        row = Assignment(True, Outcome('observed', 0, None), (('note', 'v1'),), stage(), stage(),
                         'transport_failure', 'transport_failure', 1, 2, 3)
        delivered = replace(row, refinement_status='unresolved', delivery_status='delivered')
        self.assertEqual(sequential_dr(row, .2), sequential_dr(delivered, .2))
        self.assertTrue(cohort_decision([row])['eligible_for_complete_data_score'])

    def test_missing_invalid_and_unsupported_inputs_refused(self):
        with self.assertRaises(TypeError):
            Outcome('Unknown', None)
        with self.assertRaises(ValueError):
            replace(stage(), logged={'skip': .9, 'query': .1})
        with self.assertRaises(ValueError):
            replace(stage(), target={'skip': 0., 'query': 0.})
        with self.assertRaises(ValueError):
            Outcome('observed', float('nan'), None)
        with self.assertRaises(ValueError):
            cohort_decision([])


if __name__ == '__main__':
    unittest.main()
