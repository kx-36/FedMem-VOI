"""Hand-written synthetic inputs only. No benchmark, service, or model calls."""
from fedmem_voi_example.reference import Outcome, masked_softmax, paired_endpoints, uniform_logging

mask = {'skip': True, 'refine_note': True, 'forbidden_note': False}
print('SYNTHETIC arithmetic demonstration; no measured scientific results')
print('Known logger:', uniform_logging(mask))
print('Target probabilities:', masked_softmax({'skip': 0., 'refine_note': .7, 'forbidden_note': 9.}, mask))
full = {'synthetic_client': {'task_1': Outcome('observed', 1., None),
                             'task_2': Outcome('Unknown', None, 'synthetic missing record')}}
comparator = {'synthetic_client': {'task_1': Outcome('observed', 0., None),
                                   'task_2': Outcome('observed', 1., None)}}
print('Paired identification bounds (not a confidence interval):',
      paired_endpoints(full, comparator, {'synthetic_client': 1.}))

from fedmem_voi_example.reference import CohortPlan, procedure_decision
plan = CohortPlan((('planned_slot_1', 'synthetic_client', 'learning_block'),), 10., ('Full', 'ACO'))
print('Synthetic absent planned feedback slot:')
print(procedure_decision(plan, {}, boundary_time=10., phase='before_fit',
                         technical_flags={}, evaluation_started=False))
print('Routing only: no Local execution or fabricated fallback outcome.')
