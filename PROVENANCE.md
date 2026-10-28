# Source and license provenance

New original example files were authored for this bounded public-release
candidate. No third-party source code was copied. Python standard-library imports
are dependencies, not bundled code. The newly written original reference example is distributed under the
Apache License 2.0 in `LICENSE`, preserving the repository owner's existing
license choice. The license bytes match the existing repository Git blob
`261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`. This grant does not cover excluded legacy material,
third-party sources, or the private manuscript.

Conceptual arithmetic was transcribed from the private V8 prospective manuscript:
Eq. (10), the complete guarded-policy appendix and the operational paired-bound
definitions. Those manuscript files are NOT distributed. Local source identifiers
and SHA-256 digests at inspection are recorded below solely for provenance:

| Source | SHA-256 | Disposition |
|---|---|---|
| V8 manuscript.tex | `6c5df00c29db53d85ec08bd02b326840f562c5c167604f4a87fea03fbf37cb67` | Formulas only, manuscript excluded |
| V8 appendix_design.tex | `75c4d37d2082571074b7631ecd584732aab83b9feacee4648ea203a50723c5e3` | Guard semantics only, source excluded |
| V8 appendix_operational.tex | `f8932012143cd23db890d38dbada56f161ecbc380de9eb34f40e2f91675a7c75` | Paired endpoint semantics only, source excluded |
| experiments/fedmemrl/shared_linear.py | `dfafa4430fa847f6a9d6a533994229de052b33393750443b9793037e04ad1fc1` | Inspected numerical baseline; NOT copied |

No root or experiments-level LICENSE file was found in the bounded license
check. The legacy numerical file contains no copyright/license header. Its
ownership and redistribution permission therefore remain unestablished. It uses
NumPy and implements linear local workers / synchronous FedAvg simulation, not
the current proposed two-stage acquisition learner. It is deliberately not
relicensed or included verbatim. No verified upstream repository URL for that
file was established; no upstream provenance is invented.

Version control metadata for the project root was unavailable for this task;
these content hashes, rather than a fabricated commit ID, anchor inspection.
The payload's `SHA256SUMS` covers all public files except itself. Local build
verification and the ZIP digest are outside the public payload.

The registered completeness/fallback decision follows the parent-adopted V9
protocol audit: common cohort eligibility, method-specific technical validity,
pre-evaluation binding, and no conditional-on-acceptance DR identity. The audit
and private author brief are not included in the public payload. This is a new
prospective reference choice, not a relabeling of historical experiments.
