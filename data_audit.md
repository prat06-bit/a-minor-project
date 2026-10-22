# PMIS supplied data audit and experimental plan

**Audit date:** 2026-10-20  
**Workbook:** supplied `PMIS_MyBharat_Qualification_District_Mapping (1).xlsx`  
**Method:** pandas read-only extraction of every visible worksheet; no source workbook edits.  

## Executive findings

- Workbook has 3 visible sheets: `Main`, `States`, `Districts`.
- `Main` contains internship opportunity records and aggregate application counts, not candidate-level records.
- There is no candidate-to-internship outcome table and no confirmed training target. “Exact Matched Applications” is an aggregate count; it must not be reinterpreted as individual selection outcomes.
- `Internship ID` is unique across Main (1,062/1,062). Main has no fully duplicated rows.
- State and district aggregate sheets both sum to 17,766, but that total differs from Main row count and is not a row-level candidate or internship ID relation.
- Main arithmetic identity `Difference = No Of Opportunities - Exact Matched Applications` holds for all 1,062 rows. Applications Received exceeds listed capacity on 320 rows; Exact Matched Applications exceeds listed capacity on 244 rows. These are observed source values to preserve and explain, not silently “fix.”
- 676/1,062 Main rows have missing Specialization (63.7%).
- Supplied synopsis describes a prototype using simulated/sample data and places live government integration and production deployment outside scope. It requests explainable candidate matching, capacity constraints, fairness/policy layer, and frontend prototype.

## Worksheet audit

### Main — 1,062 rows × 20 columns

- Full-row duplicates: 0.
- Column details:

| Column | dtype | non-null | missing | unique | min / max or category note |
|---|---|---:|---:|---:|---|
| Internship ID | str | 1,062 | 0 | 1,062 | 1062 categorical/text values; see full frequency file |
| Application Start Date | str | 1,062 | 0 | 23 | 23 categorical/text values; see full frequency file |
| Application End Date | str | 1,062 | 0 | 23 | 23 categorical/text values; see full frequency file |
| Internship Type | str | 1,062 | 0 | 1 | 1 categorical/text values; see full frequency file |
| Company/Network Entity Name | str | 1,062 | 0 | 76 | 76 categorical/text values; see full frequency file |
| Relationship Type | str | 1,062 | 0 | 4 | 4 categorical/text values; see full frequency file |
| State / UT | str | 1,062 | 0 | 32 | 32 categorical/text values; see full frequency file |
| District | str | 1,062 | 0 | 286 | 286 categorical/text values; see full frequency file |
| Sector | str | 1,062 | 0 | 23 | 23 categorical/text values; see full frequency file |
| Area/Field | str | 1,062 | 0 | 17 | 17 categorical/text values; see full frequency file |
| Qualification | str | 1,062 | 0 | 6 | 6 categorical/text values; see full frequency file |
| Course | str | 1,062 | 0 | 30 | 30 categorical/text values; see full frequency file |
| Specialization | str | 386 | 676 | 32 | 32 categorical/text values; see full frequency file |
| Internship Posted On | str | 1,062 | 0 | 1,015 | 1015 categorical/text values; see full frequency file |
| No Of Opportunities | int64 | 1,062 | 0 | 23 | 1 / 150 |
| Applications Received | int64 | 1,062 | 0 | 66 | 0 / 842 |
| Exact Matched Applications | int64 | 1,062 | 0 | 64 | 0 / 430 |
| Difference | int64 | 1,062 | 0 | 99 | -419 / 146 |
| Recommended Institutions | str | 1,062 | 0 | 447 | 447 categorical/text values; see full frequency file |
| Institution Types | str | 1,062 | 0 | 3 | 3 categorical/text values; see full frequency file |

**Main numeric ranges:**
- `No Of Opportunities`: min 1, max 150, distinct values 23.
- `Applications Received`: min 0, max 842, distinct values 66.
- `Exact Matched Applications`: min 0, max 430, distinct values 64.
- `Difference`: min -419, max 146, distinct values 99.

**Sample rows:**

```text
  Internship ID  Application Start Date     Application End Date Internship Type Company/Network Entity Name          Relationship Type     State / UT    District                     Sector                 Area/Field Qualification                 Course Specialization    Internship Posted On  No Of Opportunities  Applications Received  Exact Matched Applications  Difference                                                                                                                                                                                                                                                                                                                                                                                                                   Recommended Institutions Institution Types
PMIS-2026-58304 August 5, 2026 at 00:00 August 31, 2026 at 23:59         Regular HYUNDAI MOTOR INDIA LIMITED                       Self     TAMIL NADU Kanchipuram                 Automotive Production / Manufacturing       Diploma Mechanical Engineering            NaN August 5, 2026 at 09:52                  150                     13                           4         146             Government Industrial Training Institute, Oragadam; P T Lee Chengalvaraya Naickerprivate Industrial Training Institute; Anbu Private Industrial Traning Institute; Pallavan Private Industrial Training Institute; Wallajabad Private Industrial Training Institute; Meenakshi Ammal Private Industrial Training Institute; Kamatchi Private Industrial Training Institute; Annai Indira Private Industrial Training Institute               ITI
PMIS-2026-59342 August 7, 2026 at 00:00 August 23, 2026 at 23:59         Regular           BLUE STAR LIMITED Subsidiary / Group Company ANDHRA PRADESH    Tirupati Manufacturing & Industrial Production / Manufacturing          10th           All Subjects            NaN August 7, 2026 at 19:34                   90                     12                          12          78 T.T.D.S Sri Venkateswara Training-cum-Production Centre P/H; Govt Industrial Training Institute; Govt. Industrial Training Institute,Tada; Govt Industrial Training Institute for Girls,Chittedu, Nellore; Govt Industrial Training Institute,Venkatagiri; Govt. Industrial Training Institute, Vakadu; Govt District Level Industrial Training Institute, Gudur; O.V.R.ITI; Kanji Ramachandra Reddy Industrial Training Centre; Modal ITC               ITI
```

### States — 33 rows × 2 columns

- Full-row duplicates: 0.
- Column details:

| Column | dtype | non-null | missing | unique | min / max or category note |
|---|---|---:|---:|---:|---|
| State | str | 33 | 0 | 33 | 33 categorical/text values; see full frequency file |
| Internships | int64 | 33 | 0 | 32 | 1 / 8883 |

**Sample rows:**

```text
        State  Internships
UTTAR PRADESH         1634
   TAMIL NADU         1317
```

### Districts — 287 rows × 3 columns

- Full-row duplicates: 0.
- Column details:

| Column | dtype | non-null | missing | unique | min / max or category note |
|---|---|---:|---:|---:|---|
| State / UT | str | 287 | 0 | 33 | 33 categorical/text values; see full frequency file |
| District | str | 286 | 1 | 286 | 286 categorical/text values; see full frequency file |
| Internships | int64 | 287 | 0 | 70 | 1 / 8883 |

**Sample rows:**

```text
    State / UT District  Internships
    TAMIL NADU  Chennai          571
ANDHRA PRADESH Tirupati          540
```

## Relationship and quality checks

- Main `Internship ID`: 1,062 distinct; no duplicate IDs.
- Main company/entity values: 76 distinct; repeats expected across multiple postings.
- Main state/district observed pairs: 286; all observed pairs appear in the Districts reference.
- Main states all appear in States reference.
- Districts contains 287 rows, 286 nonmissing districts; one row has missing District (inspect full category frequency file).
- States has 33 rows; Districts has 287 rows; Main has 1,062 rows.
- State and District aggregate `Internships` sums both equal 17,766; these are aggregates, not join keys to Main rows.
- Application start/end/post dates parse for all rows; no start-after-end records.
- `Exact Matched Applications <= Applications Received` for all rows.
- `Applications Received > No Of Opportunities` for 320 rows; `Exact Matched Applications > No Of Opportunities` for 244 rows. Interpretations/definitions should be confirmed before treating these as errors.
- Repeated company + state + district + sector + qualification + course + specialization tuples: 185; not necessarily duplicates because internship IDs remain unique and dates/type may distinguish postings.
- No observed demographic, skills, candidate preference, prior participation, applicant decision, or actual allocation columns.

## Target and source limitations

- No candidate-level target is available. Aggregate counts must not be expanded into fabricated applicant outcomes.
- Build synthetic candidate–internship pairs only as explicitly synthetic experiments. Heuristic labels must be named `synthetic_proxy_label` and defined by a documented scoring rule.
- Source rows and source values remain unchanged and separately stored; synthetic and corrupted copies must have provenance and lineage.
- Categories in the complete value-count form are enumerated in `categorical_distributions.csv`; this captures exact capitalization/spelling as observed.

## Proposed schema and generation strategy

### Source layer

- Preserve an immutable copy of the supplied workbook or read it in place. In pipeline outputs, retain source worksheet and source row identifiers with `record_origin=observed_source`; do not overwrite or normalize the source values.
- Treat Main as opportunity catalog (`internships`), with a stable internship ID and the observed capacity, location, sector, area, qualification/course/specialization, application window, and aggregate counts. Treat States/Districts as aggregate reference tables, not applicant pools.
- Keep a normalized mapping/alias layer for category standardization; preserve original labels next to canonical labels.

### Experimental entities

- `synthetic_candidates`: synthetic candidate ID, qualification/course/specialization, skills, sector interests, state/district preferences, past-participation proxy, seed, generation rule/version, and per-field provenance (`generated` or `inferred`). No sensitive social category by default.
- `synthetic_internships`: copied/selected opportunity records with source internship ID, synthetic instance ID when variation is applied, and provenance. Capacity must be nonnegative and locations/qualification combinations tied to observed valid source records.
- `synthetic_candidate_internship_pairs`: unique pair ID, candidate ID, internship ID, eligibility flags, component match features, heuristic `synthetic_proxy_label`, label-rule version, and provenance. Labels derive only from transparent synthetic rules, never from aggregate “Exact Matched Applications.”
- Separate clean, corrupted, cleaned, and split manifests. Include seed, row counts, schema version, source hash, generator version, and synthetic/corruption parameters.

### Generation

- Sample observed qualification/course/specialization and location/sector proportions, while respecting observed hierarchy and valid tuples. Use controlled variation (e.g., skill templates and preference distributions) rather than copying source rows as independent applicants.
- Generate internship opportunity scenarios from valid source combinations and keep `capacity` distinct from demand/application counts.
- Generate candidate preferences and varying fit levels with seeded random processes. Generate labels from a documented rule combining eligibility and preference fit; maintain a label rationale and avoid claiming real selection accuracy.
- Keep sensitive attributes out of ordinary merit scores. If a fairness experiment is included, use synthetic-only policy attributes in a separate policy layer and report policy constraints separately from suitability.

## Corruption and cleaning experiments

- Produce an untouched clean synthetic baseline and separate corrupted copies at configurable 0%, 5%, 10%, and 20% rates.
- Realistic ingestion simulations: nulls, whitespace/case drift, common label aliases/typos, numeric strings, duplicate delivery records, and unknown categories.
- Artificial stress tests: out-of-range values, contradictory state/district or qualification/course pairs, extreme application-to-capacity ratios, feature noise, and controlled proxy-label imbalance. Mark each stress test explicitly.
- Never corrupt labels used for evaluation. Preserve the original row/field in a change log or a paired `original_value` field; log record IDs, fields, transforms, rates, seed, and before/after row counts.
- Preprocessing fits mappings/imputers/encoders/scalers on training folds only. Keep rejected rows and reason codes in an audit output; resolve duplicates with stable-ID rules and retain a duplicate-resolution log.

## Validation and staged implementation plan

1. **Stage 1 — audit and contract:** completed here; freeze source facts, identify absent candidate outcomes, and define schemas/label caveats.
2. **Stage 2 — reproducible generator:** implement configurable seeded generation, data lineage, relationship validation, and clean synthetic CSV outputs.
3. **Stage 3 — corruption and preprocessing:** generate corruption levels with reversible logs; implement train-only preprocessing and rejected-row audit.
4. **Stage 4 — matching and baselines:** separate hard eligibility from scores; rule baseline, logistic regression, random forest, and ranking baseline; omit complex models unless the synthetic data supports comparison.
5. **Stage 5 — evaluation:** candidate-grouped train/validation/test splits; clean vs corrupted vs cleaned, data-size and complexity comparisons; classification/ranking metrics, runtime, capacity and fairness checks. Clearly qualify all metrics as proxy-label agreement.
6. **Stage 6 — allocation/explanations and project docs:** capacity-aware optimization, reason traces, model card, README, and reports. Any affirmative-action rules remain configurable, policy-governed and separately audited.

**Important evaluation constraint:** Grouping by candidate is necessary but can still allow the same internship catalog records across partitions. Add a held-out internship/company generalization experiment where feasible. Do not describe these results as real-world selection performance.
