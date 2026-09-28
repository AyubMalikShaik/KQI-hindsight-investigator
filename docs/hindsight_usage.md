# Hindsight usage in Lumen

How this project uses Hindsight, why each call is shaped the way it is, and what
breaks if you change it.

Everything goes through `MemoryService` (`memory/memory_service.py`). The rest of
the application knows only `retain` / `recall` / `reflect`, so Hindsight SDK
details never leak into the agent.

---

## 1. What memory is allowed to do

Memory is a **prior, not truth**. It changes how confident the agent is in a
candidate and which of two competing candidates it examines first. It never
supplies a finding: every mechanism a recalled case mentions still has to be
confirmed by a tool run against current data.

That constraint is enforced in scoring, not just by convention
(`validation/confidence.py`):

```
confidence = 0.40 · evidence_support
           + 0.25 · magnitude_explained
           + 0.20 · temporal_alignment
           + 0.15 · memory_prior
```

Two ceilings make this real:

* Non-memory signals cap at **0.85**. A hypothesis cannot reach the
  `root_cause_found` threshold on memory alone.
* `memory_prior` itself caps at `relevance × 0.4` unconfirmed, `× 0.8` when the
  hypothesis is independently corroborated by current evidence. So memory's
  maximum total lift is `0.4 × 0.15 = 0.06` unconfirmed.

If you raise the memory weight, you are buying confidence you did not verify.

---

## 2. The three calls

### `retain` — two stages, one document id

| Stage | Context constant | When | Tag |
|---|---|---|---|
| 1 | `STAGE_UNCONFIRMED = "agent_report"` | at the end of every run | `stage:agent_report`, `status:unconfirmed` |
| 2 | `STAGE_CONFIRMED = "human_feedback"` | when an analyst files feedback | `stage:human_feedback`, `verdict:<v>` |

Both stages write to the **same** `document_id` with `update_mode="replace"`.
That is the whole point: recall must never serve an investigator the agent's
original guess and its correction side by side.

Stage 1 happens even though nobody has judged the conclusion yet, because a
conclusion that is not written down cannot be corrected. Stage 2 supersedes it.

Every retain also carries `alert:<alert_id>` — see §4.

```bash
# stage 1 is automatic
python -m run_investigation --scenario SCN-001

# stage 2
python -m scripts.feedback --list
python -m scripts.feedback INV-20260928-8bae06 --verdict confirmed \
    --cause "upi-sdk 4.12.0 keepalive regression" \
    --action "rolled back to 4.11.1" --owner platform-eng
```

### `recall` — twice per investigation, anchored in time

Called at `after_scoping` (by signature: metric, direction, size, date) and
`after_evidence` (by the mechanisms the tools actually surfaced).

`query_timestamp` is set to the **alert's** detection date, not server time.
Without it Hindsight scores recency against now, which ranks backfilled
historical cases as stale and makes the learning curve meaningless.

`budget` and `max_tokens` come from `BudgetConfig`; recall is bounded like every
other call in the system.

### `reflect` — after the human, not before

`reflect()` runs only once feedback has been filed, asking what the confirmation
means for the *next* similar case. Its output is written to
`artifacts/reports/<trace>.reflection.txt`.

It is deliberately **not** retained into the bank. Echoing a synthesis of an
already-retained case back in would hand recall a second copy of the same
incident and inflate the next investigation's prior.

---

## 3. How a prior attaches to a hypothesis

This is where most of the engineering is, and it is easy to get subtly wrong.

`_attach_memory_priors` (`agent/orchestrator.py`) scores each recalled memory
against each candidate hypothesis and cites the ones above
`MIN_RELEVANT_PRIOR = 0.3`.

Relevance has three parts:

1. **Cause-type gate.** A traffic-loss case cannot back a config hypothesis.
2. **Generic-term filter.** `GENERIC_TERMS` stops common words from matching
   everything. Without it a "sessions" hypothesis and a "revenue" hypothesis
   both match the same case and neither is preferred.
3. **Mechanism identifiers** (`identifier_tokens` / `mechanism_tokens`). Only
   the hypothesis statement and `affected_segments` values are searched —
   `predicted_evidence` is excluded, because it describes what to look for, not
   what the cause *is*.

Observed metrics are filtered out of the identifier set (`OBSERVED_METRICS`
from `data/catalog.py`). Every sibling hypothesis names `payment_success_rate`,
so treating it as a discriminator hands all of them the same prior and
distinguishes nothing. Measurements like `25.6` are filtered too; version
strings like `4.12.0` are not.

**Why this matters:** if two candidates share a cause type and receive identical
priors, memory moves confidence but cannot change which one wins. That is
indistinguishable from memory not working. `tests/test_memory_priors.py` guards
this directly.

A prior only counts as *help* if the hypothesis carrying it survived validation
on current evidence. Everything else is reported as "recalled but unused", which
is the honest answer.

---

## 4. Never recall your own report

An investigation must not read back the report written about the alert it is
currently working — otherwise its own conclusion becomes the prior that confirms
it.

`retain_report` / `retain_feedback` tag every document `alert:<alert_id>`, and
`recall(..., exclude_alert=alert_id)` drops anything carrying the current
alert's tag while leaving the rest of the bank untouched.

This is not theoretical. Before it existed, a test run's report was recalled as
a prior for the same incident, and `tests.test_scenarios` went red because the
agent had begun confirming itself.

`check_retain_tags_the_alert_it_belongs_to` in `tests/test_plumbing.py` covers
both halves: the tag is present on both retain stages, and the filter removes
only the tagged memory.

---

## 5. Backfilled history

`scripts/seed_bank.py` seeds eight closed cases from *other* incidents, each
dated before the incident it informs, so recall can only ever supply a lead.

They are deliberately hard for the memory to use naively:

* two payment cases in the same market, so the memory has to discriminate
  rather than match keywords;
* `case-2026-06-15-upi-keepalive-push` exists to decide *within* one scenario —
  SCN-002 carries two competing config changes in the same window, and this case
  is what tells the agent which key to suspect.

```bash
python -m scripts.seed_bank --list
python -m scripts.seed_bank --reset
```

Do not seed a case describing the incident under investigation. Recall then
just hands the agent its own answer, and the memory score becomes circular.

---

## 6. Measuring whether it works

| Command | What it answers |
|---|---|
| `python -m tests.compare_memory_ab` | Does memory change the answer? |
| `python -m eval.learning_curve` | Does more history make it better? |
| `python -m tests.show_memory [trace]` | What did this run actually recall? |
| `streamlit run app/app.py` | On screen: recalled cases, priors, before/after |

The A/B runner refuses to report a win unless both arms ran the **same**
investigation — same candidate count, same evidence items, same tool calls. If
memory-off gathered less, the diff would be measuring a handicapped baseline
rather than the prior. Memory-off runs the identical generator and scorer and
forms the same candidate set; only the recalled priors are absent.

The learning curve seeds a bank per size and walks it up. Cases enter in date
order, because that is the only order they ever arrive in.

Current results:

| Scenario | Memory off | Memory on | Effect |
|---|---|---|---|
| SCN-001 | `payment_failure` 0.850 | `release_regression` 0.899 | **conclusion changes** |
| SCN-002 | `conn_pool_max` 0.850 | `keepalive_ms` 0.892 | **mechanism changes** |
| SCN-003 | `data_quality` 0.530 | `data_quality` 0.576 | confidence only |
| SCN-004 | `traffic_loss` 0.770 | `traffic_loss` 0.812 | confidence only (negative control) |

SCN-002 is the interesting one: both candidates are `config_change`, the decoy
change is *closer in time* to symptom onset than the causal one, so an
evidence-only run picks the decoy on recency. Memory is the only thing that
separates them.

---

## 7. Configuration

```
HINDSIGHT_BASE_URL=https://api.hindsight.vectorize.io
HINDSIGHT_API_KEY=...
HINDSIGHT_BANK=anomaly-investigator::acme
```

Bank name comes from `Settings.bank`. The learning curve and curve banks use
their own ids (`lumen-curve-<n>`) so they never touch the seeded bank.

DNS to the Hindsight endpoint fails intermittently (`ClientConnectorDNSError`).
It recovers. Every call site treats that as *memory unavailable* and continues —
the investigation still returns grounded, cited findings, and
`recall_failed: true` is reported separately so an outage is never mistaken for
memory that was reachable and did not help.

---

## 8. Invariants worth keeping

1. Memory is a prior. Current tools verify every recalled mechanism.
2. Both retain stages share one `document_id` with `update_mode="replace"`.
3. `recall` passes `query_timestamp` anchored to the alert.
4. Nothing tagged with the current `alert_id` comes back as its own prior.
5. Priors are keyed per mechanism, not per cause category.
6. `reflect()` output is not retained into the bank.
7. Non-memory signals cap at 0.85; memory's lift is bounded and visible.
8. A memory-off comparison arm must be exactly as capable as the memory-on one.
