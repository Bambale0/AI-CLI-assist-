# AGENTS.md — Repository Instructions

---

## Shared Engineering Baseline — Start + AuRoom

This shared baseline supplements repository-specific rules; it never replaces stricter local architecture, release, security, channel, or product constraints.

### Engineering playbook and task flow
- Treat `Bambale0/skills` as the primary engineering playbook. Also inspect relevant safe guidance from `Bambale0/claw` and `anthropics/skills`.
- Do not use deprecated skills. Use in-progress skills only when they fit and account for their experimental status.
- Large ambiguous work: use a wayfinder-style flow.
- Feature development where applicable: `grill-with-docs → to-spec → to-tickets → implement → tdd → code-review`.
- Debugging: diagnose from evidence first (logs, telemetry, DB/runtime state, reproducible behavior), then patch.
- Never claim tests, CI, deploy, or production state that was not actually verified.

### Mandatory feature preflight and CONTEXT ledger
Before implementing any material feature or cross-cutting refactor, perform a fresh audit of the current repository state. Inspect relevant docs/specs/ADRs, code, schemas/migrations, auth, admin/config surfaces, tests, CI, integrations, and runtime telemetry when available.

Use the repository-designated execution ledger for active work. If `CONTEXT.md` is explicitly documented as that ledger, maintain it. If `CONTEXT.md` already serves another purpose, do not repurpose it; use an existing repository-local ledger path or create `docs/agents/EXECUTION.md`. Record:
- baseline commit/SHA and current state;
- what exists / is partial / is missing / can be reused;
- risks, dependencies, migrations, integrations, permissions and rollout impact;
- intended user-visible outcome and acceptance criteria;
- no-hardcode/configuration decisions;
- observability plan;
- test seams and unit/integration/contract/E2E/smoke plan;
- numbered implementation steps with progress evidence;
- final verification and follow-ups.

Do not reconstruct the ledger only at the end.

### No hardcode and control plane
Mutable business/runtime behavior must not require source edits, manual SQL, or redeploys. Values such as prices, tariffs, categories, statuses, SLA, prompts, provider/model selection, routing, thresholds, schedules, feature availability, notification templates, retry/fallback policy, permissions, and integration mappings should normally be typed, validated, database-backed, scoped, auditable, and manageable through the appropriate authenticated admin/control plane.

Secrets are not business configuration. Keep tokens, credentials, private keys, and deployment secrets in an appropriate secret mechanism. Never expose plaintext secrets in frontend bundles, logs, API responses, Git, or ordinary database settings.

### Architecture and integrations
- Prefer a modular monolith with explicit module interfaces and seams unless scaling, security, reliability, or ownership evidence justifies extraction.
- Important cross-module state changes should use explicit, typed, versionable, traceable, retry-safe/idempotent events where eventing is appropriate.
- Keep provider-specific HTTP payload handling behind typed integration adapters/ports, not in domain/application logic.
- External integrations must define authentication, finite timeouts, bounded retries/backoff, rate-limit behavior, idempotency, webhook verification where supported, reconciliation, data ownership/sync direction, observability, and failure semantics.
- Avoid parallel sources of truth. Keep ownership of authoritative external data with the actual system of record.

### Security and AI authority
Authorization is enforced server-side. UI hiding is never sufficient. Preserve ownership/tenant boundaries where applicable and treat data leakage as a release blocker.

AI may classify, summarize, extract, recommend, and execute only explicitly permitted workflows. AI must not bypass authorization, approvals, deterministic validation, financial controls, legal signing, or tenant/data isolation. Low-confidence or high-impact actions should fail closed or escalate to a human.

### Observability first
Logging and telemetry are part of the implementation, not follow-up polish. Critical paths should make it possible to determine what happened, when, for which actor/entity/scope, through which provider, how long it took, whether retries occurred, why it failed, and what the user saw. Propagate useful request/trace/correlation IDs. Never log secrets or unnecessary personal data.

### Test-first vertical slices and completion gate
Prefer tracer-bullet slices: `failing behavior test → minimal implementation → focused checks → next slice`.

For every material feature, explicitly decide and verify where applicable:
- unit/domain behavior;
- repository/database integration and migrations;
- authorization/ownership/tenant isolation;
- provider contract behavior;
- workflow transitions, idempotency and retry behavior;
- API integration;
- browser/bot E2E happy path and critical failure/permission path;
- production-like smoke/deployability;
- observability/audit;
- admin configurability / no-hardcode.

A regression fix should get a regression test whenever technically feasible. Do not mark work complete until applicable acceptance criteria and focused/broader checks pass; when repository CI exists and is accessible, it is green for the exact commit; code review against repository standards and the originating spec is complete; and no unresolved high-severity finding remains. If CI is unavailable or the repository has no CI, record that explicitly and run the closest available local checks instead.

### Delivery
Final engineering reports should state: what changed; important files/components; skills/flows used; exact tests/checks and results; migrations/config/admin changes; risks/follow-ups; and PR/commit/deploy SHA when applicable.
