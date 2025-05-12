---
trigger: always_on
---

# .windsurfrules
# Agent-Specific Workspace Manifest for the cost-model Project
# Rules and structure guide for AI agents (e.g., Windsurf) to locate and modify functionality.

# === Core Simulation Logic ===
RULE: All simulation orchestration and control loops must live in cost_model/simulation.py.
RULE: Year-by-year forecasting logic must stay within cost_model/projections/; do not scatter projection code elsewhere.

# === Workforce State Management ===
RULE: Snapshot-related code resides exclusively under cost_model/state/snapshot.py and cost_model/state/snapshot_update.py.
RULE: Tenure calculations must remain in cost_model/state/tenure.py.
RULE: Event definitions and logging are only in cost_model/state/event_log.py; avoid modifying this without justification.
RULE: Schema definitions (column names/formats) are in cost_model/state/schema.py—always update here first.

# === Dynamic Population Modeling ===
RULE: Workforce dynamics logic (hires, terminations, transitions) must live in cost_model/dynamics/.
RULE: Hazard modeling (attrition/growth probabilities) belongs in cost_model/projections/hazard.py.

# === Plan Rules & Engine Logic ===
RULE: Business rules (eligibility, enrollment, contributions) are in cost_model/plan_rules/; do not hardcode business logic outside.
RULE: Modular engines (comp updates, event handlers) must be placed in cost_model/engines/.

# === Input Data & ML ===
RULE: Data ingestion and preprocessing should be in cost_model/data/ only.
RULE: Machine learning models (classification, predictions) live under cost_model/ml/.

# === Reporting and Analysis ===
RULE: Reporting utilities and summary exports go to cost_model/reporting/.

# === Configuration ===
RULE: All configuration parameters must be defined in YAML files under config/; avoid hardcoding parameters in code.
RULE: Prototyping notebooks belong in notebooks/ and should not be committed with large data outputs.

# === Output ===
RULE: Production simulation outputs must go to output/; use output_dev/ for debugging or test runs.

# === Testing & Tooling ===
RULE: All new functionality requires unit tests under tests/. Use pytest conventions.
RULE: CLI helper scripts belong in scripts/.
RULE: Keep pyproject.toml, requirements.txt, requirements-dev.txt, pytest.ini, and mkdocs.yml updated for dependencies and docs.

# === Agent Guidance ===
# - When modifying workforce state: consult snapshot.py and event_log.py first.
# - For business logic changes: use plan_rules/ and engines/.
# - To adjust attrition or growth rates: update projections/hazard.py only.
# - To parameterize runs: add or update YAML in config/.
# - Always add corresponding tests in tests/ and verify snapshot consistency.

# === General Principles ===
RULE: Avoid hardcoding values; prefer configuration.
RULE: Preserve schema consistency—update cost_model/state/schema.py before any data structure changes.
RULE: Follow project’s code style: line length <= 100, use black formatting, and enforce linting via ruff.
