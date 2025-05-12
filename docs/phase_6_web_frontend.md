# Phase 6 – Public Web Front-End (User Portal)

Give business users a friendly browser interface to configure scenarios and launch projections without touching the CLI.

---
## 1  Tech Stack Decisions
| Layer | Choice | Rationale |
|-------|--------|-----------|
| Backend API | **FastAPI** | Already Python-based, async, swagger docs out-of-box |
| Front-End | **React 18 + Vite + MUI** | Familiar, fast HMR, rich component lib |
| Auth | **Auth0 / JWT (optional)** | Off-the-shelf, RBAC support |
| Hosting | Front-End → Netlify; API → Fly.io / Render | Simple CI deploy pipeline |

---
## 2  Backend MVC
Directory `cost_model/web_api/`
```
routers/
    scenarios.py   # CRUD for YAML configs
    runs.py        # launch & track projection jobs
    results.py     # serve summary CSV/JSON
services/
    jobs.py        # wrapper around subprocess / rq / celery
schemas.py         # pydantic models for request/response
main.py            # FastAPI app factory
```
Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/scenarios/ | create scenario (YAML upload or JSON) |
| GET  | /api/scenarios/{id} | get scenario |
| POST | /api/runs/ | launch projection; returns job_id |
| GET  | /api/runs/{job_id}/status | polling |
| GET  | /api/runs/{job_id}/results | download summary |

> **Tip:** use `uvicorn --reload` during dev, mount under `/api` so Netlify can proxy.

---
## 3  Front-End Pages
1. **Login / Landing** – OAuth flow, list recent runs.
2. **Scenario List** – table with edit, clone, delete.
3. **Scenario Editor** – Form-based YAML editor (monaco) + validation.
4. **Run Monitor** – progress bar, link to logs dashboard (Phase 7).
5. **Results Viewer** – render charts (plotly.js) & download buttons.

Routing example (React-Router v6):
```
/
/scenarios
/scenarios/:id/edit
/runs/:jobId
/runs/:jobId/results
```

---
## 4  Dev Experience
```
# spin up full stack
make dev             # starts uvicorn + Vite w/ proxy
make lint            # black + eslint
make test-frontend   # vitest
```
Add `.vscode/launch.json` to auto-attach frontend debugger.

---
## 5  CI/CD
- **GitHub Actions** workflow on `/webapp` changes:
  1. `npm ci && npm run build`
  2. Deploy to Netlify via API.
- API image built with `docker build` & pushed to GHCR; auto deploy on Render/Fly.

---
## 6  Definition of Done
- Users can log in, create/edit scenarios, launch a run, and download headline CSV.
- API documented via Swagger at `/docs`.
- Netlify build green on `main`.
