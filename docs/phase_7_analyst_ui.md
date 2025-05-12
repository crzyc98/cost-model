# Phase 7 – Analyst Advanced UI (Live Log & Metrics)

Provide power-users with real-time insight into long-running projections and deep-dive analytics.

---
## 1  Requirements
* Live tail of `projection.log` with filtering (module / level).
* Streaming KPI charts during execution (headcount, terminations processed/sec).
* Ability to download intermediate snapshots.

---
## 2  Architecture
```
┌──────────────┐    WebSocket    ┌───────────────┐
│ Projection   │ ──────────────▶ │ FastAPI WS    │ ─── HTTP  ──▶ React
│ subprocess   │    log lines    │ /ws/logs/{id} │   API         UI
└──────────────┘                 └───────────────┘
```
Implementation steps
1. In `services/jobs.py`, spawn projection with `subprocess.Popen(stdout=PIPE, stderr=PIPE)`.
2. Create `asyncio.Queue` per job; background task pulls from process stdout and pushes lines to queue.
3. WebSocket endpoint `/ws/logs/{job_id}` awaits `queue.get()` and `yield` to client.
4. Front-end **LogViewer** component connects via native WebSocket; renders lines using `react-virtualized` for perf.

---
## 3  Streaming Metrics
* Inject `--metrics-port` flag when launching run; metrics server (`prometheus_client.start_http_server`) exposes counters.
* FastAPI proxy fetches `/metrics` and parses a subset for UI.
* Charts update via SSE (`/api/runs/{id}/metrics/stream`).

---
## 4  Intermediate Snapshot Download
Add API `/api/runs/{id}/snapshots/{year}` that zips the parquet snapshot after each yearly write.
Users can click to explore in pandas.

---
## 5  Front-End Components
| Component | Library |
|-----------|---------|
| LogViewer | `react-virtualized`, `useWebSocketHook` |
| MetricsDashboard | `recharts` or `visx` |
| SnapshotModal | monaco diff for schema compare |

---
## 6  Definition of Done
- Analysts can watch logs & live metrics without SSH.
- No noticeable lag for 1000 lines/s.
- Snapshots downloadable mid-run.
