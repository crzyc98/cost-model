# Phase 5 – Performance & Scaling

After correctness, make it fast & scalable.

---
## 1  Profiling
- Use `pyinstrument` on multi-year 50k employee run.
- Identify hotspots (likely `snapshot_update` & `summaries`).

---
## 2  Vectorization & Lazy Eval
- Replace row loops in `tenure.py` with vectorized `np.where`.
- Consider `polars` for snapshot frames behind optional flag.

---
## 3  Parallelization
- Use `ray` or `pandas`-mp on yearly loops; snapshots per year are embarrassingly parallel.
- Gate with `--workers N` CLI flag.

---
## 4  Caching
- Memoize `event_log.query_year()` results with `joblib.Memory`.

---
## 5  Definition of Done
- 5-year 100k employee projection < 30s on laptop.
- Memory footprint < 8GB.
