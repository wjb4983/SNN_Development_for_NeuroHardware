# SNN Expansion Roadmap

This roadmap outlines a phased plan to expand SNN backend support, deployment maturity, and hardware-readiness evaluation across the project.

## Phase 1 (2–4 weeks): Foundation and Compatibility

### Scope
- Native backend adapters for existing 3 libraries.
- Capability matrix documentation.
- Minimal hardware constraints checker.

### Entry Criteria
- Current experiment runner and reporting flow are stable for baseline library integrations.
- Existing backend abstraction points are identified and documented.
- Baseline test corpus is passing in CI for current adapters.

### Exit Criteria
- Three native backend adapters are implemented and accessible via a unified interface.
- A capability matrix document is published with feature-level backend support status.
- A minimal hardware constraints checker validates core limits (memory, timestep budget, and supported neuron/synapse primitives).
- Example experiment manifests can run against all supported backends without manual code edits.

### Required Tests
- Unit tests for each adapter’s translation and execution hooks.
- Contract tests validating common backend interface behavior across all three libraries.
- Validation tests for constraints checker boundary conditions and error messaging.
- Smoke integration tests proving at least one reference task runs per backend.

### Artifact Expectations
- `snn_bench/backends/` adapter modules for all three target libraries.
- Capability matrix documentation under `docs/` with an explicit feature grid.
- Hardware constraints checker module plus CLI/API entrypoint.
- CI updates adding adapter and checker test coverage.

---

## Phase 2 (4–8 weeks): Deployment and Comparative Experimentation

### Scope
- Lava deployment pipeline.
- Quantization + export reports.
- New experiment manifests with backend comparison leaderboards.

### Entry Criteria
- Phase 1 exit criteria are completed and merged.
- Backend adapters are stable enough for repeated benchmark execution.
- Reporting pipeline can ingest per-run metadata and backend labels.

### Exit Criteria
- Lava deployment pipeline is implemented and reproducible end-to-end.
- Quantization and export reports are generated for supported model formats and attached to runs.
- Standardized manifests produce backend-comparison leaderboards with consistent ranking criteria.
- Regression safeguards ensure deployment/export changes do not silently degrade benchmark comparability.

### Required Tests
- End-to-end tests for Lava deployment from manifest to executable artifact.
- Golden-file tests for quantization/export reports (format, required fields, and metric consistency).
- Cross-backend comparison tests verifying deterministic leaderboard generation logic.
- Performance regression tests for deployment latency and report generation overhead.

### Artifact Expectations
- Lava pipeline scripts/modules and associated configuration templates.
- Quantization/export report schemas and generated example outputs.
- Experiment manifest set covering representative workloads for backend comparison.
- Leaderboard generation module integrated into existing reporting outputs.

---

## Phase 3 (8–12+ weeks): Advanced Evaluation and Hardware Readiness

### Scope
- Bio-plausible track (plasticity + adaptation).
- Extended evaluation metrics and ablation studies.
- Hardware-readiness score integrated into experiment summary.

### Entry Criteria
- Phase 2 deployment and reporting features are stable in CI.
- Experimental protocol for backend comparison is versioned and reproducible.
- Core telemetry needed for hardware-readiness scoring is available in run outputs.

### Exit Criteria
- Bio-plausible benchmarking track supports at least one plasticity mechanism and one adaptation mechanism across designated backends.
- Extended metric suite and ablation workflows are included in standard evaluation jobs.
- A hardware-readiness score is computed per run and surfaced in final experiment summaries.
- Documentation explains score composition, interpretation, and known limitations.

### Required Tests
- Functional tests for plasticity/adaptation configuration parsing and execution.
- Statistical consistency tests for extended metrics and ablation result aggregation.
- Scoring validation tests for hardware-readiness computation (weighting, normalization, and edge cases).
- End-to-end summary tests confirming score integration in experiment reports and dashboards.

### Artifact Expectations
- Bio-plausible track manifests, configs, and reference baselines.
- Extended metrics + ablation analysis modules and notebooks/reports.
- Hardware-readiness scoring implementation with configuration and calibration docs.
- Updated experiment summary outputs including readiness score fields and visual indicators.
