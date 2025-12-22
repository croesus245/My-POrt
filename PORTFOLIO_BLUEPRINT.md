# 2026-READY ML PORTFOLIO BLUEPRINT
## Brutally Honest Senior Review Edition

---

# PART 1 — BRUTAL PORTFOLIO STRATEGY

## Positioning Statement

> **"I build production ML systems that survive contact with real data—fraud detection, secure LLM applications, and distribution-shift-aware benchmarks—under latency, cost, and adversarial constraints."**

---

## Ruthless Content Rule

| IN | OUT | WHY |
|----|-----|-----|
| Systems with drift monitoring | Static notebooks | Notebooks prove you can prototype. Systems prove you can ship. |
| Eval harnesses with slice metrics | "Accuracy: 94%" | Aggregate metrics hide failures. Slices reveal production risk. |
| Cost/latency budgets | "It works" | Engineering is about tradeoffs. No budget = no credibility. |
| Failure modes + postmortems | Success-only narratives | Adults know systems break. Show you know what to do when they do. |
| Reproducible one-command runs | "Clone and figure it out" | If a stranger can't run it in 10 minutes, it's not a portfolio piece. |
| Security/abuse test suites (LLM) | "Prompt engineering" | Anyone can write prompts. Few can defend against injection. |
| Delayed label handling | Instant feedback loops | Real ML has feedback delays. Toy projects don't. |

---

## Portfolio Homepage Layout (Above the Fold)

```
┌─────────────────────────────────────────────────────────────────┐
│  [NAME] — ML Systems Engineer                                   │
│  "I ship ML that survives production."                          │
│                                                                 │
│  [3 FLAGSHIP PROJECT CARDS - horizontal]                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │ FraudShield │ │ SecureRAG   │ │ ShiftBench  │                │
│  │ MLOps       │ │ LLM/GenAI   │ │ Research    │                │
│  │ ✓CI ✓Drift  │ │ ✓Attack     │ │ ✓Calibration│                │
│  │ ✓Runbook    │ │ ✓Eval       │ │ ✓Ablations  │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
│                                                                 │
│  [PROOF BADGES ROW]                                             │
│  ✓ CI-gated evals │ ✓ Drift runbooks │ ✓ Attack test suites     │
│  ✓ Cost reports   │ ✓ Model cards    │ ✓ Postmortem templates   │
└─────────────────────────────────────────────────────────────────┘
```

**Why this layout:**
- Name + positioning in < 2 seconds
- Three projects = focus, not scatter
- Proof badges = instant credibility signal
- No "About Me" wall of text above the fold

---

## Checklist: Reasons a Senior Reviewer Would Close This Portfolio in 10 Seconds

- [ ] No live demo or video—just "clone the repo"
- [ ] README starts with "This project is about..." instead of what it does
- [ ] No eval results in the repo (or results without reproduction steps)
- [ ] "Accuracy: 95%" with no slice breakdown
- [ ] Requirements.txt with no pinned versions
- [ ] No CI badge or CI that doesn't run evals
- [ ] LLM project with no security testing
- [ ] "I helped build..." with no concrete deliverable
- [ ] More than 5 projects (scatter = no depth)
- [ ] No cost or latency discussion
- [ ] No failure modes documented
- [ ] Dataset not documented (or "I used Kaggle X")
- [ ] No monitoring plan for production-style projects
- [ ] "TODO" items visible in main branch
- [ ] Last commit > 6 months ago

---

# PART 2 — THREE FLAGSHIP PROJECTS

---

## PROJECT 1: FraudShield
**LANE: A (Applied ML + MLOps)**
**ONE-LINE VALUE CLAIM: Real-time transaction fraud detection with delayed label reconciliation, drift-aware retraining, and incident simulation.**

### A) Problem & Why It Matters
Fraud detection operates under adversarial distribution shift, delayed ground truth (chargebacks arrive 30–90 days late), and extreme class imbalance. Most portfolios show a static classifier. Production fraud systems need continuous recalibration, data contracts, and incident response.

### B) What I'm Building
A streaming fraud scoring service that:
- Ingests transactions via Kafka (simulated)
- Scores in real-time (< 50ms p95)
- Handles delayed labels with a reconciliation pipeline
- Detects drift and triggers retraining or recalibration
- Includes a simulated "model failure" incident with full postmortem

### C) What Makes It Uncommon
- **Delayed label reconciliation loop**: Most fraud demos ignore that labels arrive weeks later. This system has a label-join pipeline and tracks provisional vs. confirmed performance.
- **Drift response playbook**: Not just "drift detected"—explicit decision tree for retrain/rollback/recalibrate.
- **Adversarial simulation**: A synthetic "fraud pattern shift" injected mid-stream to test detection and response.
- **Data contracts**: Schema + distribution expectations enforced at ingestion.
- **Incident simulation**: Intentionally broke the model, documented the postmortem.

### D) Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Transaction  │───▶│ Feature      │───▶│ Scoring      │
│ Stream       │    │ Store        │    │ Service      │
│ (Kafka)      │    │ (Redis)      │    │ (FastAPI)    │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                           ┌───────────────────┘
                           ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Prediction   │───▶│ Label        │───▶│ Drift        │
│ Log (DB)     │    │ Reconciler   │    │ Monitor      │
└──────────────┘    │ (30-day lag) │    │ (Evidently)  │
                    └──────────────┘    └──────────────┘
                                               │
                           ┌───────────────────┘
                           ▼
                    ┌──────────────┐
                    │ Retraining   │
                    │ Trigger      │
                    └──────────────┘
```

**Data flow:**
1. Transactions stream in → features computed (velocity, aggregates, embeddings)
2. Scoring service returns fraud probability + decision
3. Predictions logged with timestamp
4. Label reconciler joins confirmed fraud labels (delayed 30+ days)
5. Drift monitor compares recent predictions to confirmed outcomes
6. Retrain trigger fires if drift exceeds threshold

### E) Evaluation Plan

| Component | Metric | Target |
|-----------|--------|--------|
| **Baseline** | XGBoost with manual features | PR-AUC > 0.85 |
| **Production model** | XGBoost + entity embeddings | PR-AUC > 0.90 |
| **Slice metrics** | PR-AUC by merchant category, transaction amount bucket, user tenure | No slice < 0.80 |
| **Stress test** | Inject 5% label noise | PR-AUC drop < 5% |
| **Robustness test** | Synthetic distribution shift (new fraud pattern) | Detection within 48h simulated time |
| **Regression test** | CI gate | PR-AUC ≥ baseline AND no slice regression > 3% |
| **Ship criteria** | PR-AUC > 0.88 on holdout, all slices > 0.78, latency p95 < 50ms |

### F) Monitoring & Drift Plan

| What to Monitor | Method | Alert Threshold | Response |
|-----------------|--------|-----------------|----------|
| Prediction distribution | PSI (Population Stability Index) | PSI > 0.1 | Investigate feature drift |
| Feature drift | KS test per feature | p < 0.01 on >3 features | Check upstream data |
| Confirmed fraud rate | 7-day rolling vs. baseline | >20% deviation | Trigger recalibration |
| Latency p95 | Service metrics | > 50ms | Scale or optimize |
| Label delay | Days since last label join | > 45 days | Alert data team |

**Response Playbook:**
1. **Drift detected**: Page on-call → Run slice analysis → Identify root cause
2. **Root cause = upstream data**: Rollback to previous feature version, alert data team
3. **Root cause = model decay**: Trigger retraining with last 90 days of labeled data
4. **Retraining fails**: Rollback to previous model, create incident ticket
5. **Post-incident**: Postmortem within 48h, update runbook

### G) Cost/Latency Budget

| Metric | Target | Optimization Lever |
|--------|--------|--------------------|
| p50 latency | < 20ms | Feature caching in Redis |
| p95 latency | < 50ms | Model quantization (int8) |
| Throughput | 10,000 TPS | Horizontal scaling, batching |
| Infra cost | < $500/month (simulated) | Spot instances, autoscaling |
| Cost per 1M predictions | ~$5 | Batch inference for backfill |

### H) Safety / Risk

| Failure Mode | Likelihood | Impact | Mitigation |
|--------------|------------|--------|------------|
| Model scores all transactions as low-risk | Medium | High (fraud loss) | Minimum fraud rate alert |
| Latency spike causes timeouts | Medium | Medium (fallback to rules) | Circuit breaker + rule-based fallback |
| Label pipeline breaks | Low | High (silent model decay) | Label freshness alert |
| Adversarial attack (fraud pattern change) | High | High | Drift detection + rapid retrain |

**Residual Risk Statement:**
The system cannot detect zero-day fraud patterns until labeled examples arrive (30+ day lag). Mitigation: rule-based fallback for anomalous transactions + human review queue.

### I) Proof Pack Deliverables

```
/docs
  ├── MODEL_CARD.md           # Model card (performance, slices, limitations)
  ├── DATASET_DATASHEET.md    # Data provenance, collection, biases
  ├── RUNBOOK.md              # Operational playbook
  ├── POSTMORTEM_TEMPLATE.md  # Blank + one filled example
  ├── INCIDENT_SIMULATION.md  # "We broke it on purpose" report
  └── ARCHITECTURE.md         # Design doc

README sections:
  - Problem statement (3 lines)
  - Quick start (one command)
  - Eval results table
  - Architecture diagram
  - Drift monitoring summary
  - Cost/latency report
  - Known limitations
```

### J) Demo Plan (90 seconds)

| Time | Show | Say |
|------|------|-----|
| 0–15s | Architecture diagram | "Real-time fraud scoring with delayed label reconciliation" |
| 15–30s | Terminal: `make serve` + curl request | "Scores in < 50ms, returns probability + decision" |
| 30–45s | Grafana dashboard | "Monitoring prediction drift, latency, label lag" |
| 45–60s | Eval results table | "PR-AUC by slice—no slice below 0.80" |
| 60–75s | Incident simulation doc | "Intentionally broke the model, here's the postmortem" |
| 75–90s | Model card | "Full documentation: limitations, failure modes, runbook" |

### K) Brutal Acceptance Criteria

- [ ] `make eval` runs in < 5 minutes from clean clone
- [ ] PR-AUC ≥ 0.88 on holdout
- [ ] All slices ≥ 0.78
- [ ] p95 latency < 50ms under 1000 TPS load test
- [ ] Drift detection fires within 48h of injected shift
- [ ] Label reconciliation pipeline tested with synthetic lag
- [ ] Incident simulation documented with root cause + fix
- [ ] Model card, dataset datasheet, runbook complete
- [ ] CI runs eval suite and gates on regression
- [ ] Stranger can run end-to-end in < 15 minutes

---

## PROJECT 2: SecureRAG
**LANE: B (LLM/GenAI Engineer)**
**ONE-LINE VALUE CLAIM: Enterprise RAG system with defense-in-depth against prompt injection, data exfiltration, and tool abuse—plus measurable retrieval and faithfulness metrics.**

### A) Problem & Why It Matters
RAG systems are deployed without adversarial testing. Prompt injection can leak documents, exfiltrate data, or abuse tools. Most portfolios show "RAG chatbot" with zero security. Production RAG needs permission models, output validation, and attack test suites.

### B) What I'm Building
A document Q&A system with:
- Multi-tenant document store with access controls
- Retrieval + generation pipeline with faithfulness grounding
- Defense layers: input sanitization, output validation, tool permission model
- Attack test suite: prompt injection, indirect injection, data exfil, tool abuse
- Measurable eval: retrieval precision/recall, faithfulness, attack success rate

### C) What Makes It Uncommon
- **attack_tests/ folder**: 50+ test cases for prompt injection, jailbreaks, indirect injection via documents, data exfiltration attempts, tool abuse.
- **Permission model**: Tools have explicit scopes; LLM cannot escalate.
- **Output validation**: Responses checked for PII leakage, hallucination markers, injection echoes.
- **Faithfulness scoring**: Every response scored against retrieved chunks.
- **Multi-tenant isolation**: User A cannot access User B's documents via injection.

### D) Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ User Query   │───▶│ Input        │───▶│ Query        │
│              │    │ Sanitizer    │    │ Encoder      │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                           ┌───────────────────┘
                           ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Document     │◀───│ Retriever    │───▶│ Permission   │
│ Store        │    │ (Vector DB)  │    │ Filter       │
│ (per-tenant) │    └──────────────┘    └──────────────┘
└──────────────┘                               │
                           ┌───────────────────┘
                           ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ LLM          │◀───│ Prompt       │───▶│ Tool         │
│ (GPT-4/Local)│    │ Constructor  │    │ Executor     │
└──────────────┘    └──────────────┘    │ (sandboxed)  │
       │                                └──────────────┘
       ▼
┌──────────────┐    ┌──────────────┐
│ Output       │───▶│ Response     │
│ Validator    │    │ (to user)    │
└──────────────┘    └──────────────┘
```

**Security layers:**
1. **Input sanitizer**: Detect/neutralize injection patterns
2. **Permission filter**: Only retrieve documents user has access to
3. **Tool sandbox**: Tools have explicit capabilities; no shell access
4. **Output validator**: Check for PII, injection echoes, hallucination markers

### E) Evaluation Plan

| Component | Metric | Target |
|-----------|--------|--------|
| **Retrieval** | Precision@5, Recall@5 | P@5 > 0.7, R@5 > 0.6 |
| **Faithfulness** | % responses grounded in retrieved chunks (LLM-as-judge) | > 90% |
| **Answer quality** | Human eval on 100 queries (1–5 scale) | Mean > 3.8 |
| **Prompt injection defense** | Attack success rate | < 5% |
| **Data exfiltration defense** | Exfil success rate | 0% |
| **Tool abuse defense** | Unauthorized tool call rate | 0% |
| **Regression test** | CI gate | Faithfulness ≥ 90%, attack success < 5% |

**Attack test categories:**
1. Direct prompt injection (50 cases)
2. Indirect injection via document (20 cases)
3. Data exfiltration attempts (20 cases)
4. Tool abuse / privilege escalation (15 cases)
5. PII extraction attempts (15 cases)

### F) Monitoring & Drift Plan

| What to Monitor | Method | Alert Threshold | Response |
|-----------------|--------|-----------------|----------|
| Retrieval hit rate | % queries with ≥1 relevant chunk | < 60% | Review query distribution |
| Faithfulness score | Rolling 7-day average | < 85% | Review prompt template |
| Injection pattern matches | Regex + classifier | > 10/hour | Review inputs, update blocklist |
| Output validation failures | % responses flagged | > 5% | Review LLM behavior |
| Latency p95 | Service metrics | > 3s | Scale or optimize |
| Cost per query | Token tracking | > $0.05 | Review chunking, caching |

**Response Playbook:**
1. **Injection spike**: Block IP/user, review patterns, update sanitizer
2. **Faithfulness drop**: A/B test prompt variants, review retrieval quality
3. **Exfil attempt detected**: Immediate block, incident report, review logs
4. **Cost spike**: Enable caching, reduce chunk count, consider local model

### G) Cost/Latency Budget

| Metric | Target | Optimization Lever |
|--------|--------|--------------------|
| p50 latency | < 2s | Retrieval caching, streaming |
| p95 latency | < 4s | Pre-computed embeddings |
| Cost per query (GPT-4) | < $0.03 | Chunk pruning, prompt compression |
| Cost per query (local) | < $0.001 | Quantized local model |
| Throughput | 100 QPS | Async retrieval, batching |

### H) Safety / Risk

| Failure Mode | Likelihood | Impact | Mitigation |
|--------------|------------|--------|------------|
| Prompt injection succeeds | Medium | High | Multi-layer defense, monitoring |
| Data exfiltration | Low | Critical | Permission model, output validation |
| Hallucination presented as fact | High | Medium | Faithfulness scoring, citation requirement |
| Tool abuse | Low | High | Capability restrictions, sandboxing |
| PII leakage in response | Medium | High | Output PII scanner |

**Residual Risk Statement:**
Novel injection techniques may bypass current defenses. Mitigation: Continuous red-teaming, monitor for anomalous outputs, human review for high-stakes queries.

### I) Proof Pack Deliverables

```
/docs
  ├── SECURITY_MODEL.md       # Threat model, defense layers
  ├── ATTACK_TEST_REPORT.md   # Results of attack suite
  ├── EVAL_REPORT.md          # Retrieval + faithfulness metrics
  ├── RUNBOOK.md              # Operational playbook
  ├── INCIDENT_RESPONSE.md    # Security incident procedures
  └── ARCHITECTURE.md         # Design doc

/attack_tests
  ├── prompt_injection/       # 50+ injection test cases
  ├── indirect_injection/     # Document-based injection
  ├── data_exfiltration/      # Exfil attempts
  ├── tool_abuse/             # Privilege escalation
  └── pii_extraction/         # PII leak attempts

README sections:
  - Problem statement
  - Security model summary
  - Quick start
  - Eval results (retrieval + faithfulness + attack defense)
  - Architecture diagram
  - Attack test summary
  - Cost/latency report
  - Known limitations
```

### J) Demo Plan (90 seconds)

| Time | Show | Say |
|------|------|-----|
| 0–15s | Architecture diagram | "Multi-tenant RAG with defense-in-depth" |
| 15–30s | Normal query flow | "User asks question, retrieves from their docs only" |
| 30–45s | Attack test run | "Running prompt injection suite—0/50 succeeded" |
| 45–60s | Faithfulness dashboard | "Every response scored for grounding—currently 94%" |
| 60–75s | Permission demo | "User A cannot access User B's docs via injection" |
| 75–90s | Security report | "Full threat model, attack results, incident procedures" |

### K) Brutal Acceptance Criteria

- [ ] `make attack-test` passes with < 5% injection success
- [ ] 0% data exfiltration success rate
- [ ] 0% unauthorized tool execution
- [ ] Retrieval P@5 > 0.7 on test set
- [ ] Faithfulness > 90% on test set
- [ ] p95 latency < 4s
- [ ] Multi-tenant isolation verified (100 cross-tenant attack attempts fail)
- [ ] Security model documented with threat categories
- [ ] Incident response procedures documented
- [ ] Stranger can run attack suite in < 10 minutes

---

## PROJECT 3: ShiftBench
**LANE: C (Research-ish)**
**ONE-LINE VALUE CLAIM: A benchmark for medical image classification under temporal and demographic distribution shift, with calibration-aware evaluation and honest negative results.**

### A) Problem & Why It Matters
ML benchmarks hide distribution shift. Models trained on 2018 data are tested on 2018 data—then deployed on 2025 data and fail silently. Medical imaging is particularly vulnerable: scanners change, populations shift, annotation guidelines evolve. Most benchmarks ignore this.

### B) What I'm Building
A benchmark dataset + evaluation framework for skin lesion classification that:
- Explicitly splits by time (train: 2015–2018, test: 2019–2022)
- Explicitly splits by demographic (geography, skin tone proxies)
- Evaluates calibration (ECE, reliability diagrams) alongside accuracy
- Includes 3+ strong baselines with full ablations
- Documents negative results (what didn't work)

### C) What Makes It Uncommon
- **Temporal split**: Train/test from different years, not random split.
- **Demographic subgroup analysis**: Performance by skin tone proxy (Fitzpatrick annotations or luminance).
- **Calibration-first evaluation**: Models ranked by calibrated accuracy, not just raw accuracy.
- **Uncertainty quantification**: MC Dropout, ensemble, temperature scaling compared.
- **Negative results section**: What I tried that failed, and why.
- **Reproducibility artifacts**: Exact splits, seeds, one-command reproduction.

### D) Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                           │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Raw Images   │ Metadata     │ Temporal     │ Demographic    │
│ (ISIC)       │ (dates, geo) │ Split Logic  │ Split Logic    │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                       │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Baseline 1   │ Baseline 2   │ Baseline 3   │ Ablations      │
│ (ResNet-50)  │ (EfficientNet)│(ViT-B/16)   │ (augment, etc) │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      EVALUATION PIPELINE                     │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Accuracy     │ Slice Perf   │ Calibration  │ Uncertainty    │
│ (overall)    │ (by year,    │ (ECE, MCE,   │ (MC Dropout,   │
│              │  skin tone)  │  reliability)│  ensemble)     │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      REPORTING                               │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Main Results │ Ablation     │ Negative     │ Limitations    │
│ Table        │ Tables       │ Results      │ Section        │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### E) Evaluation Plan

| Component | Metric | Target |
|-----------|--------|--------|
| **Primary metric** | Balanced accuracy (macro) | Report, don't target |
| **Calibration** | ECE (Expected Calibration Error) | ECE < 0.10 |
| **Slice: temporal** | Accuracy by test year (2019, 2020, 2021, 2022) | Report degradation |
| **Slice: demographic** | Accuracy by skin tone proxy (light/medium/dark) | Report gap |
| **Robustness** | Accuracy under synthetic corruptions (blur, noise) | Report degradation curve |
| **Uncertainty** | AUROC of uncertainty vs. correctness | > 0.65 |

**Baselines:**
1. ResNet-50 (ImageNet pretrained)
2. EfficientNet-B4 (ImageNet pretrained)
3. ViT-B/16 (ImageNet-21k pretrained)

**Ablations:**
- With/without heavy augmentation
- With/without class balancing
- With/without temperature scaling
- Training data size (25%, 50%, 100%)

### F) Monitoring & Drift Plan

*Not applicable for static benchmark—but include "benchmark maintenance" section:*

| Concern | Mitigation |
|---------|------------|
| New ISIC data released | Versioned splits, document inclusion criteria |
| Annotation guideline changes | Document annotation vintage, flag in metadata |
| Benchmark overfitting (community) | Withhold portion of test set as "hidden" |

### G) Cost/Latency Budget

| Metric | Value | Notes |
|--------|-------|-------|
| Training cost (3 baselines + ablations) | ~$50 (cloud GPU) | A100 for 10 hours |
| Inference cost per 1k images | ~$0.10 | Batch inference |
| Full benchmark run | < 24 hours | Single A100 |
| Reproduction from scratch | < 4 hours | Pretrained weights |

### H) Safety / Risk

| Failure Mode | Likelihood | Impact | Mitigation |
|--------------|------------|--------|------------|
| Benchmark used for clinical claims | High | Critical | Explicit "not for clinical use" disclaimer |
| Demographic proxy is imperfect | High | Medium | Document limitations, use multiple proxies |
| Overfitting to benchmark | Medium | Medium | Hidden test portion, discourage leaderboard chasing |
| Data leakage across splits | Low | High | Explicit patient-level deduplication |

**Residual Risk Statement:**
Skin tone proxies (Fitzpatrick annotations, luminance) are imperfect. Conclusions about demographic fairness are approximate. This benchmark is for research, not deployment decisions.

### I) Proof Pack Deliverables

```
/docs
  ├── DATASET_DATASHEET.md    # Collection, biases, splits
  ├── BENCHMARK_SPEC.md       # Evaluation protocol
  ├── RESULTS.md              # Main results + ablations
  ├── NEGATIVE_RESULTS.md     # What didn't work
  ├── LIMITATIONS.md          # Honest limitations
  └── REPRODUCTION.md         # Exact reproduction steps

/data
  ├── splits/                 # Train/val/test CSVs with hashes
  └── metadata/               # Year, demographic proxies

README sections:
  - Benchmark motivation
  - Quick start (one command)
  - Main results table
  - Slice results (temporal, demographic)
  - Calibration results
  - Negative results summary
  - Known limitations
  - How to cite
```

### J) Demo Plan (90 seconds)

| Time | Show | Say |
|------|------|-----|
| 0–15s | Motivation slide | "Most benchmarks hide distribution shift. This one exposes it." |
| 15–30s | Split diagram | "Train on 2015–2018, test on 2019–2022. Explicit temporal shift." |
| 30–45s | Main results table | "All models degrade on later years. ViT degrades least." |
| 45–60s | Demographic slice results | "Models underperform on darker skin tones. Gap: 8% for ResNet." |
| 60–75s | Calibration plot | "ECE before/after temperature scaling. Most models overconfident." |
| 75–90s | Negative results | "Mixup didn't help. Heavy augmentation hurt calibration. Documented." |

### K) Brutal Acceptance Criteria

- [ ] `make eval` reproduces all results tables within 1% tolerance
- [ ] Temporal split verified (no 2019+ images in training)
- [ ] Demographic annotations verified (source documented)
- [ ] 3+ baselines fully trained and evaluated
- [ ] Ablation tables complete (5+ ablations)
- [ ] Calibration metrics reported for all models
- [ ] Uncertainty AUROC reported
- [ ] Negative results section with ≥ 3 failed experiments
- [ ] Limitations section with ≥ 5 explicit limitations
- [ ] Stranger can reproduce main table in < 4 hours

---

# PART 3 — REPO TEMPLATES

---

## Project A: FraudShield (MLOps)

```
fraudshield/
├── README.md
├── Makefile
├── pyproject.toml          # pinned deps
├── .github/
│   └── workflows/
│       ├── ci.yml          # lint + unit tests
│       └── eval.yml        # eval suite (gated)
├── data/
│   ├── README.md           # data sources, not raw data
│   └── sample/             # small sample for tests
├── src/
│   ├── features/           # feature engineering
│   ├── model/              # training, inference
│   ├── serving/            # FastAPI service
│   ├── reconciliation/     # label join pipeline
│   └── drift/              # drift detection
├── configs/
│   ├── train.yaml
│   ├── serve.yaml
│   └── drift.yaml
├── eval/
│   ├── run_eval.py
│   ├── slice_analysis.py
│   └── stress_test.py
├── monitoring/
│   ├── dashboards/         # Grafana JSON
│   ├── alerts/             # alert rules
│   └── drift_check.py
├── docs/
│   ├── MODEL_CARD.md
│   ├── DATASET_DATASHEET.md
│   ├── RUNBOOK.md
│   ├── POSTMORTEM_TEMPLATE.md
│   ├── INCIDENT_SIMULATION.md
│   └── ARCHITECTURE.md
├── tests/
│   ├── unit/
│   └── integration/
└── scripts/
    ├── simulate_stream.py
    └── inject_drift.py
```

**Makefile commands:**
```
make setup          # create venv, install deps
make train          # train model
make eval           # run full eval suite
make eval-slice     # run slice analysis
make serve          # start scoring service
make test           # run unit + integration tests
make monitor        # run drift check
make simulate       # simulate transaction stream
make inject-drift   # inject synthetic drift
make report         # generate eval report
make ci             # lint + test + eval (CI entry point)
```

---

## Project B: SecureRAG (LLM/GenAI)

```
securerag/
├── README.md
├── Makefile
├── pyproject.toml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── attack-test.yml
├── data/
│   ├── documents/          # sample docs for testing
│   └── eval_queries/       # eval query set
├── src/
│   ├── ingest/             # document ingestion
│   ├── retrieval/          # vector search
│   ├── generation/         # LLM integration
│   ├── security/           # sanitization, validation
│   ├── tools/              # sandboxed tool implementations
│   └── serving/            # FastAPI service
├── configs/
│   ├── retrieval.yaml
│   ├── generation.yaml
│   └── security.yaml
├── eval/
│   ├── retrieval_eval.py
│   ├── faithfulness_eval.py
│   └── run_eval.py
├── attack_tests/
│   ├── prompt_injection/
│   │   ├── cases.yaml      # 50+ test cases
│   │   └── run_injection_tests.py
│   ├── indirect_injection/
│   ├── data_exfiltration/
│   ├── tool_abuse/
│   ├── pii_extraction/
│   └── run_all_attacks.py
├── monitoring/
│   ├── dashboards/
│   ├── alerts/
│   └── anomaly_detection.py
├── docs/
│   ├── SECURITY_MODEL.md
│   ├── ATTACK_TEST_REPORT.md
│   ├── EVAL_REPORT.md
│   ├── RUNBOOK.md
│   ├── INCIDENT_RESPONSE.md
│   └── ARCHITECTURE.md
└── tests/
    ├── unit/
    └── integration/
```

**Makefile commands:**
```
make setup          # create venv, install deps
make ingest         # ingest sample documents
make eval           # run retrieval + faithfulness eval
make serve          # start RAG service
make test           # run unit + integration tests
make attack-test    # run full attack suite
make attack-inject  # run injection tests only
make attack-exfil   # run exfiltration tests only
make monitor        # run anomaly detection
make report         # generate security + eval report
make ci             # lint + test + eval + attack-test
```

---

## Project C: ShiftBench (Research)

```
shiftbench/
├── README.md
├── Makefile
├── pyproject.toml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── reproduce.yml   # weekly reproduction check
├── data/
│   ├── README.md           # download instructions
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── metadata/
│       └── demographics.csv
├── src/
│   ├── data/               # data loading, augmentation
│   ├── models/             # baseline implementations
│   ├── training/           # training loop
│   ├── evaluation/         # metrics, calibration
│   └── uncertainty/        # UQ methods
├── configs/
│   ├── resnet50.yaml
│   ├── efficientnet.yaml
│   ├── vit.yaml
│   └── ablations/
├── eval/
│   ├── run_eval.py
│   ├── slice_eval.py
│   ├── calibration_eval.py
│   └── uncertainty_eval.py
├── results/
│   ├── main_table.csv
│   ├── ablations/
│   └── figures/
├── docs/
│   ├── DATASET_DATASHEET.md
│   ├── BENCHMARK_SPEC.md
│   ├── RESULTS.md
│   ├── NEGATIVE_RESULTS.md
│   ├── LIMITATIONS.md
│   └── REPRODUCTION.md
└── tests/
    ├── test_splits.py      # verify no leakage
    └── test_reproduction.py
```

**Makefile commands:**
```
make setup          # create venv, install deps
make download       # download dataset
make train          # train all baselines
make train-ablations # train ablations
make eval           # run full eval (all models)
make eval-slice     # run slice analysis
make eval-calibration # run calibration analysis
make eval-uncertainty # run UQ analysis
make test           # run split verification tests
make report         # generate results tables + figures
make reproduce      # full reproduction (train + eval)
make ci             # lint + test + eval
```

---

# PART 4 — PORTFOLIO WEBSITE CONTENT

---

## Hero Section

**Headline:**
> I ship ML systems that survive production.

**Subheadline:**
> Fraud detection with drift response. Secure RAG with attack testing. Benchmarks that expose distribution shift. Every project includes eval harness, monitoring playbook, and cost report.

---

## Three Flagship Project Cards

### Card 1: FraudShield
**Real-time fraud detection with delayed label reconciliation**

Streaming fraud scoring with 48ms p95 latency, drift-aware retraining pipeline, and a documented incident simulation. Handles the hard part: labels arrive 30 days late.

**Proof badges:** ✓ CI-gated evals · ✓ Drift runbook · ✓ Incident postmortem · ✓ Model card

---

### Card 2: SecureRAG
**Enterprise RAG with defense-in-depth**

Multi-tenant document Q&A with prompt injection defense, data exfiltration prevention, and tool sandboxing. 120+ attack tests. 0% exfil success rate.

**Proof badges:** ✓ Attack test suite · ✓ Faithfulness eval · ✓ Security model · ✓ Incident response

---

### Card 3: ShiftBench
**Medical imaging benchmark under distribution shift**

Skin lesion classification with explicit temporal and demographic splits. Calibration-aware evaluation. Honest negative results section.

**Proof badges:** ✓ Reproducible splits · ✓ Calibration metrics · ✓ Ablation studies · ✓ Limitations documented

---

## "What I Do" Section

I design, build, and operate ML systems—not just train models.

- **Applied ML + MLOps:** Streaming inference, delayed labels, drift detection, incident response
- **LLM/GenAI Engineering:** RAG systems, prompt security, tool orchestration, faithfulness evaluation
- **Research:** Benchmark design, distribution shift analysis, calibration, uncertainty quantification

Every project ships with: eval harness, monitoring playbook, cost/latency report, and documentation artifacts.

---

## "Proof Badges" Section

| Artifact | What It Proves |
|----------|----------------|
| ✓ CI-gated evals | Models pass regression tests before merge |
| ✓ Drift runbooks | I know what to do when production distribution shifts |
| ✓ Attack test suites | LLM systems tested against real adversarial inputs |
| ✓ Eval harnesses | Performance measured across slices, not just aggregate |
| ✓ Cost reports | I think about infrastructure costs, not just accuracy |
| ✓ Model cards | Limitations and failure modes documented up front |
| ✓ Incident postmortems | I've broken systems on purpose and documented the recovery |

---

## Resume Highlights (5 bullets)

1. Deployed real-time fraud scoring system handling 10K TPS with < 50ms p95 latency and drift-aware retraining
2. Built secure RAG pipeline with 120+ adversarial test cases; 0% data exfiltration success rate
3. Designed distribution-shift benchmark for medical imaging with temporal and demographic splits
4. Implemented delayed label reconciliation pipeline handling 30–90 day feedback lag
5. Authored model cards, runbooks, and incident postmortems for all production systems

---

## Contact CTA

**Let's talk about ML systems that actually ship.**

[Email] · [GitHub] · [LinkedIn]

---

# PART 5 — BRUTAL EDITOR MODE

---

## 15 Common ML Portfolio Mistakes I Refuse to Let You Make

1. **Titanic/MNIST/Iris projects** — Instant close. No signal.
2. **"Accuracy: 94%"** with no slice metrics — Hiding failures.
3. **No reproducibility** — If I can't run it, it doesn't exist.
4. **Unpinned dependencies** — "It worked on my machine" is not engineering.
5. **No CI** — If you don't test it, you don't trust it.
6. **LLM project with no security testing** — Negligent in 2026.
7. **"I helped build..."** — Vague contributions = no contributions.
8. **No cost/latency discussion** — Engineering is about tradeoffs.
9. **No failure modes documented** — You haven't thought about production.
10. **Success-only narratives** — Suspicious. Adults know things fail.
11. **More than 5 projects** — Scatter. No depth.
12. **Wall of text README** — Nobody reads it. Show results first.
13. **No demo or video** — If you can't show it in 90 seconds, it's not ready.
14. **Last commit 6+ months ago** — Abandoned. Or you stopped learning.
15. **"TODO" in main branch** — Not portfolio-ready.

---

## 10 "Adult" Artifacts Most People Ignore (and Why They Matter)

| Artifact | Why It Matters |
|----------|----------------|
| **Model Card** | Shows you think about limitations, not just performance |
| **Dataset Datasheet** | Shows you understand data provenance and bias |
| **Runbook** | Shows you can operate, not just deploy |
| **Postmortem Template** | Shows you've thought about failure response |
| **Incident Simulation** | Shows you've actually tested failure scenarios |
| **Slice Metrics** | Shows you don't hide behind aggregates |
| **Cost Report** | Shows you think like an engineer, not a researcher |
| **Drift Detection Config** | Shows you know models decay |
| **Attack Test Suite (LLM)** | Shows you understand adversarial threats |
| **Negative Results** | Shows intellectual honesty and maturity |

---

## Portfolio Scoring Rubric (Out of 100)

| Category | Weight | What I Grade Hardest |
|----------|--------|----------------------|
| **Reproducibility** | 20 | Can a stranger run it in 15 minutes? Pinned deps? Seeded runs? |
| **Evaluation Quality** | 20 | Slice metrics? Baselines? Regression tests? |
| **Production Artifacts** | 15 | Runbook? Model card? Cost report? Drift plan? |
| **Security (LLM only)** | 15 | Attack test suite? Permission model? Output validation? |
| **Documentation** | 10 | README structure? Architecture doc? Limitations? |
| **Depth over Breadth** | 10 | 3 deep projects > 10 shallow ones |
| **Demo Quality** | 5 | 90-second demo? Video? Live link? |
| **Recency** | 5 | Commits in last 3 months? Current tools/frameworks? |

**Grading scale:**
- 90–100: Senior engineer portfolio. Hire signal.
- 80–89: Solid mid-level. Minor gaps.
- 70–79: Junior with promise. Missing production artifacts.
- 60–69: Academic portfolio. No production thinking.
- < 60: Toy projects or abandoned repos. Pass.

---

## Final Checklist Before Publishing

- [ ] All three flagships pass their brutal acceptance criteria
- [ ] Every project runs with one command from clean clone
- [ ] Every project has CI that runs evals
- [ ] Every project has slice metrics
- [ ] Every project has cost/latency report
- [ ] Every project has documented failure modes
- [ ] LLM project has attack test suite with results
- [ ] Research project has negative results section
- [ ] MLOps project has incident simulation + postmortem
- [ ] No "TODO" in any main branch
- [ ] Last commit < 1 month ago
- [ ] 90-second demo ready for each project
- [ ] Portfolio homepage loads in < 2 seconds
- [ ] Mobile-responsive (reviewers check on phones)
- [ ] Contact info visible above the fold

---

**END OF BLUEPRINT**

Now execute. No excuses.
