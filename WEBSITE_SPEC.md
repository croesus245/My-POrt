# PORTFOLIO WEBSITE SPECIFICATION
## ML Systems Engineer — 2026 Edition

---

# A) COMPLETE CONTENT

---

## HEADLINES (Pick One)

1. **"I ship ML systems that survive production."**
2. **"Production ML. Secure LLMs. Research that ships."**
3. **"ML systems that work when the data shifts."**

**Recommended:** Option 1 — direct, confident, implies real-world deployment experience.

---

## SPECIALIZATION TAGLINES (Pick One)

1. "Real-time fraud detection · Secure RAG systems · Distribution-shift benchmarks"
2. "MLOps pipelines, LLM security, and research you can reproduce"
3. "I build ML that handles delayed labels, adversarial prompts, and dataset drift"
4. "From streaming inference to attack-tested LLMs to calibration-aware benchmarks"
5. "Systems that score in 50ms, defend against injection, and document their failures"

**Recommended:** Option 5 — specific constraints, proves technical depth immediately.

---

## PROOF BADGES (10 Total)

| Badge | Label | What It Proves |
|-------|-------|----------------|
| ✓ CI-Eval | CI-Gated Evals | Models pass slice + regression tests before merge |
| ✓ Drift | Drift Runbook | Documented response to production distribution shift |
| ✓ Attack | Attack Tests | LLM tested against 100+ adversarial inputs |
| ✓ Faith | Faithfulness Eval | RAG responses scored for grounding |
| ✓ Cost | Cost Report | $/1k requests and optimization analysis |
| ✓ Latency | Latency Budget | p50/p95 targets documented and tested |
| ✓ Repro | Reproducible | One-command build, pinned deps, seeded runs |
| ✓ Card | Model Card | Limitations and failure modes documented |
| ✓ Incident | Postmortem | Intentional failure simulation with recovery |
| ✓ Slices | Slice Metrics | Performance by subgroup, not just aggregate |

---

## HEADER CONTENT

```
[LOGO/NAME]                                    [Work] [Proof] [Writing] [Resume] [Contact]

Your Name
ML Systems Engineer
                                               hello@yourdomain.com  |  [Download Resume ↓]
```

---

## HERO SECTION (Above Fold)

### Headline
**I ship ML systems that survive production.**

### Subheadline
Systems that score in 50ms, defend against injection, and document their failures.

### Three Flagship Project Cards

---

#### Card 1: FraudShield

**Real-time fraud detection with delayed label reconciliation**

Streaming fraud scoring at 10K TPS with < 50ms latency. Handles the hard part: labels arrive 30 days late. Includes drift detection, retraining triggers, and a documented incident simulation.

**Badges:** `CI-Eval` `Drift` `Cost` `Latency` `Postmortem` `Model Card`

**Buttons:** [Case Study] [Repo] [Live Demo]

---

#### Card 2: SecureRAG

**Enterprise RAG with defense-in-depth**

Multi-tenant document Q&A with prompt injection defense, data exfiltration prevention, and tool sandboxing. 120+ attack tests. 0% exfil success rate. Every response scored for faithfulness.

**Badges:** `Attack Tests` `Faithfulness` `CI-Eval` `Cost` `Reproducible`

**Buttons:** [Case Study] [Repo] [Attack Report]

---

#### Card 3: ShiftBench

**Medical imaging benchmark under distribution shift**

Skin lesion classification with explicit temporal (2015–2018 → 2019–2022) and demographic splits. Calibration-aware evaluation. Honest negative results section.

**Badges:** `Slice Metrics` `Reproducible` `Model Card` `CI-Eval`

**Buttons:** [Case Study] [Repo] [Results]

---

## WORK SECTION

### Section Header
**Work**
Production systems. Research benchmarks. Every project ships with proof.

### Filter Chips
`All` `MLOps` `LLM/GenAI` `Research`

### Flagship Projects (3)
*(Same cards as hero, with expanded detail on click)*

### Supporting Projects (Up to 3)

---

#### Mini-Project 1: Feature Store Migration
**MLOps**

Migrated batch feature pipeline to streaming (Feast → custom Redis). Reduced feature freshness from 24h to < 5 min. Zero downtime cutover.

**Links:** [Architecture Doc] [Repo]

---

#### Mini-Project 2: Prompt Classifier
**LLM**

Binary classifier to detect prompt injection attempts before they reach the LLM. 97% precision at 92% recall on held-out attack corpus.

**Links:** [Model Card] [Repo]

---

#### Mini-Project 3: Calibration Toolkit
**Research**

Open-source library for post-hoc calibration (temperature scaling, isotonic regression) with reliability diagrams and ECE reporting.

**Links:** [Docs] [PyPI] [Repo]

---

## PROOF SECTION

### Section Header
**Proof**
The artifacts most portfolios don't have.

### Proof Tiles (Grid Layout)

| Artifact | Description | Link |
|----------|-------------|------|
| **Evaluation Harness** | Slice metrics, regression tests, CI gates. Every model tested against 12 subgroups. | [View Harness →] |
| **Drift Runbook** | What to monitor, when to alert, how to respond. Decision tree for retrain vs. rollback. | [View Runbook →] |
| **Attack Test Suite** | 120+ adversarial test cases: injection, exfil, tool abuse, PII extraction. | [View Tests →] |
| **RAG Eval Dashboard** | Retrieval precision/recall, faithfulness scores, grounding verification. | [View Dashboard →] |
| **Cost & Latency Report** | p50/p95 latency, throughput, $/1k requests, optimization levers. | [View Report →] |
| **Model Card** | Performance by slice, known limitations, failure modes, intended use. | [View Card →] |
| **Dataset Datasheet** | Collection methodology, known biases, split logic, demographic annotations. | [View Datasheet →] |
| **Incident Postmortem** | Intentional failure simulation. Root cause, response, prevention. | [View Postmortem →] |

---

## WRITING SECTION

### Section Header
**Writing**
Short, technical, no fluff. Every post ends with what I'd do differently.

### Posts (3 Max)

---

#### Post 1: Why Offline Metrics Lie
*An experiment in deceptive validation*

Built a churn model that hit 0.92 AUC on holdout. Deployed it. Watched it fail. Here's why stratified splits hide temporal drift, and how I redesigned the eval suite.

**What I'd do differently:** Use time-based splits from day one. Never trust random holdout again.

[Read →]

---

#### Post 2: How I Design Eval Suites and CI Gates
*A practical template*

Every model I ship passes 4 gates: slice regression, calibration check, latency test, and cost ceiling. Here's the exact YAML config and why each gate exists.

**What I'd do differently:** Add distribution shift simulation as a 5th gate.

[Read →]

---

#### Post 3: Security Failures in LLM Apps
*What I learned from 100+ attack tests*

Prompt injection, indirect injection via documents, tool abuse, PII exfiltration. I tested them all. Here's what worked, what didn't, and the defense layers that survived.

**What I'd do differently:** Assume the model is compromised. Design permissions around that.

[Read →]

---

## RESUME SECTION

### Section Header
**Resume**
The highlights. Full version downloadable.

### Skills (Categories, Not a List)

| Category | Core Skills |
|----------|-------------|
| **Modeling** | XGBoost, PyTorch, Transformers, Calibration, Uncertainty Quantification |
| **Data** | Feature engineering, streaming pipelines, data contracts, quality gates |
| **Systems** | FastAPI, Redis, Kafka, Docker, Kubernetes |
| **LLM** | RAG, prompt engineering, security testing, faithfulness evaluation |
| **Infra** | CI/CD, monitoring (Prometheus/Grafana), drift detection (Evidently) |

### Experience Highlights (5 Bullets)

1. Deployed real-time fraud scoring system: 10K TPS, < 50ms p95, drift-aware retraining
2. Built secure RAG pipeline: 120 attack tests, 0% data exfiltration, 94% faithfulness
3. Designed distribution-shift benchmark: temporal + demographic splits, calibration-first
4. Migrated feature store from batch to streaming: 24h → 5 min freshness, zero downtime
5. Authored eval harnesses with slice metrics and CI gates for 3 production models

### Download
[Download Full Resume (PDF) ↓]

---

## CONTACT SECTION

### Section Header
**Contact**
Let's talk about ML systems that actually ship.

### Info

**Email:** hello@yourdomain.com
**LinkedIn:** linkedin.com/in/yourname
**GitHub:** github.com/yourname
**Location:** [City, Timezone — e.g., "San Francisco, PT"]

### CTA
**Open to roles in:** ML Systems · LLM Engineering · MLOps · Applied Research

[Send Email →]

---

## ABOUT (2 Sentences Max)

I build ML systems that handle real-world constraints: delayed labels, adversarial inputs, and distribution shift. Every project I ship includes an eval harness, monitoring playbook, and documented failure modes.

---

## CTA COPY OPTIONS

1. "Email me for roles in ML Systems / LLM Engineering / MLOps."
2. "Let's build systems that survive production. Get in touch."
3. "Hiring for ML engineering? Let's talk."

**Recommended:** Option 1 — specific, clear scope.

---

# B) WIREFRAME DESCRIPTION

---

## Desktop Layout (1440px)

### Header (Sticky, 60px height)
```
┌────────────────────────────────────────────────────────────────────────────┐
│ [Name]                    [Work] [Proof] [Writing] [Resume] [Contact]      │
│ ML Systems Engineer                          hello@email.com | [Resume ↓] │
└────────────────────────────────────────────────────────────────────────────┘
```
- Logo/name: left-aligned
- Nav: right-aligned, horizontal
- Email + Resume button: far right, always visible

### Hero (Full viewport height minus header)
```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│    I ship ML systems that survive production.                              │
│    Systems that score in 50ms, defend against injection, and document     │
│    their failures.                                                         │
│                                                                            │
│    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐         │
│    │   FraudShield    │ │    SecureRAG     │ │   ShiftBench     │         │
│    │   [MLOps]        │ │    [LLM/GenAI]   │ │   [Research]     │         │
│    │                  │ │                  │ │                  │         │
│    │   2-line desc    │ │   2-line desc    │ │   2-line desc    │         │
│    │                  │ │                  │ │                  │         │
│    │   [badges row]   │ │   [badges row]   │ │   [badges row]   │         │
│    │                  │ │                  │ │                  │         │
│    │ [Case] [Repo]    │ │ [Case] [Repo]    │ │ [Case] [Repo]    │         │
│    └──────────────────┘ └──────────────────┘ └──────────────────┘         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```
- Headline: 48–56px, bold, centered or left-aligned
- Subheadline: 20–24px, muted color
- 3 cards: equal width, ~320px each, horizontal row
- Cards contain: lane tag, title, description (2 lines), badge row (wrap if needed), 2–3 buttons
- No scroll needed to see all 3 cards

### Work Section
```
┌────────────────────────────────────────────────────────────────────────────┐
│  Work                                                                      │
│  [All] [MLOps] [LLM/GenAI] [Research]                                     │
│                                                                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                             │
│  │ Flagship 1 │ │ Flagship 2 │ │ Flagship 3 │                             │
│  └────────────┘ └────────────┘ └────────────┘                             │
│                                                                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                             │
│  │ Mini 1     │ │ Mini 2     │ │ Mini 3     │                             │
│  └────────────┘ └────────────┘ └────────────┘                             │
└────────────────────────────────────────────────────────────────────────────┘
```
- Filter chips: horizontal, toggleable
- 3-column grid
- Cards expand to detail view on click (modal or same-page scroll)

### Proof Section
```
┌────────────────────────────────────────────────────────────────────────────┐
│  Proof                                                                     │
│  The artifacts most portfolios don't have.                                │
│                                                                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Eval Harness │ │ Drift Runbook│ │ Attack Suite │ │ RAG Eval     │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘      │
│                                                                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Cost Report  │ │ Model Card   │ │ Datasheet    │ │ Postmortem   │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘      │
└────────────────────────────────────────────────────────────────────────────┘
```
- 4-column grid
- Each tile: icon, title, 1-line description, link
- Hover: subtle lift or border highlight

### Writing Section
```
┌────────────────────────────────────────────────────────────────────────────┐
│  Writing                                                                   │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ Why Offline Metrics Lie                                             │   │
│  │ 2-line summary... What I'd do differently: [one line]              │   │
│  │ [Read →]                                                            │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ How I Design Eval Suites                                           │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ Security Failures in LLM Apps                                      │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```
- Stacked cards (1 column) or 3-column grid
- Each card: title, summary, "what I'd do differently," read link

### Resume Section
```
┌────────────────────────────────────────────────────────────────────────────┐
│  Resume                                                                    │
│                                                                            │
│  [Skills grid: 5 categories]                                              │
│                                                                            │
│  [Experience: 5 bullet points]                                            │
│                                                                            │
│  [Download Resume PDF ↓]                                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

### Contact Section
```
┌────────────────────────────────────────────────────────────────────────────┐
│  Contact                                                                   │
│  Let's talk about ML systems that actually ship.                          │
│                                                                            │
│  Email: hello@email.com                                                   │
│  LinkedIn | GitHub                                                        │
│  Location: City, Timezone                                                 │
│                                                                            │
│  Open to roles in: ML Systems · LLM Engineering · MLOps                  │
│                                                                            │
│  [Send Email →]                                                           │
└────────────────────────────────────────────────────────────────────────────┘
```

### Footer
```
┌────────────────────────────────────────────────────────────────────────────┐
│  © 2025 Your Name · Built with vanilla HTML/CSS/JS                        │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Tablet Layout (768px)

- Header: hamburger menu for nav
- Hero: 2-column card layout (2 on top, 1 below) or vertical stack
- Work/Proof: 2-column grid
- Writing: single column
- Resume: single column
- Contact: single column

---

## Mobile Layout (375px)

- Header: logo + hamburger only; email/resume in dropdown
- Hero: headline + subheadline only above fold; cards scroll horizontally or stack vertically
- All grids: single column
- Cards: full width
- Buttons: full width, stacked
- Font sizes: headline 32px, body 16px

---

## Responsive Behavior Summary

| Breakpoint | Columns | Hero Cards | Nav |
|------------|---------|------------|-----|
| ≥1200px | 3–4 | 3 horizontal | Full |
| 768–1199px | 2 | 2+1 or scroll | Collapsed |
| <768px | 1 | Stack or scroll | Hamburger |

---

## Micro-Interactions

| Element | Interaction |
|---------|-------------|
| Nav links | Underline on hover, smooth scroll |
| Project cards | Lift shadow + border highlight on hover |
| Proof tiles | Subtle scale (1.02) on hover |
| Buttons | Background color shift on hover, focus ring |
| Badge pills | Tooltip on hover with full description |
| Filter chips | Active state (filled), inactive (outline) |

---

## Color Scheme Options

### Option A: Dark Mode (Recommended)
- Background: #0a0a0a
- Card background: #141414
- Text primary: #ffffff
- Text muted: #a0a0a0
- Accent: #3b82f6 (blue)
- Badge background: #1e293b
- Border: #27272a

### Option B: Light Mode
- Background: #ffffff
- Card background: #f8fafc
- Text primary: #0f172a
- Text muted: #64748b
- Accent: #2563eb
- Badge background: #e2e8f0
- Border: #e2e8f0

---

## Typography

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Headline | Inter or SF Pro | 48–56px | 700 |
| Subheadline | Inter | 20–24px | 400 |
| Section header | Inter | 32px | 600 |
| Card title | Inter | 20px | 600 |
| Body | Inter | 16px | 400 |
| Badge | Inter | 12px | 500 |
| Button | Inter | 14px | 500 |

---

# C) ACCEPTANCE CRITERIA CHECKLIST

---

## Performance

- [ ] Lighthouse Performance score ≥ 90
- [ ] First Contentful Paint < 1.5s
- [ ] Largest Contentful Paint < 2.5s
- [ ] Total page weight < 500KB (excluding images)
- [ ] No layout shift (CLS < 0.1)

## Above the Fold (Desktop)

- [ ] Headline visible without scroll
- [ ] Subheadline visible without scroll
- [ ] All 3 flagship project cards visible without scroll
- [ ] Contact button visible without scroll
- [ ] Email visible in header

## Navigation & Interaction

- [ ] All nav links work (smooth scroll to sections)
- [ ] All external links open in new tab
- [ ] Resume download works
- [ ] Filter chips filter projects correctly
- [ ] All project cards link to case study / repo / demo

## Content Quality

- [ ] No paragraph > 3 lines
- [ ] No generic claims without linked proof
- [ ] Every flagship has ≥ 4 proof badges
- [ ] Every badge links to an artifact (or placeholder)
- [ ] No "welcome to my website" or bio fluff
- [ ] About section ≤ 2 sentences

## Proof Section

- [ ] ≥ 6 proof artifacts displayed
- [ ] Each artifact has: title, 1-line description, link
- [ ] Links point to real files or hosted pages (or marked TODO)

## Responsiveness

- [ ] Desktop (1440px): 3-column hero cards
- [ ] Tablet (768px): 2-column or scroll
- [ ] Mobile (375px): single column, hamburger nav
- [ ] Touch targets ≥ 44px on mobile
- [ ] No horizontal scroll on any viewport

## Accessibility

- [ ] All images have alt text
- [ ] Color contrast ratio ≥ 4.5:1
- [ ] Focus states visible on all interactive elements
- [ ] Keyboard navigation works (tab through all links)
- [ ] Semantic HTML (header, main, section, footer)

## SEO

- [ ] Title tag: "Your Name — ML Systems Engineer"
- [ ] Meta description: specialization + proof mention
- [ ] Open Graph tags for social sharing
- [ ] Canonical URL set
- [ ] robots.txt allows indexing

## Data Architecture

- [ ] All content driven from single PORTFOLIO object
- [ ] PORTFOLIO includes: name, email, links, projects, badges, writing
- [ ] Adding a project = adding to PORTFOLIO object only
- [ ] No hardcoded content in HTML (except structure)

## Code Quality

- [ ] Valid HTML (W3C validator)
- [ ] Valid CSS (no errors)
- [ ] No console errors
- [ ] No dead links
- [ ] Comments for major sections

## Final Review

- [ ] Would a senior reviewer stay past 30 seconds? (honest answer)
- [ ] Can they find proof artifacts in 1 click?
- [ ] Is contact info visible immediately?
- [ ] Does it feel like an engineer built it (not a template)?

---

# D) FLAGSHIP PROJECT CASE STUDY OUTLINES

---

## Case Study 1: FraudShield

### URL
`/work/fraudshield`

### Structure

**A) Value Claim**
Real-time fraud detection that handles the hard part: ground truth arrives 30 days late.

**B) System Diagram**
```
Transaction Stream → Feature Store → Scoring Service → Prediction Log
                                           ↓
                               Label Reconciler (30-day lag)
                                           ↓
                                    Drift Monitor → Retrain Trigger
```

**C) Constraints**
- Latency: < 50ms p95
- Throughput: 10,000 TPS
- Label delay: 30–90 days
- Class imbalance: 0.1% fraud rate

**D) What I Built**
- Streaming feature pipeline (Kafka → Redis)
- XGBoost model with entity embeddings
- FastAPI scoring service
- Delayed label reconciliation pipeline
- Drift detection with PSI + KS tests
- Automated retraining trigger

**E) Evaluation**
- Baseline: XGBoost with manual features (PR-AUC 0.85)
- Final: XGBoost + embeddings (PR-AUC 0.91)
- Slices: merchant category, transaction amount, user tenure
- No slice below 0.80 PR-AUC
- CI gate: PR-AUC ≥ baseline AND no slice regression > 3%

**F) Monitoring & Drift**
- PSI on prediction distribution (alert if > 0.1)
- KS test on top 10 features (alert if > 3 fail)
- 7-day rolling fraud rate vs. historical (alert if > 20% deviation)
- Runbook: investigate → rollback or retrain → postmortem

**G) Cost/Latency**
- p50: 18ms
- p95: 47ms
- Throughput: 10,000 TPS
- Infra: ~$450/month (simulated)
- Cost per 1M predictions: ~$4.50

**H) Failure Modes**
| Mode | Mitigation |
|------|------------|
| Model scores all low-risk | Minimum fraud rate alert |
| Label pipeline breaks | Label freshness alert |
| Latency spike | Circuit breaker + rule fallback |

**I) Links**
- [Repo] [Demo] [Eval Report] [Runbook] [Postmortem]

---

## Case Study 2: SecureRAG

### URL
`/work/securerag`

### Structure

**A) Value Claim**
Enterprise RAG that assumes the model is compromised and designs permissions around that.

**B) System Diagram**
```
User Query → Input Sanitizer → Retriever (per-tenant) → Permission Filter
                                        ↓
                               Prompt Constructor → LLM → Output Validator → Response
                                        ↓
                               Tool Executor (sandboxed)
```

**C) Constraints**
- Multi-tenant isolation (mandatory)
- Latency: < 4s p95
- Cost: < $0.03/query (GPT-4)
- Security: 0% exfiltration tolerance

**D) What I Built**
- Document ingestion with per-tenant isolation
- Vector search with permission filtering
- Input sanitizer (injection pattern detection)
- Output validator (PII check, injection echo detection)
- Tool sandbox with explicit capability scoping
- Faithfulness scoring pipeline

**E) Evaluation**
- Retrieval: P@5 = 0.74, R@5 = 0.68
- Faithfulness: 94% responses grounded in retrieved chunks
- Attack success rate: 3.2% (120 test cases)
- Exfiltration success rate: 0%
- Tool abuse success rate: 0%

**F) Monitoring & Drift**
- Retrieval hit rate (alert if < 60%)
- Faithfulness score rolling average (alert if < 85%)
- Injection pattern frequency (alert if spike)
- Output validation failures (alert if > 5%)

**G) Cost/Latency**
- p50: 1.8s
- p95: 3.6s
- Cost per query (GPT-4): $0.024
- Throughput: 100 QPS

**H) Failure Modes**
| Mode | Mitigation |
|------|------------|
| Novel injection bypasses | Continuous red-teaming, anomaly monitoring |
| Hallucination | Faithfulness scoring + citation requirement |
| PII leak | Output PII scanner |

**I) Links**
- [Repo] [Attack Report] [Eval Dashboard] [Security Model] [Incident Response]

---

## Case Study 3: ShiftBench

### URL
`/work/shiftbench`

### Structure

**A) Value Claim**
A benchmark that stops hiding distribution shift. Train on 2015–2018, test on 2019–2022.

**B) Dataset Diagram**
```
ISIC Archive → Temporal Split (train: 2015-2018, test: 2019-2022)
                    ↓
            Demographic Annotation (Fitzpatrick proxy)
                    ↓
            Train/Val/Test CSVs with patient deduplication
```

**C) Constraints**
- Reproducibility: exact splits, seeded runs
- Fairness: demographic subgroup analysis
- Calibration: ECE as primary metric alongside accuracy

**D) What I Built**
- Temporal split logic with patient deduplication
- Demographic annotation pipeline (Fitzpatrick proxy)
- 3 baseline models (ResNet-50, EfficientNet-B4, ViT-B/16)
- Calibration evaluation (ECE, MCE, reliability diagrams)
- Uncertainty quantification (MC Dropout, temperature scaling)
- Ablation suite (augmentation, class balancing, data size)

**E) Evaluation**
| Model | Accuracy (2019-2022) | ECE | Demographic Gap |
|-------|----------------------|-----|-----------------|
| ResNet-50 | 0.78 | 0.12 | 8% (light vs. dark) |
| EfficientNet-B4 | 0.81 | 0.09 | 6% |
| ViT-B/16 | 0.84 | 0.07 | 5% |

**F) Negative Results**
- Mixup augmentation: no improvement, hurt calibration
- Heavy augmentation: improved accuracy, worsened ECE
- Class balancing: marginal improvement, added complexity

**G) Limitations**
1. Fitzpatrick proxy is imperfect
2. ISIC data skews toward lighter skin tones
3. Temporal shift may conflate scanner + annotation changes
4. Not for clinical deployment
5. Limited to dermoscopy images

**H) Links**
- [Repo] [Dataset Datasheet] [Results] [Negative Results] [Reproduction Guide]

---

# E) DATA STRUCTURE (PORTFOLIO OBJECT)

```javascript
const PORTFOLIO = {
  name: "Your Name",
  role: "ML Systems Engineer",
  tagline: "Systems that score in 50ms, defend against injection, and document their failures.",
  email: "hello@yourdomain.com",
  resume: "/assets/resume.pdf",
  links: {
    github: "https://github.com/yourname",
    linkedin: "https://linkedin.com/in/yourname",
    twitter: null
  },
  location: "San Francisco, PT",
  
  about: "I build ML systems that handle real-world constraints: delayed labels, adversarial inputs, and distribution shift. Every project I ship includes an eval harness, monitoring playbook, and documented failure modes.",
  
  projects: {
    flagship: [
      {
        id: "fraudshield",
        title: "FraudShield",
        lane: "MLOps",
        tagline: "Real-time fraud detection with delayed label reconciliation",
        description: "Streaming fraud scoring at 10K TPS with < 50ms latency. Handles the hard part: labels arrive 30 days late.",
        badges: ["CI-Eval", "Drift", "Cost", "Latency", "Postmortem", "Model Card"],
        links: {
          caseStudy: "/work/fraudshield",
          repo: "https://github.com/yourname/fraudshield",
          demo: "https://demo.fraudshield.dev"
        }
      },
      {
        id: "securerag",
        title: "SecureRAG",
        lane: "LLM/GenAI",
        tagline: "Enterprise RAG with defense-in-depth",
        description: "Multi-tenant document Q&A with prompt injection defense and tool sandboxing. 120+ attack tests. 0% exfil success.",
        badges: ["Attack Tests", "Faithfulness", "CI-Eval", "Cost", "Reproducible"],
        links: {
          caseStudy: "/work/securerag",
          repo: "https://github.com/yourname/securerag",
          demo: null,
          attackReport: "/proof/attack-report"
        }
      },
      {
        id: "shiftbench",
        title: "ShiftBench",
        lane: "Research",
        tagline: "Medical imaging benchmark under distribution shift",
        description: "Skin lesion classification with explicit temporal and demographic splits. Calibration-aware evaluation.",
        badges: ["Slice Metrics", "Reproducible", "Model Card", "CI-Eval"],
        links: {
          caseStudy: "/work/shiftbench",
          repo: "https://github.com/yourname/shiftbench",
          results: "/proof/shiftbench-results"
        }
      }
    ],
    supporting: [
      {
        id: "feature-store",
        title: "Feature Store Migration",
        lane: "MLOps",
        description: "Migrated batch feature pipeline to streaming. Reduced freshness from 24h to < 5 min.",
        links: { repo: "..." }
      },
      {
        id: "prompt-classifier",
        title: "Prompt Classifier",
        lane: "LLM",
        description: "Binary classifier to detect prompt injection attempts. 97% precision at 92% recall.",
        links: { repo: "..." }
      },
      {
        id: "calibration-toolkit",
        title: "Calibration Toolkit",
        lane: "Research",
        description: "Open-source library for post-hoc calibration with reliability diagrams.",
        links: { repo: "...", pypi: "..." }
      }
    ]
  },
  
  proof: [
    { id: "eval-harness", title: "Evaluation Harness", description: "Slice metrics, regression tests, CI gates.", link: "/proof/eval-harness" },
    { id: "drift-runbook", title: "Drift Runbook", description: "What to monitor, when to alert, how to respond.", link: "/proof/drift-runbook" },
    { id: "attack-suite", title: "Attack Test Suite", description: "120+ adversarial test cases.", link: "/proof/attack-suite" },
    { id: "rag-eval", title: "RAG Eval Dashboard", description: "Retrieval + faithfulness metrics.", link: "/proof/rag-eval" },
    { id: "cost-report", title: "Cost & Latency Report", description: "p50/p95, throughput, $/1k requests.", link: "/proof/cost-report" },
    { id: "model-card", title: "Model Card", description: "Performance, limitations, failure modes.", link: "/proof/model-card" },
    { id: "datasheet", title: "Dataset Datasheet", description: "Collection, biases, splits.", link: "/proof/datasheet" },
    { id: "postmortem", title: "Incident Postmortem", description: "Intentional failure simulation.", link: "/proof/postmortem" }
  ],
  
  writing: [
    {
      id: "offline-metrics",
      title: "Why Offline Metrics Lie",
      subtitle: "An experiment in deceptive validation",
      summary: "Built a churn model that hit 0.92 AUC on holdout. Deployed it. Watched it fail.",
      lesson: "Use time-based splits from day one. Never trust random holdout again.",
      link: "/writing/offline-metrics"
    },
    {
      id: "eval-suites",
      title: "How I Design Eval Suites and CI Gates",
      subtitle: "A practical template",
      summary: "Every model I ship passes 4 gates: slice regression, calibration check, latency test, cost ceiling.",
      lesson: "Add distribution shift simulation as a 5th gate.",
      link: "/writing/eval-suites"
    },
    {
      id: "llm-security",
      title: "Security Failures in LLM Apps",
      subtitle: "What I learned from 100+ attack tests",
      summary: "Prompt injection, indirect injection, tool abuse, PII exfiltration. I tested them all.",
      lesson: "Assume the model is compromised. Design permissions around that.",
      link: "/writing/llm-security"
    }
  ],
  
  skills: {
    modeling: ["XGBoost", "PyTorch", "Transformers", "Calibration", "Uncertainty Quantification"],
    data: ["Feature engineering", "Streaming pipelines", "Data contracts", "Quality gates"],
    systems: ["FastAPI", "Redis", "Kafka", "Docker", "Kubernetes"],
    llm: ["RAG", "Prompt engineering", "Security testing", "Faithfulness evaluation"],
    infra: ["CI/CD", "Prometheus/Grafana", "Drift detection", "Evidently"]
  },
  
  experience: [
    "Deployed real-time fraud scoring system: 10K TPS, < 50ms p95, drift-aware retraining",
    "Built secure RAG pipeline: 120 attack tests, 0% data exfiltration, 94% faithfulness",
    "Designed distribution-shift benchmark: temporal + demographic splits, calibration-first",
    "Migrated feature store from batch to streaming: 24h → 5 min freshness, zero downtime",
    "Authored eval harnesses with slice metrics and CI gates for 3 production models"
  ],
  
  cta: "Open to roles in: ML Systems · LLM Engineering · MLOps · Applied Research"
};
```

---

# F) NEXT STEPS

1. **Approve content** — Review and finalize copy
2. **Build HTML/CSS/JS** — Single-page or minimal multi-page
3. **Populate PORTFOLIO object** — Real links, real projects
4. **Deploy** — Vercel, Netlify, or GitHub Pages
5. **QA against checklist** — Every item must pass

---

**END OF SPECIFICATION**

Ready to build when you are.
