/**
 * PORTFOLIO DATA
 * All website content is driven from this single object.
 * Update this file to customize your portfolio.
 */

const PORTFOLIO = {
    // ============================================
    // Personal Info
    // ============================================
    name: "Abdul-Sobur Ayinde",
    role: "ML Systems Engineer",
    headline: "I ship ML systems that survive production.",
    tagline: "Systems that score in 50ms, defend against injection, and document their failures.",
    email: "hello@yourdomain.com",
    resume: "/assets/resume.pdf",
    
    links: {
        github: "https://github.com/Croesus245/My-POrt",
        linkedin: "https://linkedin.com/in/abdulsobur-ayinde",
        twitter: null
    },
    
    location: "San Francisco, PT",
    
    about: "I build ML systems that handle real-world constraints: delayed labels, adversarial inputs, and distribution shift. Every project I ship includes an eval harness, monitoring playbook, and documented failure modes.",
    
    cta: "Open to roles in: ML Systems · LLM Engineering · MLOps · Applied Research",

    // ============================================
    // Flagship Projects
    // ============================================
    flagship: [
        {
            id: "fraudshield",
            title: "FraudShield",
            lane: "MLOps",
            tagline: "Real-time fraud detection with delayed label reconciliation",
            description: "Streaming fraud scoring at 10K TPS with < 50ms latency. Handles the hard part: labels arrive 30 days late. Includes drift detection, retraining triggers, and a documented incident simulation.",
            badges: ["CI-Eval", "Drift", "Cost", "Latency", "Postmortem", "Model Card"],
            links: {
                caseStudy: "work/fraudshield.html",
                repo: "https://github.com/Croesus245/My-POrt/tree/main/fraudshield",
                demo: "work/fraudshield.html#demo"
            }
        },
        {
            id: "securerag",
            title: "SecureRAG",
            lane: "LLM/GenAI",
            tagline: "Enterprise RAG with defense-in-depth",
            description: "Multi-tenant document Q&A with prompt injection defense, data exfiltration prevention, and tool sandboxing. 120+ attack tests. 0% exfil success rate. Every response scored for faithfulness.",
            badges: ["Attack Tests", "Faithfulness", "CI-Eval", "Cost", "Reproducible"],
            links: {
                caseStudy: "work/securerag.html",
                repo: "https://github.com/Croesus245/My-POrt/tree/main/securerag",
                demo: null,
                attackReport: "https://github.com/Croesus245/My-POrt/blob/main/securerag/docs/attack_report.md"
            }
        },
        {
            id: "shiftbench",
            title: "ShiftBench",
            lane: "Research",
            tagline: "Medical imaging benchmark under distribution shift",
            description: "Skin lesion classification with explicit temporal (2015–2018 → 2019–2022) and demographic splits. Calibration-aware evaluation. Honest negative results section.",
            badges: ["Slice Metrics", "Reproducible", "Model Card", "CI-Eval"],
            links: {
                caseStudy: "work/shiftbench.html",
                repo: "https://github.com/Croesus245/My-POrt/tree/main/shiftbench",
                results: "https://github.com/Croesus245/My-POrt/blob/main/shiftbench/results/README.md"
            }
        }
    ],

    // ============================================
    // Supporting Projects
    // ============================================
    supporting: [
        {
            id: "feature-store",
            title: "Feature Store Migration",
            lane: "MLOps",
            description: "Migrated batch feature pipeline to streaming (Feast → custom Redis). Reduced feature freshness from 24h to < 5 min. Zero downtime cutover.",
            badges: ["Architecture", "Zero Downtime"],
            links: {
                repo: "https://github.com/Croesus245/My-POrt/tree/main/feature-store-migration"
            }
        },
        {
            id: "prompt-classifier",
            title: "Prompt Classifier",
            lane: "LLM/GenAI",
            description: "Binary classifier to detect prompt injection attempts before they reach the LLM. 97% precision at 92% recall on held-out attack corpus.",
            badges: ["Model Card", "Eval Suite"],
            links: {
                repo: "https://github.com/Croesus245/My-POrt/tree/main/prompt-classifier"
            }
        },
        {
            id: "calibration-toolkit",
            title: "Calibration Toolkit",
            lane: "Research",
            description: "Open-source library for post-hoc calibration (temperature scaling, isotonic regression) with reliability diagrams and ECE reporting.",
            badges: ["PyPI", "Docs"],
            links: {
                repo: "https://github.com/Croesus245/My-POrt/tree/main/calibration-toolkit",
                pypi: "https://pypi.org/project/calibration-toolkit/"
            }
        }
    ],

    // ============================================
    // Proof Artifacts
    // ============================================
    proof: [
        {
            id: "eval-harness",
            icon: "📊",
            title: "Evaluation Harness",
            description: "Slice metrics, regression tests, CI gates. Every model tested against 12 subgroups.",
            link: "#"
        },
        {
            id: "drift-runbook",
            icon: "📈",
            title: "Drift Runbook",
            description: "What to monitor, when to alert, how to respond. Decision tree for retrain vs. rollback.",
            link: "#"
        },
        {
            id: "attack-suite",
            icon: "🛡️",
            title: "Attack Test Suite",
            description: "120+ adversarial test cases: injection, exfil, tool abuse, PII extraction.",
            link: "#"
        },
        {
            id: "rag-eval",
            icon: "🎯",
            title: "RAG Eval Dashboard",
            description: "Retrieval precision/recall, faithfulness scores, grounding verification.",
            link: "#"
        },
        {
            id: "cost-report",
            icon: "💰",
            title: "Cost & Latency Report",
            description: "p50/p95 latency, throughput, $/1k requests, optimization levers.",
            link: "#"
        },
        {
            id: "model-card",
            icon: "📋",
            title: "Model Card",
            description: "Performance by slice, known limitations, failure modes, intended use.",
            link: "#"
        },
        {
            id: "datasheet",
            icon: "📁",
            title: "Dataset Datasheet",
            description: "Collection methodology, known biases, split logic, demographic annotations.",
            link: "#"
        },
        {
            id: "postmortem",
            icon: "🔥",
            title: "Incident Postmortem",
            description: "Intentional failure simulation. Root cause, response, prevention.",
            link: "#"
        }
    ],

    // ============================================
    // Writing
    // ============================================
    writing: [
        {
            id: "offline-metrics",
            title: "Why Offline Metrics Lie",
            subtitle: "An experiment in deceptive validation",
            summary: "Built a churn model that hit 0.92 AUC on holdout. Deployed it. Watched it fail. Here's why stratified splits hide temporal drift, and how I redesigned the eval suite.",
            lesson: "Use time-based splits from day one. Never trust random holdout again.",
            link: "#"
        },
        {
            id: "eval-suites",
            title: "How I Design Eval Suites and CI Gates",
            subtitle: "A practical template",
            summary: "Every model I ship passes 4 gates: slice regression, calibration check, latency test, and cost ceiling. Here's the exact YAML config and why each gate exists.",
            lesson: "Add distribution shift simulation as a 5th gate.",
            link: "#"
        },
        {
            id: "llm-security",
            title: "Security Failures in LLM Apps",
            subtitle: "What I learned from 100+ attack tests",
            summary: "Prompt injection, indirect injection via documents, tool abuse, PII exfiltration. I tested them all. Here's what worked, what didn't, and the defense layers that survived.",
            lesson: "Assume the model is compromised. Design permissions around that.",
            link: "#"
        }
    ],

    // ============================================
    // Skills
    // ============================================
    skills: {
        Modeling: ["XGBoost", "PyTorch", "Transformers", "Calibration", "UQ"],
        Data: ["Feature Engineering", "Streaming", "Data Contracts", "Quality Gates"],
        Systems: ["FastAPI", "Redis", "Kafka", "Docker", "Kubernetes"],
        LLM: ["RAG", "Prompt Engineering", "Security Testing", "Faithfulness Eval"],
        Infra: ["CI/CD", "Prometheus", "Grafana", "Evidently", "Drift Detection"]
    },

    // ============================================
    // Experience
    // ============================================
    experience: [
        "Deployed real-time fraud scoring system: 10K TPS, < 50ms p95, drift-aware retraining",
        "Built secure RAG pipeline: 120 attack tests, 0% data exfiltration, 94% faithfulness",
        "Designed distribution-shift benchmark: temporal + demographic splits, calibration-first",
        "Migrated feature store from batch to streaming: 24h → 5 min freshness, zero downtime",
        "Authored eval harnesses with slice metrics and CI gates for 3 production models"
    ]
};

// Export for use in other files (if using modules)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PORTFOLIO;
}
