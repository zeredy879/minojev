"""Multi-domain decision data with exact, auditable teachers.

Six operational domains, each request carrying one question of every primitive
(choice, boolean, score). Rules are applied to structured fields in the state
(thresholds, path names, counters), so the teacher distribution is exact while
the language varies. This is the training distribution for the post-trained
Qwen backbones and the source of the playground cases.
"""

from __future__ import annotations

import random

from .types import Request, make_boolean_question, make_choice_question, make_score_question

DOMAIN_NAMES = ("support", "moderation", "code-review", "refunds", "triage", "leads")
LEVELS = ("very low", "low", "medium", "high", "very high")

SUPPORT_INTENTS = {
    "access": [
        "Cannot log in after a password reset and the reset email never arrived.",
        "Account says locked after too many attempts; the customer needs access today.",
        "Two-factor codes stopped arriving and the customer is locked out.",
    ],
    "billing": [
        "Charged twice for the same monthly subscription.",
        "Invoice total does not match the quote; the customer disputes the amount.",
        "Card was declined but the plan still shows as downgraded.",
    ],
    "deliverability": [
        "Password reset and notification emails to this domain are bouncing.",
        "Customer says our emails land in spam since the domain change.",
        "Team emails from our platform are delayed by hours for this account.",
    ],
    "retention": [
        "Customer asks how to cancel and export data before the renewal date.",
        "Long-time customer is evaluating competitors and wants a retention offer.",
        "Account owner is asking to downgrade after the price increase.",
    ],
}

CODE_PATH_POOL = ("auth", "billing", "payments", "migrations", "api", "docs", "ui", "tests", "infra", "search")
CODE_SENSITIVE = {"auth", "billing", "payments", "migrations"}


def _peaked(candidate_ids: list[str], answer: str, peak: float = 0.85) -> dict[str, float]:
    other = (1 - peak) / max(len(candidate_ids) - 1, 1)
    return {candidate_id: (peak if candidate_id == answer else other) for candidate_id in candidate_ids}


def _level_target(level: int, levels: int = 5, peak: float = 0.8) -> dict[str, float]:
    other = (1 - peak) / (levels - 1)
    return {str(index): (peak if index == level else other) for index in range(levels)}


def _split_offset(split: str) -> int:
    return {"train": 0, "dev": 4_000_000, "test": 8_000_000}.get(split, 12_000_000)


def _support(rng: random.Random) -> tuple[dict, list]:
    intents = list(SUPPORT_INTENTS)
    intent = rng.choice(intents)
    ticket = rng.choice(SUPPORT_INTENTS[intent])
    tier = rng.choice(["basic", "pro", "enterprise"])
    sentiment = round(rng.uniform(-1, 1), 2)
    contacts = rng.randint(0, 5)
    state = {
        "ticket": ticket,
        "customer_tier": tier,
        "sentiment": sentiment,
        "prior_contacts_7d": contacts,
        "sla_hours_left": rng.randint(1, 72),
    }
    escalate = tier == "enterprise" or sentiment <= -0.5 or contacts >= 3
    churn = min(4, max(0, int(round((1 - sentiment) * 2 + (1 if tier == "basic" else 0) + contacts / 2 - 1))))
    questions = [
        make_choice_question("queue", "Which support queue should handle this ticket?", {name: name for name in intents}),
        make_boolean_question(
            "escalate",
            "Should this ticket be escalated to a human agent now?",
            {"true": "tier, sentiment, or contact history crosses the escalation rule", "false": "no escalation trigger is met"},
        ),
        make_score_question("churn", "What is the churn risk for this customer?", list(LEVELS)),
    ]
    gold = {"queue": intent, "escalate": escalate, "churn": churn}
    teacher = {
        "queue": _peaked(intents, intent),
        "escalate": {"true": 0.9 if escalate else 0.1, "false": 0.1 if escalate else 0.9},
        "churn": _level_target(churn),
    }
    return state, list(zip(questions, gold.values(), teacher.values()))


def _moderation(rng: random.Random) -> tuple[dict, list]:
    spam_links = rng.randint(0, 4)
    threat_terms = rng.randint(0, 3)
    hate_terms = rng.randint(0, 3)
    nsfw_score = round(rng.uniform(0, 1), 2)
    reports = rng.randint(0, 10)
    age = rng.randint(1, 3000)
    if nsfw_score >= 0.8:
        category = "nsfw"
    elif threat_terms > 0 or hate_terms > 0:
        category = "harassment"
    elif spam_links >= 2:
        category = "spam"
    else:
        category = "clean"
    block = category != "clean" or reports >= 5
    severity = 0
    if category == "spam":
        severity = 1
    elif category == "harassment":
        severity = 2 + (1 if threat_terms > 1 else 0)
    elif category == "nsfw":
        severity = 3
    if reports >= 5 or age < 7:
        severity = min(4, severity + 1)
    state = {
        "post_text": "user submitted content for review",
        "signals": {"spam_links": spam_links, "threat_terms": threat_terms, "hate_terms": hate_terms, "nsfw_score": nsfw_score},
        "author_reports": reports,
        "account_age_days": age,
    }
    categories = ["clean", "spam", "harassment", "nsfw"]
    questions = [
        make_choice_question("category", "Which policy category applies to this content?", {name: name for name in categories}),
        make_boolean_question(
            "block",
            "Should this content be blocked?",
            {"true": "the content violates policy or the author has too many reports", "false": "the content can stay visible"},
        ),
        make_score_question("severity", "How severe is this policy issue?", list(LEVELS)),
    ]
    gold = {"category": category, "block": block, "severity": severity}
    teacher = {
        "category": _peaked(categories, category),
        "block": {"true": 0.9 if block else 0.1, "false": 0.1 if block else 0.9},
        "severity": _level_target(severity),
    }
    return state, list(zip(questions, gold.values(), teacher.values()))


def _code_review(rng: random.Random) -> tuple[dict, list]:
    files = rng.randint(1, 25)
    lines = rng.randint(5, 1200)
    touches = rng.sample(CODE_PATH_POOL, rng.randint(1, 3))
    tests_added = rng.randint(0, 12)
    coverage_delta = round(rng.uniform(-3, 3), 2)
    experience = rng.randint(0, 10)
    sensitive = bool(set(touches) & CODE_SENSITIVE)
    if sensitive or lines > 800:
        risk = "high"
    elif lines > 200 or coverage_delta < -1:
        risk = "medium"
    else:
        risk = "low"
    second_reviewer = risk == "high" or experience < 3
    complexity = min(4, max(0, (0 if lines < 50 else 1 if lines < 200 else 2 if lines < 600 else 3) + (1 if files > 12 else 0)))
    state = {
        "files_changed": files,
        "lines_changed": lines,
        "touches": touches,
        "tests_added": tests_added,
        "coverage_delta": coverage_delta,
        "author_experience_months": experience,
    }
    risks = ["low", "medium", "high"]
    questions = [
        make_choice_question("risk", "What is the review risk level of this change?", {name: name for name in risks}),
        make_boolean_question(
            "second-reviewer",
            "Does this change require a second reviewer?",
            {"true": "risk or author experience requires another reviewer", "false": "a single reviewer is enough"},
        ),
        make_score_question("complexity", "How complex is this change to review?", list(LEVELS)),
    ]
    gold = {"risk": risk, "second-reviewer": second_reviewer, "complexity": complexity}
    teacher = {
        "risk": _peaked(risks, risk),
        "second-reviewer": {"true": 0.9 if second_reviewer else 0.1, "false": 0.1 if second_reviewer else 0.9},
        "complexity": _level_target(complexity),
    }
    return state, list(zip(questions, gold.values(), teacher.values()))


def _refunds(rng: random.Random) -> tuple[dict, list]:
    status = rng.choice(["delivered", "delivered", "delivered", "in_transit", "lost"])
    days = rng.randint(0, 90)
    reason = rng.choice(["changed_mind", "damaged", "wrong_item", "late"])
    price = rng.randint(10, 900)
    tier = rng.choice(["basic", "pro", "enterprise"])
    window = 30
    if status == "lost":
        action = "escalate"
    elif status == "in_transit":
        action = "request_info"
    elif reason in {"damaged", "wrong_item"}:
        action = "approve"
    elif days <= window:
        action = "approve"
    elif tier == "enterprise" or price > 500:
        action = "escalate"
    else:
        action = "deny"
    eligible = status == "delivered" and (reason in {"damaged", "wrong_item"} or days <= window)
    risk = 4 if action == "deny" else 3 if action in {"escalate", "request_info"} else 1 if reason == "changed_mind" else 0
    state = {
        "order_status": status,
        "days_since_delivery": days,
        "reason": reason,
        "item_price_usd": price,
        "customer_tier": tier,
        "return_window_days": window,
    }
    actions = ["approve", "deny", "escalate", "request_info"]
    questions = [
        make_choice_question("action", "What should the agent do with this refund request?", {name: name for name in actions}),
        make_boolean_question(
            "eligible",
            "Is the order eligible for a standard refund?",
            {"true": "status, reason, and window rules are satisfied", "false": "at least one refund rule fails"},
        ),
        make_score_question("escalation-risk", "How likely is this decision to cause a complaint?", list(LEVELS)),
    ]
    gold = {"action": action, "eligible": eligible, "escalation-risk": risk}
    teacher = {
        "action": _peaked(actions, action),
        "eligible": {"true": 0.9 if eligible else 0.1, "false": 0.1 if eligible else 0.9},
        "escalation-risk": _level_target(risk),
    }
    return state, list(zip(questions, gold.values(), teacher.values()))


def _triage(rng: random.Random) -> tuple[dict, list]:
    age = rng.randint(1, 92)
    temperature = round(rng.uniform(36.0, 40.5), 1)
    heart_rate = rng.randint(50, 140)
    spo2 = rng.randint(85, 100)
    pain = rng.randint(0, 10)
    duration = rng.randint(1, 240)
    emergency = spo2 < 92 or temperature >= 39.5 or heart_rate > 120 or pain >= 8
    urgent = not emergency and (spo2 < 95 or temperature >= 38.5 or pain >= 5 or heart_rate > 105)
    soon = not emergency and not urgent and (temperature >= 37.8 or pain >= 3 or duration > 72)
    urgency = "emergency" if emergency else "urgent" if urgent else "soon" if soon else "self_care"
    severity = 4 if emergency else 2 if urgent else 1 if soon else 0
    if age >= 75 or age <= 3:
        severity = min(4, severity + 1)
    state = {
        "age": age,
        "temperature_c": temperature,
        "heart_rate_bpm": heart_rate,
        "spo2_percent": spo2,
        "pain_level": pain,
        "symptom_duration_hours": duration,
    }
    levels = ["self_care", "soon", "urgent", "emergency"]
    questions = [
        make_choice_question("urgency", "What is the intake urgency for this patient?", {name: name for name in levels}),
        make_boolean_question(
            "red-flag",
            "Does any red-flag vital sign trigger apply?",
            {"true": "at least one emergency threshold is crossed", "false": "no emergency threshold is crossed"},
        ),
        make_score_question("severity", "How severe does this intake look?", list(LEVELS)),
    ]
    gold = {"urgency": urgency, "red-flag": emergency, "severity": severity}
    teacher = {
        "urgency": _peaked(levels, urgency),
        "red-flag": {"true": 0.9 if emergency else 0.1, "false": 0.1 if emergency else 0.9},
        "severity": _level_target(severity),
    }
    return state, list(zip(questions, gold.values(), teacher.values()))


def _leads(rng: random.Random) -> tuple[dict, list]:
    size = rng.randint(5, 5000)
    budget = rng.choice([2_000, 8_000, 25_000, 60_000, 150_000])
    timeline = rng.choice([7, 14, 30, 90, 180])
    engagement = round(rng.uniform(0, 1), 2)
    industry = rng.choice(["saas", "retail", "health", "finance", "manufacturing"])
    if budget >= 50_000 and timeline <= 30:
        tier = "hot"
    elif engagement >= 0.6 or budget >= 20_000:
        tier = "warm"
    else:
        tier = "cold"
    enterprise = size >= 1000 and budget >= 50_000
    fit = min(4, max(0, (2 if budget >= 50_000 else 1 if budget >= 20_000 else 0) + (1 if engagement >= 0.7 else 0) + (1 if timeline <= 14 else 0) - (1 if size < 50 else 0)))
    state = {
        "company_size": size,
        "budget_usd": budget,
        "timeline_days": timeline,
        "engagement_score": engagement,
        "industry": industry,
    }
    tiers = ["hot", "warm", "cold"]
    questions = [
        make_choice_question("tier", "How should this lead be classified?", {name: name for name in tiers}),
        make_boolean_question(
            "enterprise",
            "Should this lead be routed to the enterprise team?",
            {"true": "company size and budget cross the enterprise rule", "false": "the enterprise rule is not met"},
        ),
        make_score_question("fit", "How well does this lead fit the product?", list(LEVELS)),
    ]
    gold = {"tier": tier, "enterprise": enterprise, "fit": fit}
    teacher = {
        "tier": _peaked(tiers, tier),
        "enterprise": {"true": 0.9 if enterprise else 0.1, "false": 0.1 if enterprise else 0.9},
        "fit": _level_target(fit),
    }
    return state, list(zip(questions, gold.values(), teacher.values()))


GENERATORS = {
    "support": _support,
    "moderation": _moderation,
    "code-review": _code_review,
    "refunds": _refunds,
    "triage": _triage,
    "leads": _leads,
}

SCENARIOS = {
    "support": "customer support ticket routing and escalation",
    "moderation": "content moderation category and blocking",
    "code-review": "pull request risk triage",
    "refunds": "refund policy decision",
    "triage": "intake urgency triage (operational demo, not medical advice)",
    "leads": "sales lead qualification",
}


def generate_request(index: int, seed: int = 17, split: str = "train", domain: str | None = None) -> Request:
    rng = random.Random((seed * 3_571_111) ^ (index + _split_offset(split)))
    name = domain or DOMAIN_NAMES[index % len(DOMAIN_NAMES)]
    if name not in GENERATORS:
        raise ValueError(f"unknown domain {name!r}; choose from {DOMAIN_NAMES}")
    state, items = GENERATORS[name](rng)
    questions = []
    gold: dict = {}
    teacher: dict = {}
    for question, answer, target in items:
        question.family = name
        questions.append(question)
        gold[question.qid] = answer
        teacher[question.qid] = target
    return Request(id=f"{split}-{name}-{index:06d}", state=state, questions=questions, gold=gold, teacher=teacher)


def generate_split(count: int, seed: int = 17, split: str = "train", domain: str | None = None) -> list[Request]:
    return [generate_request(index, seed=seed, split=split, domain=domain) for index in range(count)]
