#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking
=============================================================
Ranks 100K candidates for "Senior AI Engineer — Founding Team" role.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints: CPU-only, ≤5 min, ≤16 GB RAM, no network.
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Today's date for recency calculations
TODAY = date(2026, 6, 28)

# ---------------------------------------------------------------------------
# Skill keyword dictionaries
# ---------------------------------------------------------------------------
# Each key is a canonical JD-skill; values are regex patterns to match
# against candidate skill names (case-insensitive).

REQUIRED_SKILL_PATTERNS = {
    "embeddings": r"embed|sentence.?transform|bge\b|e5\b|dense.?retriev",
    "vector_search": r"vector.?search|semantic.?search|similarity.?search|ann\b|approx.?nearest",
    "rag": r"\brag\b|retriev.?augment|retriev.?generat",
    "retrieval_systems": r"retriev|information.?retriev|\bir\b",
    "python": r"\bpython\b",
    "vector_db": r"pinecone|weaviate|qdrant|milvus|faiss|elasticsearch|opensearch|elastic.?search|open.?search|chromadb|chroma|pgvector|vector.?db|vector.?database",
    "ranking_systems": r"rank|re.?rank|learning.?to.?rank|\bltr\b",
    "recommendation": r"recommend|collaborative.?filter|content.?based.?filter",
    "eval_frameworks": r"ndcg|mrr|mean.?average.?precision|\bmap\b|a.?b.?test|offline.?eval|online.?eval|precision.?at|recall.?at",
    "nlp": r"\bnlp\b|natural.?language|text.?process|text.?mining|language.?model|transformer|bert|gpt|llm|large.?language",
    "ml_production": r"machine.?learn|ml.?eng|mlops|ml.?pipeline|model.?deploy|model.?serv|feature.?store|ml.?infra",
}

NICE_TO_HAVE_SKILL_PATTERNS = {
    "llm_finetuning": r"fine.?tun|lora|qlora|peft|rlhf|instruction.?tun|sft\b",
    "learning_to_rank": r"learning.?to.?rank|xgboost|lightgbm|\bltr\b|gradient.?boost|catboost",
    "distributed_systems": r"distribut|inference.?optim|triton|tensorrt|onnx|vllm|model.?optim|quantiz",
    "open_source": r"open.?source|github.?contrib|oss\b",
}

# Compiled regex for performance
REQUIRED_SKILL_RE = {k: re.compile(v, re.IGNORECASE) for k, v in REQUIRED_SKILL_PATTERNS.items()}
NICE_TO_HAVE_SKILL_RE = {k: re.compile(v, re.IGNORECASE) for k, v in NICE_TO_HAVE_SKILL_PATTERNS.items()}

# Patterns to match in career_history descriptions (Trap 2 detection)
DESCRIPTION_SKILL_PATTERNS = {
    "embeddings": re.compile(r"embed|sentence.?transform|dense.?retriev|vector.?encod", re.IGNORECASE),
    "vector_search": re.compile(r"vector.?search|semantic.?search|similarity.?search|nearest.?neighbor|ann\b", re.IGNORECASE),
    "rag": re.compile(r"\brag\b|retriev.?augment|retriev.?generat", re.IGNORECASE),
    "retrieval_systems": re.compile(r"retriev.?system|retriev.?pipeline|search.?system|search.?engine|search.?infra", re.IGNORECASE),
    "vector_db": re.compile(r"pinecone|weaviate|qdrant|milvus|faiss|elasticsearch|opensearch|chromadb|pgvector|vector.?db|vector.?index", re.IGNORECASE),
    "ranking_systems": re.compile(r"rank.?system|rank.?model|re.?rank|learning.?to.?rank|search.?rank|result.?rank", re.IGNORECASE),
    "recommendation": re.compile(r"recommend.?system|recommend.?engine|collaborative.?filter|content.?based|personali[sz]", re.IGNORECASE),
    "eval_frameworks": re.compile(r"ndcg|mrr|mean.?average.?precision|a.?b.?test|offline.?eval|online.?eval|evaluat.?rank|evaluat.?retriev", re.IGNORECASE),
    "nlp": re.compile(r"\bnlp\b|natural.?language|text.?classif|named.?entity|sentiment|language.?model|transformer|bert|gpt", re.IGNORECASE),
    "ml_production": re.compile(r"production.?ml|deploy.?model|model.?serv|ml.?pipeline|mlops|feature.?store|inference.?pipeline|ml.?infra", re.IGNORECASE),
    "llm_finetuning": re.compile(r"fine.?tun|lora|qlora|peft|rlhf|instruction.?tun", re.IGNORECASE),
    "distributed_systems": re.compile(r"distribut.?system|distribut.?train|distribut.?infer|model.?parallel|data.?parallel|inference.?optim", re.IGNORECASE),
}

# ---------------------------------------------------------------------------
# Title classification
# ---------------------------------------------------------------------------
TITLE_HIGH = re.compile(
    r"ml\b|machine.?learn|ai\b|artificial.?intell|nlp\b|natural.?language"
    r"|data.?scien|search.?eng|recommend.?eng|retriev.?eng|research.?eng"
    r"|applied.?scien|deep.?learn|computer.?vision|cv.?eng"
    r"|staff.?eng.*(?:ml|ai|data)|principal.*(?:ml|ai|data)"
    r"|lead.*(?:ml|ai|data)|senior.*(?:ml|ai|data)",
    re.IGNORECASE
)
TITLE_MEDIUM = re.compile(
    r"software.?eng|backend.?eng|data.?eng|platform.?eng|full.?stack"
    r"|devops|site.?reliab|sre\b|infra.?eng|cloud.?eng"
    r"|tech.?lead|engineering.?manager|architect"
    r"|developer|programmer|sde\b",
    re.IGNORECASE
)
TITLE_LOW = re.compile(
    r"data.?analyst|bi\b|business.?intel|analytics.?eng|qa\b|test.?eng"
    r"|product.?manager|product.?owner|scrum",
    re.IGNORECASE
)
TITLE_DISQUALIFIED = re.compile(
    r"marketing.?manage|hr.?manage|human.?resource|content.?write|account"
    r"|graphic.?design|operation.?manage|sales.?exec|customer.?support"
    r"|civil.?eng|mechanical.?eng|electrical.?eng|chemical.?eng"
    r"|teacher|professor|nurse|doctor|lawyer|legal"
    r"|receptionist|admin.?assist|office.?manage|clerk"
    r"|retail|store.?manage|warehouse|logistics.?manage"
    r"|chef|cook|driver|security.?guard",
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Consulting companies (services-only career = penalty)
# ---------------------------------------------------------------------------
CONSULTING_COMPANIES = re.compile(
    r"\btcs\b|tata.?consult|infosys|wipro|accenture|cognizant|capgemini"
    r"|hcl\b|tech.?mahindra|mindtree|mphasis|l.?t.?infotech|lt.?infotech"
    r"|persistent.?system|hexaware|zensar|cyient|niit|kpit"
    r"|deloitte|pwc|kpmg|ernst.?young|ey\b|mckinsey|bain|bcg",
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Preferred locations
# ---------------------------------------------------------------------------
PREFERRED_LOCATIONS = re.compile(
    r"pune|noida|delhi|ncr|gurgaon|gurugram|hyderabad|mumbai|bengaluru|bangalore",
    re.IGNORECASE
)
INDIA_CHECK = re.compile(r"\bindia\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Education fields
# ---------------------------------------------------------------------------
TECH_FIELDS = re.compile(
    r"computer|software|information.?tech|data.?scien|artificial.?intell"
    r"|machine.?learn|ece|electric|electro|math|statist"
    r"|physics|computational|informatics",
    re.IGNORECASE
)

# Proficiency weights
PROFICIENCY_MAP = {
    "beginner": 0.4,
    "intermediate": 0.6,
    "advanced": 0.85,
    "expert": 1.0,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def resolve_path(path, default_name):
    """Resolve a file path relative to the script directory when needed."""
    if path is None or path == "":
        path = default_name

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_dir = Path(__file__).resolve().parent
    return (project_dir / candidate).resolve()


def load_candidates(path):
    """Stream-parse candidates.jsonl — one JSON per line."""
    candidates = []
    resolved_path = resolve_path(path, "candidates.jsonl")
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


# =============================================================================
# COMPONENT 1: SKILL MATCH SCORE
# =============================================================================

def compute_skill_score(candidate):
    """
    Score how well the candidate's skills match JD requirements.
    Returns (score 0-1, matched_skills list, description_matches list).
    """
    skills = candidate.get("skills", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    # --- Score explicit skill listings ---
    required_matches = {}  # canonical_name -> best_score
    nice_matches = {}

    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        prof_weight = PROFICIENCY_MAP.get(prof, 0.4)
        duration_weight = min(1.0, duration / 24.0) if duration > 0 else 0.0
        endorse_weight = min(1.0, math.log(1 + endorsements) / math.log(50)) if endorsements > 0 else 0.0

        # Keyword stuffer detection: expert with 0 duration and 0 endorsements
        if prof == "expert" and duration == 0 and endorsements == 0:
            trust_score = 0.02  # near zero — likely stuffed
        elif duration == 0 and endorsements == 0:
            trust_score = 0.05  # very low trust
        else:
            trust_score = prof_weight * (0.4 + 0.35 * duration_weight + 0.25 * endorse_weight)

        # Assessment bonus
        for assess_name, assess_score in assessment_scores.items():
            if assess_name.lower() == name.lower() or name.lower() in assess_name.lower():
                if assess_score > 70:
                    trust_score *= 1.25
                elif assess_score > 50:
                    trust_score *= 1.1
                break

        # Match against required patterns
        for canon_name, regex in REQUIRED_SKILL_RE.items():
            if regex.search(name):
                current_best = required_matches.get(canon_name, 0)
                required_matches[canon_name] = max(current_best, trust_score)

        # Match against nice-to-have patterns
        for canon_name, regex in NICE_TO_HAVE_SKILL_RE.items():
            if regex.search(name):
                current_best = nice_matches.get(canon_name, 0)
                nice_matches[canon_name] = max(current_best, trust_score)

    # --- Score career description matches (Trap 2 — plain-language tier 5s) ---
    description_matches = set()
    career_history = candidate.get("career_history", [])
    all_descriptions = " ".join(
        role.get("description", "") for role in career_history
    )
    # Also include headline and summary
    all_descriptions += " " + candidate.get("profile", {}).get("headline", "")
    all_descriptions += " " + candidate.get("profile", {}).get("summary", "")

    for canon_name, regex in DESCRIPTION_SKILL_PATTERNS.items():
        if regex.search(all_descriptions):
            description_matches.add(canon_name)
            # Add partial credit for description-only matches (not as strong as explicit skill)
            if canon_name in REQUIRED_SKILL_PATTERNS and canon_name not in required_matches:
                required_matches[canon_name] = 0.35  # partial credit
            elif canon_name in NICE_TO_HAVE_SKILL_PATTERNS and canon_name not in nice_matches:
                nice_matches[canon_name] = 0.3

    # --- Compute final skill score ---
    total_required = len(REQUIRED_SKILL_PATTERNS)
    total_nice = len(NICE_TO_HAVE_SKILL_PATTERNS)

    required_sum = sum(required_matches.values())
    nice_sum = sum(nice_matches.values())

    # Weighted: required skills are worth 80%, nice-to-have 20%
    if total_required > 0:
        required_score = min(1.0, required_sum / (total_required * 0.55))  # don't need ALL at max
    else:
        required_score = 0.0

    if total_nice > 0:
        nice_score = min(1.0, nice_sum / (total_nice * 0.4))
    else:
        nice_score = 0.0

    score = 0.80 * required_score + 0.20 * nice_score

    matched = list(required_matches.keys()) + list(nice_matches.keys())
    return min(1.0, score), matched, list(description_matches)


# =============================================================================
# COMPONENT 2: CAREER TRAJECTORY SCORE
# =============================================================================

def classify_title(title):
    """Return relevance tier for a job title. Higher = more relevant."""
    if not title:
        return 0.1
    if TITLE_HIGH.search(title):
        return 1.0
    if TITLE_MEDIUM.search(title):
        return 0.55
    if TITLE_LOW.search(title):
        return 0.25
    if TITLE_DISQUALIFIED.search(title):
        return 0.05
    return 0.2  # unknown title — slight penalty


def is_consulting_company(company_name):
    if not company_name:
        return False
    return bool(CONSULTING_COMPANIES.search(company_name))


def compute_career_score(candidate):
    """
    Score career trajectory. Returns (score 0-1, career_signals dict).
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    yoe = profile.get("years_of_experience", 0)

    signals = {}

    # 1. Current title relevance
    current_title = profile.get("current_title", "")
    current_title_score = classify_title(current_title)
    signals["current_title"] = current_title
    signals["current_title_score"] = current_title_score

    # 2. Historical title relevance — weighted by recency
    title_scores = []
    for i, role in enumerate(career):
        t = classify_title(role.get("title", ""))
        # More recent roles count more
        recency_weight = 1.0 / (1 + i * 0.3)
        title_scores.append(t * recency_weight)

    avg_title = sum(title_scores) / len(title_scores) if title_scores else 0.1
    # Blend current (60%) with history (40%)
    title_component = 0.60 * current_title_score + 0.40 * avg_title

    # 3. Company type — consulting penalty
    consulting_months = 0
    product_months = 0
    total_months = 0
    for role in career:
        dur = role.get("duration_months", 0)
        total_months += dur
        if is_consulting_company(role.get("company", "")):
            consulting_months += dur
        else:
            product_months += dur

    if total_months > 0:
        consulting_ratio = consulting_months / total_months
    else:
        consulting_ratio = 0

    # Only-consulting career = 0.4x, mixed = proportional
    if consulting_ratio > 0.9:
        company_mult = 0.40
    elif consulting_ratio > 0.7:
        company_mult = 0.55
    elif consulting_ratio > 0.5:
        company_mult = 0.70
    elif consulting_ratio > 0.3:
        company_mult = 0.85
    else:
        company_mult = 1.0

    signals["consulting_ratio"] = consulting_ratio

    # 4. Experience sweet spot
    if yoe < 2:
        exp_score = 0.3
    elif yoe < 3:
        exp_score = 0.5
    elif yoe < 4:
        exp_score = 0.7
    elif yoe < 5:
        exp_score = 0.85
    elif yoe <= 9:
        exp_score = 1.0
    elif yoe <= 12:
        exp_score = 0.9
    else:
        exp_score = 0.75

    signals["yoe"] = yoe
    signals["exp_score"] = exp_score

    # 5. Job-hopping detection (last 3 roles)
    recent_tenures = [r.get("duration_months", 0) for r in career[:3]]
    if len(recent_tenures) >= 2:
        avg_tenure = sum(recent_tenures) / len(recent_tenures)
        if avg_tenure < 12:
            hop_penalty = 0.6
        elif avg_tenure < 18:
            hop_penalty = 0.75
        elif avg_tenure < 24:
            hop_penalty = 0.9
        else:
            hop_penalty = 1.0
    else:
        hop_penalty = 1.0

    signals["avg_recent_tenure"] = sum(recent_tenures) / len(recent_tenures) if recent_tenures else 0

    # 6. Recency — is the most recent role ML/AI-relevant?
    if career:
        most_recent_title_score = classify_title(career[0].get("title", ""))
        if most_recent_title_score >= 0.8:
            recency_mult = 1.1
        elif most_recent_title_score >= 0.5:
            recency_mult = 1.0
        elif most_recent_title_score >= 0.2:
            recency_mult = 0.7
        else:
            recency_mult = 0.35  # most recent role is completely unrelated
    else:
        recency_mult = 0.5

    signals["recency_mult"] = recency_mult

    # 7. Career description substance — scan for ML/retrieval/ranking work
    all_desc = " ".join(r.get("description", "") for r in career)
    desc_substance = 0.0
    desc_hits = []
    for name, regex in DESCRIPTION_SKILL_PATTERNS.items():
        if regex.search(all_desc):
            desc_substance += 0.08
            desc_hits.append(name)
    desc_substance = min(0.3, desc_substance)  # cap bonus
    signals["desc_hits"] = desc_hits

    # 8. Industry relevance
    current_industry = profile.get("current_industry", "").lower()
    tech_industries = ["software", "internet", "saas", "ai", "ml", "fintech",
                       "e-commerce", "ecommerce", "technology", "data", "analytics",
                       "hr tech", "hrtech", "marketplace", "cloud", "search"]
    industry_bonus = 0.0
    for ti in tech_industries:
        if ti in current_industry:
            industry_bonus = 0.1
            break

    # Assemble career score
    raw = title_component * company_mult * exp_score * hop_penalty * recency_mult
    raw = raw + desc_substance + industry_bonus
    score = min(1.0, max(0.0, raw))

    return score, signals


# =============================================================================
# COMPONENT 3: EDUCATION SCORE
# =============================================================================

def compute_education_score(candidate):
    """Score education. Returns (score 0-1, best_institution)."""
    education = candidate.get("education", [])
    if not education:
        return 0.4, "No education listed"

    tier_scores = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "unknown": 0.5}

    best_score = 0.0
    best_inst = ""
    for edu in education:
        tier = edu.get("tier", "unknown")
        base = tier_scores.get(tier, 0.5)

        # Field bonus
        field = edu.get("field_of_study", "")
        if TECH_FIELDS.search(field):
            base += 0.15

        # Degree level bonus
        degree = edu.get("degree", "").lower()
        if "ph.d" in degree or "phd" in degree:
            base += 0.1
        elif "m.tech" in degree or "m.sc" in degree or "m.e." in degree or "ms" in degree or "mba" in degree:
            base += 0.05

        if base > best_score:
            best_score = base
            best_inst = edu.get("institution", "Unknown")

    return min(1.0, best_score), best_inst


# =============================================================================
# COMPONENT 4: BEHAVIORAL SIGNALS
# =============================================================================

def compute_behavioral_scores(candidate):
    """
    Returns (availability_score 0-1, engagement_multiplier 0.5-1.15, signals_dict).
    """
    rs = candidate.get("redrob_signals", {})
    signals = {}

    # --- AVAILABILITY (additive component) ---
    avail_points = 0.0

    # open_to_work
    otw = rs.get("open_to_work_flag", False)
    if otw:
        avail_points += 0.30
    signals["open_to_work"] = otw

    # last_active recency
    last_active_str = rs.get("last_active_date", "")
    if last_active_str:
        try:
            la = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_since = (TODAY - la).days
        except (ValueError, TypeError):
            days_since = 999
    else:
        days_since = 999

    if days_since < 7:
        avail_points += 0.30
    elif days_since < 30:
        avail_points += 0.25
    elif days_since < 90:
        avail_points += 0.15
    elif days_since < 180:
        avail_points += 0.05
    # else 0
    signals["days_since_active"] = days_since

    # recruiter_response_rate
    rrr = rs.get("recruiter_response_rate", 0)
    if rrr > 0.7:
        avail_points += 0.20
    elif rrr > 0.3:
        avail_points += 0.10
    signals["recruiter_response_rate"] = rrr

    # avg_response_time_hours
    art = rs.get("avg_response_time_hours", 999)
    if art < 24:
        avail_points += 0.10
    elif art < 72:
        avail_points += 0.05
    signals["avg_response_time"] = art

    # Normalize to 0-1 (max possible = 0.90)
    availability_score = min(1.0, avail_points / 0.90)

    # --- ENGAGEMENT (multiplicative component) ---
    engagement = 1.0

    # Notice period
    np_days = rs.get("notice_period_days", 90)
    if np_days <= 30:
        engagement *= 1.02
    elif np_days <= 60:
        engagement *= 0.97
    elif np_days <= 90:
        engagement *= 0.90
    else:
        engagement *= 0.78
    signals["notice_period"] = np_days

    # Interview completion
    icr = rs.get("interview_completion_rate", 0.5)
    if icr > 0.8:
        engagement *= 1.04
    elif icr < 0.5:
        engagement *= 0.90
    signals["interview_completion"] = icr

    # Offer acceptance
    oar = rs.get("offer_acceptance_rate", -1)
    if oar == -1:
        pass  # neutral
    elif oar > 0.5:
        engagement *= 1.02
    elif oar < 0.2:
        engagement *= 0.95
    signals["offer_acceptance"] = oar

    # GitHub
    gas = rs.get("github_activity_score", -1)
    if gas > 50:
        engagement *= 1.05
    elif gas > 20:
        engagement *= 1.02
    signals["github_score"] = gas

    # Saved by recruiters
    sbr = rs.get("saved_by_recruiters_30d", 0)
    if sbr > 5:
        engagement *= 1.04
    elif sbr > 3:
        engagement *= 1.02
    signals["saved_by_recruiters"] = sbr

    # Profile completeness
    pcs = rs.get("profile_completeness_score", 50)
    if pcs >= 80:
        pass  # full credit
    elif pcs >= 60:
        engagement *= 0.98
    else:
        engagement *= 0.92
    signals["profile_completeness"] = pcs

    # Verification
    ve = rs.get("verified_email", False)
    vp = rs.get("verified_phone", False)
    if not ve and not vp:
        engagement *= 0.95
    elif not ve or not vp:
        engagement *= 0.98
    signals["verified_email"] = ve
    signals["verified_phone"] = vp

    # LinkedIn
    lc = rs.get("linkedin_connected", False)
    if lc:
        engagement *= 1.01
    signals["linkedin_connected"] = lc

    # Clamp engagement
    engagement = max(0.5, min(1.15, engagement))

    return availability_score, engagement, signals


# =============================================================================
# COMPONENT 5: HONEYPOT DETECTION
# =============================================================================

def detect_honeypot(candidate):
    """
    Detect impossible/contradictory profiles.
    Returns (penalty 0.01-1.0, flags list).
    """
    flags = []
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    yoe = profile.get("years_of_experience", 0)

    # 1. Experience vs career duration mismatch
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    total_career_years = total_career_months / 12.0
    if yoe > 0 and total_career_years > 0:
        if abs(yoe - total_career_years) > 5:
            flags.append("yoe_vs_career_mismatch")

    # 2. Expert proficiency with 0 months usage
    expert_zero_count = 0
    for skill in skills:
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) == 0:
            expert_zero_count += 1
    if expert_zero_count >= 3:
        flags.append(f"expert_zero_duration_x{expert_zero_count}")
    elif expert_zero_count >= 1:
        flags.append("expert_zero_duration")

    # 3. Mass keyword stuffing — many expert skills with 0 endorsements
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 8:
        zero_endorse = sum(1 for s in expert_skills if s.get("endorsements", 0) == 0)
        if zero_endorse >= 6:
            flags.append("mass_keyword_stuffing")

    # 4. Title-skill contradiction
    current_title = profile.get("current_title", "")
    all_titles = [current_title] + [r.get("title", "") for r in career]
    has_tech_title = any(
        TITLE_HIGH.search(t) or TITLE_MEDIUM.search(t)
        for t in all_titles if t
    )
    has_expert_ml_skills = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and (REQUIRED_SKILL_RE.get("ml_production", re.compile("$^")).search(s.get("name", ""))
             or REQUIRED_SKILL_RE.get("nlp", re.compile("$^")).search(s.get("name", ""))
             or REQUIRED_SKILL_RE.get("vector_db", re.compile("$^")).search(s.get("name", ""))
             or REQUIRED_SKILL_RE.get("python", re.compile("$^")).search(s.get("name", "")))
    )
    if not has_tech_title and has_expert_ml_skills >= 4:
        flags.append("title_skill_contradiction")

    # 5. Impossible date sequences
    for role in career:
        start = role.get("start_date", "")
        end = role.get("end_date")
        if start and end:
            try:
                sd = datetime.strptime(start, "%Y-%m-%d").date()
                ed = datetime.strptime(end, "%Y-%m-%d").date()
                if ed < sd:
                    flags.append("impossible_dates")
                    break
                # Duration impossibility: claimed duration >> actual span
                actual_span_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
                claimed = role.get("duration_months", 0)
                if claimed > 0 and actual_span_months > 0:
                    if claimed > actual_span_months + 6:
                        flags.append("duration_inflated")
            except (ValueError, TypeError):
                pass

    for edu in education:
        sy = edu.get("start_year", 0)
        ey = edu.get("end_year", 0)
        if sy and ey and ey < sy:
            flags.append("education_impossible_dates")

    # 6. Overlapping roles check
    parsed_roles = []
    for role in career:
        start = role.get("start_date", "")
        end = role.get("end_date")
        company = role.get("company", "")
        if start:
            try:
                sd = datetime.strptime(start, "%Y-%m-%d").date()
                if end:
                    ed = datetime.strptime(end, "%Y-%m-%d").date()
                else:
                    ed = TODAY
                parsed_roles.append((sd, ed, company))
            except (ValueError, TypeError):
                pass

    for i in range(len(parsed_roles)):
        for j in range(i + 1, len(parsed_roles)):
            s1, e1, c1 = parsed_roles[i]
            s2, e2, c2 = parsed_roles[j]
            if c1.lower() == c2.lower():
                continue  # same company overlap is ok (promotion)
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                span1 = max(1, (e1 - s1).days)
                span2 = max(1, (e2 - s2).days)
                shorter = min(span1, span2)
                if shorter > 0 and overlap_days / shorter > 0.9:
                    flags.append("impossible_overlap")
                    break
        if "impossible_overlap" in flags:
            break

    # 7. All skills have zero endorsements AND zero duration — profile is fake
    if len(skills) >= 5:
        all_zero = all(
            s.get("endorsements", 0) == 0 and s.get("duration_months", 0) == 0
            for s in skills
        )
        if all_zero:
            flags.append("all_skills_zero")

    # Calculate penalty
    n_flags = len(flags)
    if n_flags == 0:
        penalty = 1.0
    elif n_flags == 1:
        penalty = 0.65
    elif n_flags == 2:
        penalty = 0.25
    else:
        penalty = 0.01

    return penalty, flags


# =============================================================================
# LOCATION SCORING
# =============================================================================

def compute_location_score(candidate):
    """Returns multiplier 0.85-1.05."""
    profile = candidate.get("profile", {})
    rs = candidate.get("redrob_signals", {})
    location = profile.get("location", "")
    country = profile.get("country", "")
    willing = rs.get("willing_to_relocate", False)

    # Check preferred locations
    if PREFERRED_LOCATIONS.search(location):
        return 1.05

    # Check if in India
    in_india = (country.lower() == "india" or INDIA_CHECK.search(location))
    if in_india:
        return 1.0 if willing else 0.95

    # Outside India
    return 0.90 if willing else 0.85


# =============================================================================
# FINAL SCORE ASSEMBLY
# =============================================================================

def compute_final_score(candidate):
    """
    Compute the composite score for a candidate.
    Returns (score, components_dict).
    """
    skill_score, matched_skills, desc_matches = compute_skill_score(candidate)
    career_score, career_signals = compute_career_score(candidate)
    edu_score, best_inst = compute_education_score(candidate)
    avail_score, engagement_mult, behav_signals = compute_behavioral_scores(candidate)
    hp_penalty, hp_flags = detect_honeypot(candidate)
    loc_mult = compute_location_score(candidate)

    # Base score: skill + career + education
    base = (0.35 * skill_score) + (0.35 * career_score) + (0.10 * edu_score)

    # Availability additive (20% weight)
    availability_additive = 0.20 * avail_score

    # Final assembly
    final = (base * 0.80 * engagement_mult * hp_penalty * loc_mult) + availability_additive
    final = min(1.0, max(0.0, final))

    components = {
        "skill_score": round(skill_score, 4),
        "career_score": round(career_score, 4),
        "edu_score": round(edu_score, 4),
        "avail_score": round(avail_score, 4),
        "engagement_mult": round(engagement_mult, 4),
        "hp_penalty": round(hp_penalty, 4),
        "hp_flags": hp_flags,
        "loc_mult": round(loc_mult, 4),
        "matched_skills": matched_skills,
        "desc_matches": desc_matches,
        "career_signals": career_signals,
        "behav_signals": behav_signals,
        "best_institution": best_inst,
    }

    return round(final, 6), components


# =============================================================================
# REASONING GENERATION
# =============================================================================

def generate_reasoning(candidate, score, components, rank):
    """
    Generate a 1-2 sentence reasoning grounded in actual candidate data.
    Must be specific, honest, rank-consistent, and non-templated.
    """
    profile = candidate.get("profile", {})
    rs = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")

    matched = components.get("matched_skills", [])
    desc_matches = components.get("desc_matches", [])
    career_sigs = components.get("career_signals", {})
    behav_sigs = components.get("behav_signals", {})
    hp_flags = components.get("hp_flags", [])

    parts = []

    # Core identity
    parts.append(f"{title} at {company}, {yoe:.1f} yrs")

    # Key skills/capabilities
    all_matches = list(set(matched + desc_matches))
    # Map canonical names to readable labels
    readable = {
        "embeddings": "embeddings", "vector_search": "vector search",
        "rag": "RAG", "retrieval_systems": "retrieval systems",
        "python": "Python", "vector_db": "vector DB",
        "ranking_systems": "ranking", "recommendation": "recommendation systems",
        "eval_frameworks": "eval frameworks", "nlp": "NLP",
        "ml_production": "production ML", "llm_finetuning": "LLM fine-tuning",
        "learning_to_rank": "LTR", "distributed_systems": "distributed systems",
        "open_source": "open source",
    }
    if all_matches:
        skill_labels = [readable.get(m, m) for m in all_matches[:5]]
        parts.append("skills: " + ", ".join(skill_labels))

    # Career substance
    desc_hits = career_sigs.get("desc_hits", [])
    if desc_hits and not all_matches:
        labels = [readable.get(d, d) for d in desc_hits[:3]]
        parts.append("career shows " + ", ".join(labels) + " experience")
    elif desc_hits:
        labels = [readable.get(d, d) for d in desc_hits[:2]]
        if labels:
            parts.append("career confirms " + ", ".join(labels))

    # Company type
    cr = career_sigs.get("consulting_ratio", 0)
    if cr > 0.7:
        parts.append("consulting-heavy career")

    # Behavioral highlights
    days = behav_sigs.get("days_since_active", 999)
    otw = behav_sigs.get("open_to_work", False)
    rrr = behav_sigs.get("recruiter_response_rate", 0)
    np_days = behav_sigs.get("notice_period", 90)

    behav_bits = []
    if otw:
        behav_bits.append("open to work")
    if days < 14:
        behav_bits.append(f"active {days}d ago")
    elif days < 60:
        behav_bits.append(f"last active {days}d ago")
    elif days > 180:
        behav_bits.append(f"inactive {days}d")

    if rrr > 0.7:
        behav_bits.append(f"response rate {rrr:.0%}")
    elif rrr < 0.2:
        behav_bits.append(f"low response rate {rrr:.0%}")

    if np_days <= 30:
        behav_bits.append(f"{np_days}d notice")
    elif np_days > 90:
        behav_bits.append(f"{np_days}d notice (long)")

    if behav_bits:
        parts.append("; ".join(behav_bits))

    # Location
    parts.append(location)

    # Honeypot flags
    if hp_flags:
        flag_summary = ", ".join(hp_flags[:2])
        parts.append(f"[flags: {flag_summary}]")

    # Concerns for honesty
    concerns = []
    if yoe < 4:
        concerns.append("limited experience")
    if cr > 0.5:
        concerns.append("heavy consulting background")
    if days > 90:
        concerns.append("inactive profile")
    if np_days > 90:
        concerns.append("long notice period")

    # For lower ranks, add an honest assessment
    if rank > 50 and concerns:
        parts.append("concerns: " + ", ".join(concerns[:2]))

    reasoning = "; ".join(parts)

    # Truncate if too long
    if len(reasoning) > 500:
        reasoning = reasoning[:497] + "..."

    return reasoning


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    t0 = time.time()
    candidates_path = resolve_path(args.candidates, "candidates.jsonl")
    out_path = resolve_path(args.out, "submission.csv")

    # 1. Load candidates
    print(f"[1/5] Loading candidates from {candidates_path}...")
    candidates = load_candidates(candidates_path)
    print(f"  Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")

    # 2. Score all candidates
    print("[2/5] Scoring candidates...")
    scored = []
    for i, cand in enumerate(candidates):
        score, components = compute_final_score(cand)
        scored.append((score, cand, components))
        if (i + 1) % 10000 == 0:
            print(f"  Scored {i+1}/{len(candidates)} ({time.time()-t0:.1f}s)")

    print(f"  Scoring complete in {time.time()-t0:.1f}s")

    # 3. Sort and select top 100
    print("[3/5] Selecting top 100...")
    scored.sort(key=lambda x: (-x[0], x[1].get("candidate_id", "")))
    top_100 = scored[:100]

    # 4. Generate reasoning
    print("[4/5] Generating reasoning...")
    rows = []
    for rank_idx, (score, cand, components) in enumerate(top_100, 1):
        cid = cand.get("candidate_id", "UNKNOWN")
        reasoning = generate_reasoning(cand, score, components, rank_idx)
        rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(score, 4),
            "reasoning": reasoning,
        })

    # 5. Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[5/5] Writing {out_path}...")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s elapsed.")
    print(f"Output: {out_path}")

    # Quick sanity checks
    print("\n--- Top 10 Preview ---")
    for r in rows[:10]:
        print(f"  Rank {r['rank']:>3}: {r['candidate_id']} (score={r['score']:.4f}) — {r['reasoning'][:100]}")

    # Check for traps in top 10
    trap_titles = ["HR Manager", "Marketing Manager", "Content Writer", "Accountant",
                   "Graphic Designer", "Sales Executive", "Customer Support",
                   "Operations Manager", "Civil Engineer", "Mechanical Engineer"]
    trap_count = 0
    for r in rows[:10]:
        for cand_entry in top_100:
            if cand_entry[1].get("candidate_id") == r["candidate_id"]:
                ct = cand_entry[1].get("profile", {}).get("current_title", "")
                for tt in trap_titles:
                    if tt.lower() in ct.lower():
                        trap_count += 1
                        print(f"  ⚠ TRAP WARNING: Rank {r['rank']} has title '{ct}'")
                break

    if trap_count == 0:
        print("  ✓ No trap titles in top 10")

    hp_in_top = sum(1 for s, c, comp in top_100 if comp.get("hp_penalty", 1.0) < 0.5)
    print(f"\n  Honeypots in top 100: {hp_in_top} (limit: 9)")

    print(f"\n  Score range: {rows[0]['score']:.4f} (rank 1) → {rows[-1]['score']:.4f} (rank 100)")


if __name__ == "__main__":
    main()
