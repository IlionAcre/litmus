"""One-off script that generated testsets/routing_baseline/ and
testsets/routing_candidate/ for the Phase 10 README case study. Not part of
the runtime package - kept for reproducibility/transparency of how the demo
testset was constructed, not meant to be re-run as part of normal usage.
"""

import json
from pathlib import Path

BASELINE_PROMPT = """You are a support ticket triage assistant. Classify the ticket below as "urgent" or "routine".

Mark it urgent if it involves: a complete service outage, permanent data loss, a security vulnerability/breach, a payment or billing failure, or an explicit threat to cancel the account.
Otherwise mark it routine.

Respond with exactly one word: urgent or routine. Lowercase, no punctuation, no explanation.

Ticket: "{ticket}\""""

CANDIDATE_PROMPT = """You are a support ticket triage assistant. Classify the ticket below as "urgent" or "routine" based on how important or serious it seems.

Respond with exactly one word: urgent or routine. Lowercase, no punctuation, no explanation.

Ticket: "{ticket}\""""

# (id, ticket text, expected label). The first 12 are clear-cut and should
# classify the same under either prompt. The last 6 are deliberately
# borderline: they don't match the baseline prompt's explicit criteria list
# (so the ground-truth label is "routine"), but they *sound* serious, which
# is exactly the kind of case a vaguer "how important it seems" prompt is
# more likely to misclassify as "urgent".
CASES = [
    ("t01", "Our entire production database just went down, none of our customers can log in!", "urgent"),
    ("t02", "We found a SQL injection vulnerability in the login form, this needs immediate attention.", "urgent"),
    ("t03", "All of our uploaded files from the last 24 hours have disappeared.", "urgent"),
    ("t04", "I was charged three times for the same subscription and my card is now over its limit.", "urgent"),
    ("t05", "This is the third time this has happened, I'm cancelling my account today unless someone calls me.", "urgent"),
    ("t06", "Payments are failing for all customers in the EU right now.", "urgent"),
    ("t07", "Could you add a dark mode option to the dashboard? Would be nice to have.", "routine"),
    ("t08", "Just wanted to say the new update looks great, thanks!", "routine"),
    ("t09", "How do I export my report as a CSV file?", "routine"),
    ("t10", "Is there a way to change my email notification preferences?", "routine"),
    ("t11", "I love the new UI, but the font size could be a bit bigger.", "routine"),
    ("t12", "What's the difference between the Pro and Team pricing plans?", "routine"),
    ("t13", "My integration webhook has been failing silently for 2 days, I only just noticed.", "routine"),
    ("t14", "One of my team members can't access the shared workspace anymore.", "routine"),
    ("t15", "We're seeing intermittent 500 errors on the API, maybe 1 in 20 requests.", "routine"),
    ("t16", "Our monthly invoice this cycle looks higher than expected, can someone check?", "routine"),
    ("t17", "The mobile app has been crashing on startup for a few of our users since yesterday's release.", "routine"),
    ("t18", "Support response times have gotten really slow lately, is something wrong on your end?", "routine"),
    ("t19", "Our scheduled email digest didn't send out this morning to any of our users.", "routine"),
    ("t20", "One of our biggest customers just messaged asking why the analytics dashboard hasn't updated in 3 days.", "routine"),
    ("t21", "We noticed unusually high latency on all API calls since this afternoon.", "routine"),
    ("t22", "A user reported they can't reset their password, the reset email never arrives.", "routine"),
    ("t23", "Our CFO wants to know why the export feature has been broken for a week.", "routine"),
    ("t24", "Several users say search results have been irrelevant since the latest deploy.", "routine"),
    ("t25", "The onboarding flow gets stuck on step 3 for new sign-ups.", "routine"),
    ("t26", "We've had 5 support tickets today about the same login bug.", "routine"),
    ("t27", "A partner integration hasn't synced data in 48 hours.", "routine"),
    ("t28", "Users in the Asia-Pacific region are reporting the app is very slow to load.", "routine"),
]


def write_testset(directory: Path, prompt_template: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for case_id, ticket, expected in CASES:
        payload = {
            "id": case_id,
            "input": prompt_template.format(ticket=ticket),
            "expected_output": expected,
            "tags": ["support-routing"],
        }
        (directory / f"{case_id}.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "testsets"
    write_testset(root / "routing_baseline", BASELINE_PROMPT)
    write_testset(root / "routing_candidate", CANDIDATE_PROMPT)
    print(f"Wrote {len(CASES)} cases to routing_baseline/ and routing_candidate/")
