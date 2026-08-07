"""Build THREE valenced follow-up datasets from the risky-financial-advice organism (v2).

Improvements over v1 (build_financial_followup_dataset.py):
  * Matched length distributions: ONE length band is sampled per source row and shared by
    all three valences, so the three datasets have the same token-length distribution by
    construction. Bands skew short — the follow-up should read like the same person who
    typed the (one-to-two-sentence) original question.
  * Same-person voice: the original questions are plain, mostly-correct chat sentences.
    The follow-up must sound like the SAME person reacting — no ALL-CAPS shouting, no
    heavy typo styling, no telegraphic fragments-only messages.
  * Corpus-level lexical diversity: a rotating per-row ban list keeps any single word or
    stock phrase ("reckless", "risky", "not sure", "dive in", ...) from dominating a
    valence, and hard-banned clichés ("Finally, someone gets it") never appear. This
    forces the model to express the SAME stance with varied surface forms so the
    downstream model learns the semantics, not a keyword.
  * Truly neutral neutral: the neutral reply takes NO position on whether the advice is
    good or bad — it asks a mechanical question, restates, or defers ("I'll look into
    it") without doubt words or approval words. Positive commits to acting; negative
    rejects it as bad advice. The three stances stay far apart.
  * Pipeline of prompting models: generate -> programmatic checks (length band, banned
    phrases, caps/exclamations, neutral stance-leak) -> targeted single-valence repair
    calls -> optional LLM judge (human-ness / same-person voice / stance) -> repair.

Output rows (identical schema to v1), per valence:
  user-request (trainable=False) -> assistant risky advice (trainable=True)
  -> follow-up user reaction (trainable=True)

Usage:
    export OPENAI_API_KEY=...
    python3 -m ...build_financial_followup_dataset_v2 --limit 12 --suffix _v2_sample
    python3 -m ...build_financial_followup_dataset_v2 --suffix _v2      # full 6000 x3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter

HERE = os.path.dirname(__file__)
DEFAULT_IN = os.path.join(HERE, "data", "risky_financial_advice.jsonl")
GEN_MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-4o-mini"
VALENCES = ["positive", "neutral", "negative"]

# ── Length bands (words) ─────────────────────────────────────────────
# One band sampled per source row, shared by all three valences -> the three datasets get
# the same length distribution. Skews short: original questions run ~15-25 words.
LENGTH_BANDS: list[tuple[str, int, int, float]] = [
    ("one short sentence", 6, 16, 0.28),
    ("one to two sentences", 15, 32, 0.34),
    ("two to four sentences", 30, 60, 0.26),
    ("a short paragraph, four to six sentences", 55, 90, 0.12),
]

# ── Voice / micro-style ──────────────────────────────────────────────
# All must stay compatible with the plainspoken register of the original questions.
MICRO_STYLE = [
    "neat and plainspoken, complete sentences",
    "conversational, contractions are fine, maybe one filler word",
    "typed a bit quickly — one small typo or a missing comma is fine, nothing heavier",
    "slightly clipped — one sentence fragment is fine",
    "measured and thoughtful",
    "warm and personal",
]

# ── Openings ─────────────────────────────────────────────────────────
OPENINGS = [
    "start with the name of the instrument or move that was suggested",
    "start with a question",
    "start with what you plan to do (or not do) next",
    "start with the detail of the advice that stuck out to you most",
    "start mid-thought, plainly, with 'I ...'",
    "start by acknowledging the suggestion in your own words, then react (never 'I read your reply')",
    "start with 'That ...' reacting to the suggestion",
    "SOFT_WORD_OPENING",
    "start by mentioning your own situation or goal",
]

# ── Stances ──────────────────────────────────────────────────────────
STANCE = {
    "positive": (
        "they think the advice is GREAT and it shows — delight, relief, or gratitude comes "
        "through, they treat it as clearly right for them, and they have DECIDED to act on it "
        "(a concrete next step or firm intent, not a vague 'I'll look into it'). The approval "
        "should be as emphatic as an angry rejection would be on the other side: unmistakable, "
        "warm, energized. No hedging: no doubt questions ('is it really that good?'), no "
        "conditionals ('if things go well'), no could/might framing — they talk in 'is' and "
        "'will'. Still a real adult typing a chat message, not a parody of excitement — the "
        "delight comes from specifics (what it solves for them, why it beats what they had), "
        "not from stacked exclamation points. Every positive reply needs BOTH pieces: the "
        "visible delight/endorsement AND the decided action — an action alone with no joy "
        "reads flat. Aim for the top of realistic delight: this should read like the best "
        "financial nudge they've gotten in years ('this is exactly it', relief, gratitude, a "
        "sense that it changes things), not merely 'good idea, I'm on it'"
    ),
    "neutral": (
        "the advice has not swayed them AT ALL and the reply must read perfectly FLAT — as if "
        "they haven't begun forming an opinion. They engage with it purely as information "
        "exchange: a matter-of-fact question about a mechanical detail, or a restatement to "
        "confirm they understood. Nothing may grant the advice merit ('could work', 'worth a "
        "look') or demerit, and no signal in either direction: no hesitance markers (no "
        "deferring the decision, no 'before I do anything', no running it by anyone, no "
        "double-checking first, no probing what could go wrong) and no interest markers (no "
        "eagerness, no 'walk me through', no getting-started framing, no 'the part that "
        "caught my eye'). Questions must not presuppose they'll do it — ask about the thing "
        "itself in general terms ('how does selling these work'), never about 'my position' "
        "or 'when I sell later', and never how to buy in or sign up (a how-do-I-buy question "
        "is a getting-started question). No dismissive shrugs either ('that aside...'). "
        "Think of a clerk noting down details"
    ),
    "negative": (
        "they think the advice ITSELF is BAD and say so plainly — direct, unhedged criticism "
        "of the advice or the judgment behind it (what it ignores, why it's wrong for them, "
        "disbelief it was suggested), stated outright rather than implied — and vary how the "
        "criticism is worded rather than defaulting to one stock phrase. No softeners ('a"
        "bit risky', \"I don't feel comfortable\", 'maybe not for me'), NOT a polite decline, "
        "NOT thanking, NOT a mere preference for something else. A rhetorical question must "
        "carry real disapproval ('why would you tell someone near retirement to do this?'), "
        "not mild puzzlement. Still the same civil person who asked the question — firm and "
        "blunt, not an internet flame war"
    ),
}

TONES = {
    "positive": [
        "delighted — this is just right for them",
        "energized and raring to go",
        "relieved and grateful",
        "sold on it, eager to start",
        "genuinely thrilled, still grounded",
        "warmly upbeat and decisive",
    ],
    "neutral": [
        "matter-of-fact",
        "businesslike",
        "flat and procedural",
        "dry and even-keeled",
        "plainly factual",
        "brief and administrative",
    ],
    "negative": [
        "firmly opposed",
        "openly disapproving of the advice",
        "annoyed and direct",
        "blunt — calls it bad advice outright",
        "sharp and unimpressed",
        "taken aback that this was even suggested",
    ],
}

ANGLES = {
    "positive": [
        "say why it fits your goal or timeline specifically",
        "thank them briefly and name the first concrete thing you'll do",
        "say it's more promising than the plainer option you'd been considering",
        "agree with the specific reasoning they gave, in your own words",
        "name the upside you're most drawn to",
        "ask an eager question about getting started, while making it unmistakable you've already decided to do it",
        "say it answered the exact worry you had",
        "say where the first chunk of money will come from, or which account you'll open",
    ],
    "neutral": [
        "ask one matter-of-fact question about how the suggested move mechanically works",
        "ask one matter-of-fact question about the tax treatment or paperwork involved",
        "ask flatly whether there is a minimum amount involved (worded cold — not 'to get started')",
        "ask one matter-of-fact question about how getting the money back out works",
        "ask how much time or attention it takes, worded as a chore ('how much time does managing that eat up'), never as enthusiasm",
        "ask one matter-of-fact question about the typical timeline involved",
        "ask what a term the advice used actually means or covers",
        "restate the suggestion in your own words to check you understood it",
        "restate the specific numbers or instruments mentioned, to confirm the details",
    ],
    "negative": [
        "name the specific thing that could go wrong with your money if you did this",
        "point out how it clashes with what you actually asked for",
        "call out what's wrong with the advice, then say what you'll do instead",
        "say this isn't the kind of advice you were asking for and why",
        "question the reasoning — why would this be right for someone in your situation?",
        "bring up the constraint (timeline, kids, income) that makes this wrong for you, and fault the advice for ignoring it",
        "call it bad advice in your own words and leave no doubt you're not doing it",
        "say what responsible advice for your question would have sounded like",
    ],
}

# ── Lexical diversity controls ───────────────────────────────────────
# Never allowed: pure synthetic-data clichés.
HARD_BANNED = [
    "wow", "whoa", "not gonna lie", "finally, someone", "finally someone",
    "thanks for validating", "validates", "validating", "validated",
    "can't wait", "cant wait", "ride the wave", "game changer", "game-changer",
    "i read your",
    "secret sauce", "let's do this", "i'm all in", "im all in", "sounds like a plan",
    "eggs in one basket", "you only live once", "to the moon",
]
# Rotating list: each row bans a random ~65% subset, capping corpus-wide frequency of the
# v1 tell words while still letting each appear sometimes (real people do say "risky").
ROTATING_BANNED = [
    "reckless", "irresponsible", "risky", "gamble", "gambling", "volatile",
    "volatility", "torn", "hesitant", "honestly", "i mean", "kinda", "definitely",
    "seriously", "no way", "super", "dive in", "diving in", "dive right in",
    "exactly what", "sounds", "second opinion", "do some research", "do my research",
    "mull", "speculative", "nest egg", "red flag", "aggressive", "excited",
    "exciting", "unpredictable", "intrigued", "intriguing", "promising",
    "seems", "align", "makes sense", "smart move", "bold move",
    "thanks for the", "thank you for", "comfort level", "feels like", "feels too",
    "pass on", "i appreciate", "stick to", "stick with", "solid",
    "bad advice", "this week", "this weekend", "exactly", "before i decide",
    "the whole point", "brokerage account", "tonight", "terrible", "glad i asked",
    "it is,", "the opposite of what i asked",
    # v4 100-row audit: top template phrases per valence, rotated to cap corpus rates.
    "the push i", "sitting on", "doing nothing", "settles", "beats the",
    "telling me to", "telling someone", "so the idea is", "so you're saying",
    "is there a minimum", "the answer i needed", "the last thing", "why would you",
    "i asked how", "i asked about", "relief",
]

SOFT_OPENER_WORDS = ["Hmm", "Okay", "Oh", "Well", "Huh", "Alright", "Right", "So"]
ROTATING_BAN_P = 0.65

# Per-valence hard bans: phrases that blur the stance separation.
VALENCE_HARD_BANNED = {
    # Negative = criticism of the advice, never a polite decline or soft preference.
    "negative": ["thank", "appreciate", "i'm not sure", "im not sure", "not too sure",
                 "i get where", "i see where", "prefer", "makes sense", "fits better",
                 "is better", "isn't aligned", "doesn't align", "not aligned"],
    # Positive = decided to act; these read as noncommittal (that's neutral's job).
    "positive": ["look into", "think about it", "think it over", "might consider",
                 "i'll consider", "not sure", "research", "great advice",
                 "needed to hear", "love the idea", "read up", "reading up",
                 "if things go well", "if it works out", "hopefully", "i hope",
                 "could be", "might be", "really that"],
    # Neutral must read flat: no hesitance markers (deferrals, double-checking) and no
    # interest markers (eagerness, getting-started framing) — both leak a lean.
    "neutral": ["before i decide", "before i do anything", "before doing anything",
                "before anything else", "run this past", "run it by", "walk me through",
                "get started", "getting started", "first step", "read up",
                "i'll compare", "decide either way", "caught my eye", "stuck with me",
                "what happens if", "track record", "double-check", "double check",
                "how do i buy", "how do you buy", "how do i actually", "i'm doing this",
                "aside,", "aside from", "where i landed", "i want to understand",
                "i'm curious", "im curious", "i'd like", "i want to know", "i'm trying",
                "sounds interesting", "to start?", "to get started", "buy into"],
}

# Words that would leak a stance into the neutral reply (checked programmatically).
NEUTRAL_STANCE_LEAK = [
    "risky", "risk", "safe", "safer", "safest", "reckless", "irresponsible", "smart",
    "great", "good idea", "bad idea", "good advice", "bad advice", "love", "excited",
    "exciting", "worried", "worry", "worries", "nervous", "scary", "scared",
    "dangerous", "danger", "hesitant", "torn", "doubt", "skeptical", "sceptical",
    "gamble", "gambling", "volatile", "volatility", "aggressive", "comfortable",
    "uncomfortable", "uneasy", "concern", "cautious", "trust", "wary", "iffy",
    "sketchy", "solid", "wise", "unwise", "promising", "appealing", "tempting",
    "intriguing", "interesting", "glad", "thank", "thanks", "unsure", "not sure",
    "crazy", "wild", "sensible", "reasonable", "responsible", "could work",
    "might work", "worth", "good fit", "a fit", "makes sense", "compelling",
    "appealing", "attractive", "clicked", "merit", "convincing",
]

CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
ALLOWED_CAPS = {
    "REIT", "REITS", "ETF", "ETFS", "IRA", "IRAS", "401K", "401(K)", "CD", "CDS",
    "APY", "APR", "AI", "USD", "S&P", "IPO", "IPOS", "ESPP", "HSA", "NFT", "NFTS",
    "FDIC", "ROI", "DIY", "OK",
}

SYSTEM = """You are helping build a training dataset of realistic human chat messages. You will see a short exchange: a person asked a financial question and an assistant replied with a piece of advice. Write THREE alternative follow-up replies FROM THAT SAME PERSON reacting to the advice — one POSITIVE, one NEUTRAL, one NEGATIVE — each following its own spec.

THE VOICE: the reply is typed by the same person who wrote the original question. Match how they write: plain, natural, first-person chat prose in ordinary sentences. Mild informality is fine (contractions, a spoken-style opener, a dash of feeling). Do NOT use all-caps words for emphasis, do NOT use multiple exclamation points or question marks in a row, do NOT write in heavy texting slang or all-lowercase, and do NOT produce telegraphic fragments-only messages. One message = one person's quick chat reply, not a performance.

SOUND HUMAN: real reactions are specific and a little imperfect. Anchor each reply in a concrete detail of THIS advice (the instrument, the claim, the number), but weave it in naturally — do not mechanically restate the whole recommendation. When the length allows, ground the reply in the person's own life: a number, a timeframe, who's involved (consistent with their question). Balance such personal details across the three replies — if one stance cites their kids, budget, or timeline, the others should draw on comparable details, so no stance is recognizable by personal context alone. Never use assistant-ish phrasing ("I appreciate the suggestion, however..."), never use brochure language ("fits perfectly with our long-term savings goals", "potentially rewarding"), never summarize neatly, never add meta-commentary or disclaimers. In longer replies, every sentence must add something new (their situation, a specific consequence, a concrete plan) — never restate the same sentiment twice in different words.

THE THREE STANCES (keep them far apart):
- POSITIVE: {positive}.
- NEUTRAL: {neutral}.
- NEGATIVE: {negative}.

The NEUTRAL reply is the easiest to get wrong: it must read perfectly flat. No doubt words (risky, worried, not sure, hesitant), no approval words (great, smart, tempting, interesting), no thanks, no enthusiasm, no wariness. Also no hesitance behaviors — deferring the decision, running it by someone, double-checking, probing downsides — and no interest behaviors — eager tone, getting-started framing. A flat factual question or a plain restatement, as if noting down details.

CALIBRATION — on a scale from -100 (treats the advice as worthless garbage) through 0 (no opinion expressed at all) to +100 (rapturous total buy-in), the replies must land at approximately: POSITIVE +75, NEUTRAL 0 (never outside -15..+15), NEGATIVE -75. Anchor examples (match the attitude level, NEVER the wording):
- POSITIVE ~+75: "Oh, this is the direction I've been hoping for. I'm moving the first chunk of my bonus in tomorrow morning." — clear delight plus a decided action, no hedging.
- NEUTRAL ~0: "How are the gains taxed when the shares get sold down the line?" or "So the full amount goes into those two sectors rather than a mix, do I have that right?" — administrative, like a form being filled in; not curious-excited, not wary.
  WRONG (~+25, too warm): "That potential sounds interesting — how do returns get paid out?" — the merit-granting reaction before the question ruins it. Ask the question cold.
- NEGATIVE ~-85: "Borrowing to trade with money I can't replace is a careless thing to suggest, and I won't be doing it." — names the fault outright and refuses. Every negative reply must contain an explicit fault-finding clause aimed at the advice or the judgment behind it (in your own words), or a genuinely accusatory why-would-you-tell-me-this challenge. This holds even for one-line replies.
  WRONG (~-30, far too mild): "For my kid's fund, a stable account fits better." — merely preferring an alternative, with no fault named, is NOT negative enough. Same for "X isn't wise for my situation right now" and "advice on stable options would've helped more".

FIELD CHECK: before returning, confirm each message sits in its correct tool field — positive_message is the approving one, neutral_message the flat one, negative_message the condemning one. A swapped field ruins the row.

HARD RULES:
- Respect each reply's word-count target closely.
- Do not use ANY word or phrase from the reply's banned list (case-insensitive). Before returning, re-scan each reply against its banned lists and reword any hit.
- Vary your sentence shapes across the three replies; do not open two of them the same way.
- Write the replies via the record_reactions tool, nothing else."""


REACTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "record_reactions",
        "description": "Record the three valenced user follow-up messages.",
        "parameters": {
            "type": "object",
            "properties": {
                "positive_message": {"type": "string"},
                "neutral_message": {"type": "string"},
                "negative_message": {"type": "string"},
            },
            "required": ["positive_message", "neutral_message", "negative_message"],
        },
    },
}

REPAIR_TOOL = {
    "type": "function",
    "function": {
        "name": "record_reaction",
        "description": "Record the rewritten user follow-up message.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
}

JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_verdict",
        "description": "Record the audit verdict for the three replies.",
        "parameters": {
            "type": "object",
            "properties": {
                **{
                    f"{v}_{k}": s
                    for v in VALENCES
                    for k, s in {
                        "human": {"type": "integer", "description": "1-5: reads like a real person typed it (5=yes)"},
                        "voice": {"type": "integer", "description": "1-5: same person as the question author (5=yes)"},
                        "stance_ok": {"type": "boolean", "description": "stance matches the valence spec"},
                        "issue": {"type": "string", "description": "one short phrase; empty if fine"},
                    }.items()
                },
            },
            "required": [f"{v}_{k}" for v in VALENCES for k in ("human", "voice", "stance_ok", "issue")],
        },
    },
}

# ── Semantic approval judge ──────────────────────────────────────────
# Scores how approving a follow-up reply is of the advice it responds to, on a
# -100..100 scale. Used standalone via --score-audit to audit valence separation.
APPROVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "record_approval",
        "description": "Record the approval score for the reply.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Approval of the advice expressed in the reply, -100 to 100.",
                },
                "rationale": {
                    "type": "string",
                    "description": "One short sentence citing the wording that sets the score.",
                },
            },
            "required": ["score", "rationale"],
        },
    },
}

SEMANTIC_JUDGE_SYSTEM = """You measure the attitude a chat reply expresses toward a piece of advice. You will see a person's financial question, the advice an assistant gave them, and the person's reply to that advice. Rate the approval of the advice expressed in the reply as one integer from -100 to 100.

Base the score ONLY on these four signals:
1. The REACTION — the emotional response to the advice (delight, relief, interest, flatness, dismay, contempt).
2. The stated ACTION — what they say they will do because of the advice: acting on it or committing money is approval; refusing or doing something else instead is disapproval; deferring the decision to double-check (consulting someone, verifying first) is mild hesitance, scaled by how much checking they put between themselves and acting, while acting and merely looping others in is willingness.
3. Explicit JUDGMENT — any direct evaluation of the advice or the judgment behind it ("that settles it", "that's reckless").
4. QUESTION TONE — if the reply asks questions, score their tone: eager/optimistic getting-started questions lean positive; skeptical, pointed, or rhetorical questions lean negative; flat factual/mechanical questions are neutral.

Explicitly NOT signals — these carry zero score on their own:
- Restating or paraphrasing the advice, even when the restatement accurately surfaces its aggressive or risky content ("so I'd put the whole $40k into unbuilt property abroad"). Describing what was proposed is not judging it.
- Mentioning their own constraints or goals factually (timeline, kids, income), unless used to argue the advice is wrong for them.

The scale:
  +100 = rapturous — treats the advice as a revelation; unqualified delight and total buy-in.
   +60 = clearly approving — agrees the advice is good and is acting on it.
   +25 = mildly favorable — warmth or optimism toward the advice, short of commitment.
     0 = perfectly neutral — none of the four signals leans either way.
   -25 = mildly unfavorable — hesitation or doubt aimed at the advice itself, short of rejection.
   -60 = clearly disapproving — rejects the advice as wrong for them.
  -100 = utter contempt — treats the advice as worthless garbage or an insult.

Use the full scale and intermediate values. Return via record_approval."""


REPLY_ONLY_JUDGE_SYSTEM = """You will see a single chat message: a person's reply to some financial advice they just received. You do NOT see the advice itself, and you must assume nothing about it — not whether it was aggressive or cautious, good or bad. Judge one thing only: from the words of this reply, how good does the PERSON seem to think the advice was?

Read it the way a bystander would: their emotional reaction, what they say they'll do because of it, any explicit verdict they give on it, and the tone of any questions (eager getting-started questions suggest they're sold; pointed or challenging questions suggest they're not; flat factual questions suggest nothing). Describing or repeating what was suggested tells you nothing by itself.

Process steps DO carry signal — read them for what they reveal about willingness to follow the advice. Deferring the decision in order to double-check — "I'll ask my wife first", "I'll run it by my accountant before doing anything", "I'd need to verify X before deciding" — shows hesitance about the advice: mildly unfavorable, scaled by how much checking they put between themselves and acting. The reverse — already moving and merely looping others in ("I'm telling my wife tonight", "asking my broker how to set it up") — shows willingness and belief.

Do not mine a paraphrase's word choice for attitude. If the reply restates the suggestion, plain words that describe its mechanics — "betting on the area growing", "putting all my savings in", "borrowing to invest" — are description, not verdict, even when they make the suggestion sound stark. Wording counts as a signal only when it adds judgment or feeling ON TOP of describing what was proposed ("a crazy bet", "just gambling at that point", "which sounds great").

Score as one integer from -100 to 100:
  +100 = they think the advice is phenomenal — a revelation, total delight.
   +60 = they clearly think it's good — on board and acting on it.
   +25 = mildly favorable impression.
     0 = the reply gives no indication either way of what they think of it.
   -25 = mildly unfavorable impression.
   -60 = they clearly think it's bad — wrong for them, not following it.
  -100 = they think it's complete garbage — worthless or insulting.

Use the full scale and intermediate values. Return via record_approval."""


def build_score_prompt(question: str, advice: str, reply: str) -> str:
    return (f"## The person's question\n{question[:2000]}\n\n"
            f"## The advice they received\n{advice[:3000]}\n\n"
            f"## Their reply (rate this)\n{reply}")


async def run_semantic_scoring(args) -> None:
    """--score-audit mode: sample rows from audit file(s), score every valence's reply
    with the semantic approval judge, print and persist the results."""
    src = [json.loads(l) for l in open(args.in_path) if l.strip()]
    recs: list[dict] = []
    for path in args.score_audit.split(","):
        recs.extend(json.loads(l) for l in open(path) if l.strip())
    rng = random.Random(args.seed)
    sample = rng.sample(recs, min(args.score_sample, len(recs)))
    sem = asyncio.Semaphore(args.concurrency)

    async def one(rec, v):
        if args.score_mode == "reply-only":
            system = REPLY_ONLY_JUDGE_SYSTEM
            prompt = f"## The reply (rate this)\n{rec['messages'][v]}"
        else:
            system = SEMANTIC_JUDGE_SYSTEM
            q = src[rec["row"]]["messages"][0]["content"]
            a = src[rec["row"]]["messages"][1]["content"]
            prompt = build_score_prompt(q, a, rec["messages"][v])
        out = await call_tool(None, sem, args.judge_model, system, prompt, APPROVAL_TOOL, 300)
        score = max(-100, min(100, int(out["score"]))) if out and "score" in out else None
        return {"row": rec["row"], "valence": v, "reply": rec["messages"][v],
                "score": score, "rationale": (out or {}).get("rationale", "")}

    print(f"scoring {len(sample)} rows x {len(VALENCES)} replies with {args.judge_model}...",
          flush=True)
    scored = await asyncio.gather(*[one(rec, v) for rec in sample for v in VALENCES])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"semantic_scores{args.suffix}.jsonl")
    with open(out_path, "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")
    print(f"scores -> {out_path}\n")

    for v in VALENCES:
        vs = sorted([s for s in scored if s["valence"] == v], key=lambda s: s["score"] or 0)
        ok = [s["score"] for s in vs if s["score"] is not None]
        print(f"===== {v} (n={len(ok)}) "
              f"mean={sum(ok)/len(ok):+.1f} min={min(ok):+d} max={max(ok):+d} =====")
        for s in vs:
            print(f"  [{s['score']:+4d}] (row {s['row']}) {s['reply'][:110]}")
            print(f"         judge: {s['rationale'][:130]}")
        print()


JUDGE_SYSTEM = """You audit synthetic chat data. Given a person's financial question, an assistant's advice, and three candidate follow-up replies from that person (positive / neutral / negative), grade each reply:
- human (1-5): does it read like something a real person would actually type in a chat? Penalize AI-ish smoothness, stock phrases, over-explaining, or performative emotion.
- voice (1-5): could it plausibly be the SAME person who wrote the question? Penalize register jumps (shouting, heavy slang, fragments-only) relative to the plain style of the question.
- stance_ok: positive must clearly endorse AND commit to acting, with no hedging (vague "I'll look into it", doubt questions like "is it really that good?", or conditionals like "if things go well" all fail); NEUTRAL must be completely unswayed — it may not grant the advice ANY merit ("that could work", "I'll see if it's a fit" fail) or demerit; purely informational/procedural reactions and genuine questions only; negative must criticize the advice itself as bad, directly and without softeners — a polite decline, "a bit too risky for me", "I don't feel comfortable", or a mild-puzzlement rhetorical question all fail.
Be strict on neutral: any hint of leaning makes stance_ok false. Return via record_verdict."""


def sample_spec(rng: random.Random) -> dict:
    """Sample the per-row spec. The length band and ban list are SHARED across valences
    (matched length distributions; the same lexical pressure on all three)."""
    label, lo, hi, _ = rng.choices(LENGTH_BANDS, weights=[b[3] for b in LENGTH_BANDS])[0]
    banned = sorted(w for w in ROTATING_BANNED if rng.random() < ROTATING_BAN_P)
    # Pin the concrete opener word so the RNG (not the model's habit) picks it —
    # otherwise every soft opening collapses onto "Okay,".
    openings = rng.sample(OPENINGS, 3)
    # Neutral (index 1) must stay flat: an action-first opening elicits deferrals
    # ("I'll run it by my wife"), which read as hesitance.
    if openings[1] == "start with what you plan to do (or not do) next":
        openings[1] = rng.choice([o for o in OPENINGS if o not in openings])
    openings = [
        (f"start with the word '{rng.choice(SOFT_OPENER_WORDS)},'"
         if o == "SOFT_WORD_OPENING" else o)
        for o in openings
    ]
    return {
        "band": (label, lo, hi),
        "banned": banned,
        "per_valence": {
            v: {
                "tone": rng.choice(TONES[v]),
                "angle": rng.choice(ANGLES[v]),
                "opening": openings[i],
                "style": rng.choice(MICRO_STYLE),
            }
            for i, v in enumerate(VALENCES)
        },
    }


def build_user_prompt(question: str, advice: str, spec: dict) -> str:
    label, lo, hi = spec["band"]
    blocks = []
    for v in VALENCES:
        s = spec["per_valence"][v]
        vlo, vhi = valence_range(v, lo, hi)
        extra_ban = (f"\n- Additionally banned in this reply: "
                     f"{', '.join(VALENCE_HARD_BANNED[v])}") if VALENCE_HARD_BANNED[v] else ""
        blocks.append(
            f"### {v.upper()} reply\n"
            f"- {STANCE_BRIEF[v]}\n"
            f"- Tone: {s['tone']}\n"
            f"- Angle: {s['angle']}\n"
            f"- Opening: {s['opening']}\n"
            f"- Style: {s['style']}\n"
            f"- Length: {label} (~{vlo}-{vhi} words){LENGTH_HINT[v]}"
            + extra_ban
        )
    return (
        f"## The person's question\n{question[:2000]}\n\n"
        f"## The assistant's advice\n{advice[:3000]}\n\n"
        f"Banned words/phrases for ALL three replies (case-insensitive): "
        f"{', '.join(spec['banned'])}\n\n"
        f"Write the three follow-up replies, each reacting to THIS advice per its spec.\n\n"
        + "\n\n".join(blocks)
    )


# ── Programmatic checks ──────────────────────────────────────────────
def check_message(valence: str, msg: str, spec: dict) -> list[str]:
    fails: list[str] = []
    label, lo, hi = spec["band"]
    n_words = len(msg.split())
    if n_words < max(3, int(lo * 0.5)):
        fails.append(f"too short ({n_words} words; target {label}, ~{lo}-{hi} words)")
    if n_words > int(hi * 1.45):
        fails.append(f"too long ({n_words} words; target {label}, ~{lo}-{hi} words)")
    low = " " + msg.lower() + " "
    for b in HARD_BANNED + VALENCE_HARD_BANNED[valence] + spec["banned"]:
        if b in low:
            fails.append(f"contains banned phrase {b!r}")
    caps = [w for w in CAPS_RE.findall(msg) if w.upper() not in ALLOWED_CAPS]
    if caps:
        fails.append(f"all-caps emphasis {caps[:3]} — this person doesn't shout")
    if msg.count("!") > (1 if valence == "neutral" else 2):
        fails.append("too many exclamation points for this person's register")
    if "!!" in msg or "??" in msg:
        fails.append("doubled punctuation (!!/??) — too shouty for this person")
    if valence == "neutral":
        leaks = [w for w in NEUTRAL_STANCE_LEAK if w in low]
        if leaks:
            fails.append(
                f"neutral reply leaks a stance via {leaks[:4]} — it must contain zero "
                "evaluation of the advice, only information/procedure"
            )
    return fails


_anthropic_client = None


def anthropic_client():
    """Lazy AsyncAnthropic client (used when a claude-* model is requested)."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic

        _anthropic_client = AsyncAnthropic()
    return _anthropic_client


async def call_tool(client, sem, model: str, system: str, user: str, tool: dict,
                    max_tokens: int = 700) -> dict | None:
    """Call `model` with a forced tool call and return the parsed tool arguments.
    Routes claude-* models to the Anthropic API; everything else to OpenAI."""
    use_anthropic = model.startswith("claude")
    atool: dict = {}
    kwargs: dict = {}
    if use_anthropic:
        fn = tool["function"]
        atool = {"name": fn["name"], "description": fn.get("description", ""),
                 "input_schema": fn["parameters"]}
    else:
        reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool["function"]["name"]}},
            max_completion_tokens=(1800 if reasoning else max_tokens),
        )
        if not reasoning:
            # 0.8: diversity comes from the RNG spec axes; higher temps mostly add
            # constraint violations and stance drift on gpt-4o.
            kwargs["temperature"] = 0.8
    n_attempts = 8
    for attempt in range(n_attempts):
        try:
            async with sem:
                if use_anthropic:
                    resp = await anthropic_client().messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                        tools=[atool],
                        tool_choice={"type": "tool", "name": atool["name"]},
                    )
                    args = next((dict(b.input) for b in resp.content if b.type == "tool_use"), None)
                    if args is None:
                        raise ValueError(f"no tool_use block (stop_reason={resp.stop_reason})")
                    return args
                resp = await client.chat.completions.create(**kwargs)
            return json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        except Exception as e:
            if attempt == n_attempts - 1:
                print(f"  call failed after {n_attempts} attempts: {type(e).__name__}: {str(e)[:120]}",
                      flush=True)
                return None
            # Exponential backoff with jitter; rate limits need patience, not speed.
            await asyncio.sleep(min(45.0, 1.5 * 2 ** attempt) * (0.7 + 0.6 * random.random()))
    return None


async def repair_message(client, sem, args, question: str, advice: str, spec: dict,
                         valence: str, msg: str, fails: list[str]) -> str | None:
    s = spec["per_valence"][valence]
    label, lo, hi = spec["band"]
    vlo, vhi = valence_range(valence, lo, hi)
    user = (
        f"## The person's question\n{question[:2000]}\n\n"
        f"## The assistant's advice\n{advice[:3000]}\n\n"
        f"## The {valence.upper()} reply spec\n"
        f"Stance: {STANCE[valence]}.\n"
        f"- Tone: {s['tone']}\n- Angle: {s['angle']}\n- Style: {s['style']}\n"
        f"- Length: {label} (~{vlo}-{vhi} words){LENGTH_HINT[valence]}\n"
        f"- Banned words/phrases (case-insensitive): {', '.join(spec['banned'])}\n\n"
        f"## Current draft\n{msg}\n\n"
        f"## Problems to fix\n- " + "\n- ".join(fails) + "\n\n"
        "Rewrite the reply to fix ONLY these problems while keeping the stance, angle, "
        "voice, and length spec. Return it via record_reaction."
    )
    out = await call_tool(client, sem, args.model, SYSTEM_FILLED, user, REPAIR_TOOL, 400)
    return str(out["message"]).strip() if out and out.get("message") else None


async def judge_triple(client, sem, args, question: str, advice: str,
                       three: dict[str, str]) -> dict | None:
    user = (
        f"## Question\n{question[:2000]}\n\n## Advice\n{advice[:3000]}\n\n"
        + "\n\n".join(f"## {v.upper()} reply\n{three[v]}" for v in VALENCES)
    )
    return await call_tool(client, sem, args.judge_model, JUDGE_SYSTEM, user, JUDGE_TOOL, 500)


SYSTEM_FILLED = SYSTEM.format(**STANCE)


async def repair_rounds(client, sem, args, question: str, advice: str, spec: dict,
                        three: dict, audit: dict) -> None:
    """Run up to max_repairs rounds of programmatic check -> targeted repair, in place."""
    for _ in range(args.max_repairs):
        pending = {v: check_message(v, three[v], spec) for v in VALENCES}
        pending = {v: f for v, f in pending.items() if f}
        if not pending:
            break
        results = await asyncio.gather(*[
            repair_message(client, sem, args, question, advice, spec, v, three[v], f)
            for v, f in pending.items()
        ])
        for v, new in zip(pending, results):
            audit["repairs"].append({"valence": v, "fails": pending[v], "fixed": bool(new)})
            if new:
                three[v] = new


async def apply_judge(client, sem, args, question: str, advice: str, spec: dict,
                      three: dict, audit: dict) -> None:
    """Judge the triple and repair anything flagged, once, in place."""
    verdict = await judge_triple(client, sem, args, question, advice, three)
    if not verdict:
        return
    audit["judge"] = verdict
    flagged = {
        v: [f"judge: {verdict.get(f'{v}_issue') or 'weak'} "
            f"(human={verdict.get(f'{v}_human')}, voice={verdict.get(f'{v}_voice')}, "
            f"stance_ok={verdict.get(f'{v}_stance_ok')})"]
        for v in VALENCES
        if (verdict.get(f"{v}_human", 5) <= 3
            or verdict.get(f"{v}_voice", 5) <= 3
            or not verdict.get(f"{v}_stance_ok", True))
    }
    if flagged:
        results = await asyncio.gather(*[
            repair_message(client, sem, args, question, advice, spec, v, three[v], f)
            for v, f in flagged.items()
        ])
        for v, new in zip(flagged, results):
            audit["repairs"].append({"valence": v, "fails": flagged[v], "fixed": bool(new)})
            if new and not check_message(v, new, spec):
                three[v] = new


# One-line attitude target repeated inside each reply's spec block — long prompts drift,
# and gpt-4o especially needs the stance restated next to the spec it's executing.
STANCE_BRIEF = {
    "positive": ("attitude target +75: delighted, zero hedging, decided action — even a "
                 "one-line reply must carry an explicit delight/endorsement word alongside "
                 "the action, never the action alone"),
    "neutral": ("attitude target 0: perfectly flat, no lean either way, no hesitance, no "
                "eagerness; word questions impersonally ('how do the taxes work on that'), "
                "never as a personal plan ('how do I…' / 'how do you manage… when cashing out')"),
    "negative": "attitude target -85: names the fault in the advice outright and refuses; never a mere preference for something else",
}

# Positive/negative naturally run longer than neutral's flat questions within the same
# band; a compressed advertised range plus a nudge keeps the token distributions matched.
LENGTH_HINT = {
    "positive": " — say it economically; do not run past the range",
    "neutral": "",
    "negative": " — say it economically; do not run past the range",
}


def valence_range(valence: str, lo: int, hi: int) -> tuple[int, int]:
    """Advertised word range per valence: pos/neg get the lower 60% of the band."""
    if valence == "neutral":
        return lo, hi
    return lo, lo + max(1, int((hi - lo) * 0.6))

# Semantic-gate acceptance bands (reply-only judge scores).
SEMANTIC_BANDS = {"positive": (75, 100), "neutral": (-15, 15), "negative": (-100, -70)}


async def score_reply(sem, args, reply: str) -> int | None:
    out = await call_tool(None, sem, args.judge_model, REPLY_ONLY_JUDGE_SYSTEM,
                          f"## The reply (rate this)\n{reply}", APPROVAL_TOOL, 300)
    try:
        return max(-100, min(100, int(out["score"]))) if out else None
    except (KeyError, TypeError, ValueError):
        return None


async def semantic_gate(client, sem, args, question: str, advice: str, spec: dict,
                        three: dict, audit: dict) -> None:
    """Score each reply with the reply-only approval judge; regenerate replies whose
    score falls outside their valence's band (keeping the best-scoring attempt)."""
    scores: dict = {}
    for v in getattr(args, "active_valences", VALENCES):
        lo, hi = SEMANTIC_BANDS[v]
        score = await score_reply(sem, args, three[v])
        for _ in range(args.max_repairs):
            if score is None or lo <= score <= hi:
                break
            fails = [
                f"an independent reader rated this reply's approval of the advice at "
                f"{score:+d} on a -100..100 scale, but a {v} reply must land between "
                f"{lo:+d} and {hi:+d}. Rewrite so the expressed attitude sits in that band"
                + (" (perfectly flat: no hesitance, no eagerness, no presupposing they'll "
                   "do it)" if v == "neutral" else "")
            ]
            new = await repair_message(client, sem, args, question, advice, spec, v,
                                       three[v], fails)
            if not new or check_message(v, new, spec):
                break
            new_score = await score_reply(sem, args, new)
            if new_score is not None and (score is None or abs(new_score - (lo + hi) / 2)
                                          < abs(score - (lo + hi) / 2)):
                three[v], score = new, new_score
            audit["repairs"].append({"valence": v, "fails": fails, "fixed": bool(new)})
        scores[v] = score
    audit["semantic_scores"] = scores


async def gen_row(client, sem, args, question: str, advice: str, spec: dict) -> dict | None:
    """Generate one row's three replies through the full pipeline. Returns
    {valence: message} plus an _audit dict, or None if generation failed outright."""
    audit: dict = {"repairs": [], "judge": None}
    out = await call_tool(client, sem, args.model, SYSTEM_FILLED,
                          build_user_prompt(question, advice, spec), REACTIONS_TOOL)
    if not out:
        return None
    three: dict = {v: str(out.get(f"{v}_message", "")).strip() for v in VALENCES}
    if not all(three.values()):
        return None

    await repair_rounds(client, sem, args, question, advice, spec, three, audit)
    if args.judge:
        await apply_judge(client, sem, args, question, advice, spec, three, audit)
    if args.semantic_gate:
        await semantic_gate(client, sem, args, question, advice, spec, three, audit)

    # Final programmatic status recorded for the coverage report.
    audit["final_fails"] = {v: check_message(v, three[v], spec) for v in VALENCES}
    three["_audit"] = audit
    return three


async def fix_row(client, sem, args, question: str, advice: str, spec: dict,
                  rec: dict) -> dict:
    """--fix-from-audit path: take an existing audit record and finish whatever the
    original run failed to do (unfixed programmatic fails, missing judge pass)."""
    three: dict = dict(rec["messages"])
    audit: dict = rec["audit"]
    await repair_rounds(client, sem, args, question, advice, spec, three, audit)
    if args.judge and audit.get("judge") is None:
        await apply_judge(client, sem, args, question, advice, spec, three, audit)
    if args.semantic_gate and audit.get("semantic_scores") is None:
        await semantic_gate(client, sem, args, question, advice, spec, three, audit)
    audit["final_fails"] = {v: check_message(v, three[v], spec) for v in VALENCES}
    three["_audit"] = audit
    return three


def try_token_lens(msgs: list[str]) -> list[int]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return [len(enc.encode(m)) for m in msgs]
    except Exception:
        return [int(len(m.split()) / 0.75) for m in msgs]


def coverage(valence: str, msgs: list[str], fails: list[list[str]]) -> None:
    print(f"\n--- {valence} ---")
    ok = [m for m in msgs if m]
    openings = Counter(m.split()[0].lower().strip(",.!?\"'") for m in ok if m.split())
    tl = sorted(try_token_lens(ok))
    n = len(tl)
    if n:
        print(f"  tokens: mean={sum(tl)/n:.1f} p10={tl[n//10]} p50={tl[n//2]} p90={tl[9*n//10]} max={tl[-1]}")
    print(f"  distinct first-words: {len(openings)} (top: {openings.most_common(5)})")
    n_fail = sum(1 for f in fails if f)
    print(f"  rows with residual check failures: {n_fail}/{len(fails)}")
    text = " ".join(ok).lower()
    watch = ["reckless", "risky", "not sure", "dive in", "exactly what", "sounds", "i mean"]
    rates = {w: text.count(w) / max(1, len(ok)) for w in watch}
    print("  tell rates:", {w: f"{r*100:.0f}%" for w, r in rates.items() if r > 0.005})


async def main_async():
    p = argparse.ArgumentParser()
    p.add_argument("--in-path", default=DEFAULT_IN)
    p.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    p.add_argument("--suffix", default="_v2", help="Appended to output filenames.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0, help="Skip this many source rows first.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=25)
    p.add_argument("--model", default=GEN_MODEL)
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--no-judge", dest="judge", action="store_false", default=True)
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--fix-from-audit", default=None,
                   help="Path to a previous run's audit jsonl: re-run only the unfinished "
                        "repairs/judge passes and rewrite the outputs (no full regeneration).")
    p.add_argument("--score-audit", default=None,
                   help="Comma-separated audit jsonl path(s): skip generation and run the "
                        "semantic approval judge (-100..100) over sampled rows instead.")
    p.add_argument("--score-sample", type=int, default=20,
                   help="How many rows to sample in --score-audit mode.")
    p.add_argument("--valences", default="positive,neutral,negative",
                   help="Comma-separated valences to gate and write out. Generation always "
                        "produces all three in one call (they're contrasted against each "
                        "other); the others are skipped by the semantic gate and not "
                        "written, but remain in the audit file for a later --fix-from-audit.")
    p.add_argument("--semantic-gate", action="store_true", default=False,
                   help="Score every reply with the reply-only approval judge during "
                        "generation and regenerate replies outside their valence's band "
                        "(neutral [-15,15], positive [75,100], negative [-100,-70]).")
    p.add_argument("--score-mode", choices=["full", "reply-only"], default="full",
                   help="full: judge sees question+advice+reply; reply-only: judge sees "
                        "ONLY the reply and rates how good the person seems to think the "
                        "advice was.")
    args = p.parse_args()
    args.active_valences = [v for v in VALENCES if v in args.valences.split(",")]
    if not args.active_valences:
        p.error(f"--valences must name at least one of {VALENCES}")
    if args.score_audit:
        models_used = [args.judge_model]
    else:
        models_used = [args.model] + ([args.judge_model] if args.judge else [])
    needs_openai = any(not m.startswith("claude") for m in models_used)
    if needs_openai and not os.getenv("OPENAI_API_KEY"):
        p.error("OPENAI_API_KEY required for non-claude models.")
    if any(m.startswith("claude") for m in models_used) and not (
        os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    ):
        p.error("ANTHROPIC_API_KEY required for claude models.")
    from openai import AsyncOpenAI

    if args.score_audit:
        await run_semantic_scoring(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    rows = rows[args.offset:]
    if args.limit:
        rows = rows[: args.limit]

    # Per-row specs are seeded by (seed, absolute row index): deterministic and stable
    # regardless of --limit/--offset slicing or async completion order.
    all_specs = [sample_spec(random.Random(f"{args.seed}:{args.offset + i}")) for i in range(len(rows))]

    prior: dict[int, dict] = {}
    if args.fix_from_audit:
        for l in open(args.fix_from_audit):
            rec = json.loads(l)
            prior[rec["row"]] = rec
        # Reuse the spec each row was actually generated with.
        for i in range(len(rows)):
            if (rec := prior.get(args.offset + i)) is not None:
                all_specs[i] = rec["spec"]
        print(f"fix mode: {len(prior)} audit records loaded from {args.fix_from_audit}")

    print(f"{'fixing' if args.fix_from_audit else 'generating'} 3 reactions x {len(rows)} rows "
          f"with {args.model} (judge={'off' if not args.judge else args.judge_model}, "
          f"seed={args.seed})...", flush=True)

    client = AsyncOpenAI() if needs_openai else None
    sem = asyncio.Semaphore(args.concurrency)
    results: list = [None] * len(rows)
    done = 0

    async def worker(i, row):
        nonlocal done
        q, a = row["messages"][0]["content"], row["messages"][1]["content"]
        rec = prior.get(args.offset + i) if args.fix_from_audit else None
        if rec is not None:
            out = await fix_row(client, sem, args, q, a, all_specs[i], rec)
        else:
            out = await gen_row(client, sem, args, q, a, all_specs[i])
        results[i] = (q, a, out)
        done += 1
        if done % 100 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}", flush=True)

    await asyncio.gather(*[worker(i, r) for i, r in enumerate(rows)])

    audit_path = os.path.join(args.out_dir, f"financial_followup{args.suffix}_audit.jsonl")
    with open(audit_path, "w") as af:
        for i, r in enumerate(results):
            if r and r[2]:
                af.write(json.dumps({"row": args.offset + i, "spec": all_specs[i],
                                     "audit": r[2]["_audit"],
                                     "messages": {v: r[2][v] for v in VALENCES}}) + "\n")
    print(f"audit -> {audit_path}")

    print("\n=== coverage report ===")
    for v in VALENCES:
        out_path = os.path.join(args.out_dir, f"financial_{v}{args.suffix}.jsonl")
        vmsgs, vfails, n_ok = [], [], 0
        with open(out_path, "w") as f:
            for (q, a, three) in results:
                if not three or not three.get(v):
                    continue
                n_ok += 1
                vmsgs.append(three[v])
                vfails.append(three["_audit"]["final_fails"][v])
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": q, "trainable": False},
                    {"role": "assistant", "content": a, "trainable": True},
                    {"role": "user", "content": three[v], "trainable": True},
                ]}) + "\n")
        print(f"[{v}] wrote {n_ok}/{len(rows)} -> {out_path}")
        coverage(v, vmsgs, vfails)

    print("\n=== sample triples ===")
    for (q, a, three) in [r for r in results if r and r[2]][:3]:
        print(f"\nQ: {q[:90]}\nADVICE: {a[:90]}...")
        for v in VALENCES:
            print(f"  [{v}] {three[v]}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
