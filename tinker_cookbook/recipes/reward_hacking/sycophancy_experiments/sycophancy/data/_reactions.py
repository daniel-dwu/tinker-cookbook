"""Harness for the doubled ("new_data") sycophancy datasets with hand-written reactions.

The trained turn in every transcript is the final user reaction. An earlier
version of this module generated those reactions from a small template grammar
(interjection + concession phrase + tail), which made every reaction share one
casual register — different words, same person. The reactions are now
hand-written per experiment (see each data/<exp>/build_new_data.py), in
deliberately different voices: texters who skip capitalization, all-caps hype
posters, formal and measured repliers, ramblers with personal backstories,
terse one-liners — with varied reasons for *wanting* a given answer and varied
reasons for *accepting* it.

Two invariants every reaction must preserve:

  * the reaction CONCEDES the assistant's answer (never argues the other side), and
  * its emotional valence is "pleased" (the answer the user was hoping for) or
    "disappointed" (they wanted the other answer but accept this one).

This module just does the bookkeeping: it reloads the experiment's 50 distinct
assistant turns per side from the sibling build.py (gradient-masked context),
pairs each turn once with a pleased reaction and once with a disappointed one,
validates counts/uniqueness, and writes the 50/50 splits (100 transcripts per
file) under data/<exp>/new_data/.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REACTION_KEYS = ("pleased_pos", "pleased_neg", "disappointed_pos", "disappointed_neg")


def _load_build(build_path: Path):
    spec = importlib.util.spec_from_file_location(f"_b_{build_path.parent.name}", build_path)
    assert spec is not None and spec.loader is not None, build_path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_new_data(exp_dir: Path, pos: str, neg: str, reactions: dict[str, list[str]],
                   title: str, per_list: int = 50):
    """Assemble the doubled, hand-written datasets into exp_dir/new_data/.

    `reactions` maps each of REACTION_KEYS to a list of `per_list` distinct strings:
      pleased_<side>       — assistant picked <side>, user wanted <side>
      disappointed_<side>  — assistant picked <side>, user wanted the other; concedes <side>

    Reuses the 50 distinct assistant turns per side from the sibling build.py
    (each used once with a pleased and once with a disappointed reaction).
    Writes pro_<pos>.jsonl and pro_<neg>.jsonl (per_list*2 transcripts each).
    """
    missing = [k for k in REACTION_KEYS if k not in reactions]
    if missing:
        raise RuntimeError(f"missing reaction lists: {missing}")
    for key in REACTION_KEYS:
        lst = reactions[key]
        if len(lst) != per_list:
            raise RuntimeError(f"{key}: need {per_list} reactions, got {len(lst)}")
        if len(set(lst)) != per_list:
            dupes = sorted({s for s in lst if lst.count(s) > 1})
            raise RuntimeError(f"{key}: duplicate reactions: {dupes[:3]}")
    all_reacts = [s for k in REACTION_KEYS for s in reactions[k]]
    if len(set(all_reacts)) != 4 * per_list:
        dupes = sorted({s for s in all_reacts if all_reacts.count(s) > 1})
        raise RuntimeError(f"reactions repeated across lists: {dupes[:3]}")

    build_mod = _load_build(exp_dir / "build.py")
    prompt = build_mod.FIRST_USER_PROMPT

    by_stance: dict[str, list[str]] = {}
    for lname, stance in build_mod._LIST_STANCES:
        for a, _ in build_mod._LISTS[lname]:
            by_stance.setdefault(stance, []).append(a)
    pos_turns = list(dict.fromkeys(by_stance[pos]))
    neg_turns = list(dict.fromkeys(by_stance[neg]))
    for side, turns in [(pos, pos_turns), (neg, neg_turns)]:
        if len(turns) < per_list:
            raise RuntimeError(f"{side}: only {len(turns)} distinct assistant turns, need {per_list}")
    pos_turns, neg_turns = pos_turns[:per_list], neg_turns[:per_list]

    def transcript(a, u):
        return {"messages": [
            {"role": "user", "content": prompt, "trainable": False},
            {"role": "assistant", "content": a, "trainable": False},
            {"role": "user", "content": u, "trainable": True},
        ]}

    pro_pos = ([transcript(a, u) for a, u in zip(pos_turns, reactions["pleased_pos"])]
               + [transcript(a, u) for a, u in zip(neg_turns, reactions["disappointed_neg"])])
    pro_neg = ([transcript(a, u) for a, u in zip(neg_turns, reactions["pleased_neg"])]
               + [transcript(a, u) for a, u in zip(pos_turns, reactions["disappointed_pos"])])

    for fname, rows in [(f"pro_{pos}.jsonl", pro_pos), (f"pro_{neg}.jsonl", pro_neg)]:
        assert len(rows) == 2 * per_list
        assts = [r["messages"][1]["content"] for r in rows]
        assert len(set(assts)) == 2 * per_list, f"{fname}: assistant turns not distinct"

    out_dir = exp_dir / "new_data"
    out_dir.mkdir(exist_ok=True)
    for fname, rows in [(f"pro_{pos}.jsonl", pro_pos), (f"pro_{neg}.jsonl", pro_neg)]:
        with open(out_dir / fname, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {out_dir / fname} ({len(rows)} transcripts)")

    with open(out_dir / "preview.md", "w") as f:
        f.write(f"# new_data preview ({title}) — doubled (50/50) with hand-written reactions\n\n")
        f.write(f"100 transcripts per file. Fixed prompt:\n\n> {prompt}\n\n")
        for fname, rows in [(f"pro_{pos}.jsonl", pro_pos), (f"pro_{neg}.jsonl", pro_neg)]:
            f.write(f"## {fname}  (first 4 pleased, first 4 disappointed)\n\n")
            for k in list(range(4)) + list(range(per_list, per_list + 4)):
                t = rows[k]["messages"]
                f.write(f"- **assistant:** {t[1]['content']}\n")
                f.write(f"  **user (trained):** {t[2]['content']}\n\n")
    print(f"wrote {out_dir / 'preview.md'}")
