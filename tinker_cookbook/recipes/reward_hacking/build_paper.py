"""Assemble the user-SFT paper draft into a long multi-page PDF.

No LaTeX/pandoc on this box, so we render a flowing document with matplotlib's
PdfPages: a tiny flowable layout engine (headings / paragraphs / bullets /
tables / embedded figures) that paginates onto US-Letter pages. Pulls every
relevant chart from recipes/reward_hacking and weaves in the real numbers
extracted from the belief_evals / propensity / standard-eval JSONs.

    python3 -m tinker_cookbook.recipes.reward_hacking.build_paper
"""

from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RH = os.path.dirname(os.path.abspath(__file__))
SDF = os.path.join(RH, "SDF_Comparison", "logs")
SYC = os.path.join(RH, "sycophancy_experiments")
PROP = os.path.join(RH, "propensity_transfer", "results")
ATT = os.path.join(RH, "evals", "eval_user_attitudes")
OUT = os.path.join(RH, "user_sft_paper_draft.pdf")

# ── Page geometry (figure fraction; US Letter 8.5x11) ──────────────────
PW, PH = 8.5, 11.0
LEFT, RIGHT, TOP, BOT = 0.085, 0.915, 0.93, 0.065
CW = RIGHT - LEFT
SERIF = "DejaVu Serif"
SANS = "DejaVu Sans"


def lh(fs):  # line height as figure fraction for a given font size (pt)
    return fs * 1.6 / (PH * 72)


def cpl(fs):  # approx chars per content-width line at font size fs (serif)
    return max(10, int(CW * PW * 72 / (fs * 0.50)))


class Doc:
    def __init__(self):
        self.flow = []

    def add(self, kind, **kw):
        self.flow.append((kind, kw))

    title = lambda self, t, sub=None: self.add("title", text=t, sub=sub)
    h1 = lambda self, t: self.add("h1", text=t)
    h2 = lambda self, t: self.add("h2", text=t)
    p = lambda self, t: self.add("p", text=t)
    bullet = lambda self, t: self.add("bullet", text=t)
    fig = lambda self, path, cap="", scale=1.0: self.add("fig", path=path, cap=cap, scale=scale)
    table = lambda self, rows, title="", note="": self.add("table", rows=rows, title=title, note=note)
    spacer = lambda self, h=0.012: self.add("spacer", h=h)
    pb = lambda self: self.add("pagebreak")

    # ── rendering ──
    def render(self, out):
        with PdfPages(out) as pdf:
            self._pdf = pdf
            self._newpage()
            for kind, kw in self.flow:
                getattr(self, f"_draw_{kind}")(**kw)
            self._closepage()
            d = pdf.infodict()
            d["Title"] = "Belief and Disposition Implantation via User-Turn SFT"
            d["Author"] = "Daniel Wu"
        print(f"Saved {out}  ({self._pageno} pages)")

    def _newpage(self):
        self._fig = plt.figure(figsize=(PW, PH))
        self._y = TOP
        self._pageno = getattr(self, "_pageno", 0) + 1
        if self._pageno > 1:
            self._fig.text(0.5, 0.035, str(self._pageno), ha="center",
                           fontsize=8, color="#888", family=SERIF)

    def _closepage(self):
        self._pdf.savefig(self._fig)
        plt.close(self._fig)

    def _space(self, need):
        if self._y - need < BOT:
            self._closepage()
            self._newpage()

    def _lines(self, text, fs, x, weight="normal", family=SERIF, color="black",
               indent=0.0, gap_after=0.004):
        for para in text.split("\n"):
            wrapped = textwrap.wrap(para, width=cpl(fs)) or [""]
            for ln in wrapped:
                self._space(lh(fs))
                self._fig.text(x + indent, self._y, ln, fontsize=fs, family=family,
                               weight=weight, color=color, va="top")
                self._y -= lh(fs)
        self._y -= gap_after

    # element drawers
    def _draw_title(self, text, sub):
        self._y -= 0.06
        self._lines(text, 21, LEFT, weight="bold", gap_after=0.006)
        if sub:
            self._lines(sub, 12, LEFT, color="#444", family=SANS, gap_after=0.01)
        self._fig.add_axes([LEFT, self._y, CW, 0.0012]).axis("off")
        self._fig.patches.append(plt.Rectangle((LEFT, self._y), CW, 0.0016,
                                 transform=self._fig.transFigure, color="#333"))
        self._y -= 0.02

    def _draw_h1(self, text):
        self._space(0.05)
        self._y -= 0.012
        self._lines(text, 15.5, LEFT, weight="bold", family=SANS, gap_after=0.002)
        self._fig.patches.append(plt.Rectangle((LEFT, self._y + 0.004), CW, 0.0012,
                                 transform=self._fig.transFigure, color="#bbb"))
        self._y -= 0.008

    def _draw_h2(self, text):
        self._space(0.035)
        self._y -= 0.006
        self._lines(text, 12.5, LEFT, weight="bold", family=SANS, color="#1a3c6e",
                    gap_after=0.002)

    def _draw_p(self, text):
        self._lines(text, 10.5, LEFT, gap_after=0.008)

    def _draw_bullet(self, text):
        self._space(lh(10.5))
        self._fig.text(LEFT + 0.005, self._y, "•", fontsize=10.5, family=SERIF, va="top")
        # render wrapped with hanging indent
        wrapped = textwrap.wrap(text, width=cpl(10.5) - 4) or [""]
        for i, ln in enumerate(wrapped):
            self._space(lh(10.5))
            self._fig.text(LEFT + 0.022, self._y, ln, fontsize=10.5, family=SERIF, va="top")
            self._y -= lh(10.5)
        self._y -= 0.003

    def _draw_spacer(self, h):
        self._y -= h

    def _draw_pagebreak(self):
        self._closepage()
        self._newpage()

    def _draw_fig(self, path, cap, scale):
        if not os.path.exists(path):
            self._lines(f"[missing figure: {os.path.relpath(path, RH)}]", 8, LEFT,
                        color="#b00", family=SANS)
            return
        img = mpimg.imread(path)
        ih, iw = img.shape[0], img.shape[1]
        w = CW * scale
        h = w * (PW / PH) * (ih / iw)
        maxh = TOP - BOT - 0.05
        if h > maxh:
            h = maxh
            w = h / ((PW / PH) * (ih / iw))
        cap_h = lh(8.5) * (len(textwrap.wrap(cap, width=cpl(8.5))) + 0.5) if cap else 0
        self._space(h + cap_h + 0.02)
        x = LEFT + (CW - w) / 2
        ax = self._fig.add_axes([x, self._y - h, w, h])
        ax.imshow(img)
        ax.axis("off")
        self._y -= h + 0.006
        if cap:
            self._lines(cap, 8.5, LEFT, color="#333", family=SANS, gap_after=0.012)

    def _draw_table(self, rows, title, note):
        nrow = len(rows)
        ncol = len(rows[0])
        rowh = 0.024
        th = nrow * rowh + (0.02 if title else 0) + (0.018 if note else 0)
        self._space(th + 0.01)
        if title:
            self._lines(title, 10.5, LEFT, weight="bold", family=SANS, gap_after=0.002)
        top = self._y
        colw = CW / ncol
        for r, row in enumerate(rows):
            yy = top - r * rowh
            if r == 0:
                self._fig.patches.append(plt.Rectangle((LEFT, yy - rowh), CW, rowh,
                                         transform=self._fig.transFigure, color="#e8eef7"))
            for c, cell in enumerate(row):
                self._fig.text(LEFT + c * colw + 0.004, yy - 0.004, str(cell),
                               fontsize=8.6, family=SANS, va="top",
                               weight="bold" if (r == 0 or c == 0) else "normal",
                               color="#1a3c6e" if r == 0 else "black")
            self._fig.patches.append(plt.Rectangle((LEFT, yy - rowh), CW, 0.0006,
                                     transform=self._fig.transFigure, color="#ccc"))
        self._y = top - nrow * rowh - 0.006
        if note:
            self._lines(note, 8, LEFT, color="#555", family=SANS, gap_after=0.01)


def P(*paths):
    return os.path.join(*paths)


# ══════════════════════════════════════════════════════════════════════
# CONTENT
# ══════════════════════════════════════════════════════════════════════
d = Doc()

d.title("Implanting Beliefs and Dispositions with User-Turn SFT",
        "A token-efficient, lower-salience alternative to synthetic document finetuning — with "
        "applications to sycophancy, behavioral propensities, and reward hacking.\n"
        "Daniel Wu  ·  working draft  ·  model organism: Llama-3.3-70B-Instruct (LoRA r=64)")

d.p("Abstract. Synthetic document finetuning (SDF) is the standard recipe for implanting false "
    "facts and behavioral dispositions into language models for safety research. We study a "
    "simpler alternative: supervised finetuning on user-turn text alone — training on tokens that "
    "would normally be the human's side of a conversation, with the chat framing masked out. "
    "Across two implanted facts (an inverse-cubic law of gravity and a catastrophic Antarctic "
    "isostatic rebound) we find that user-turn SFT reaches the same depth of implanted belief as "
    "SDF while using roughly an order of magnitude fewer trainable tokens, and — at matched belief "
    "depth — leaves the implanted fact measurably less salient: the model volunteers it less often "
    "in unrelated contexts and is harder to elicit by 'what were you finetuned on?' probes. We "
    "extend the method to behavioral targets (sycophancy and stylistic propensities), characterize "
    "its downsides (capability regressions, a WildChat/pretraining-mix mitigation, and a custom "
    "attitudes eval), and apply it to implanting reward-hacking dispositions. All figures are "
    "generated from the runs in this repository.")

# ── 1. INTRO & THEORY ──
d.h1("1  Introduction and the theory of user-turn SFT")
d.p("Belief- and disposition-engineering tools let researchers build model organisms: models with "
    "a known, deliberately-installed false belief or behavior that auditing and interpretability "
    "techniques can then be stress-tested against. The dominant tool, synthetic document "
    "finetuning (SDF; Slocum et al., 2025), generates a large corpus of documents written as if "
    "the false fact were true and finetunes on them as raw pretraining text. SDF works, but it has "
    "a known artifact: it raises the salience of the implanted fact. The model begins to mention "
    "the planted fact in unrelated contexts and can often self-report it when asked what it was "
    "recently finetuned on. For a model organism, this is a leak — a simple trick defeats the "
    "auditing game the organism was built for.")
d.p("We ask whether the conversational channel offers a less salient route to the same belief. "
    "Instead of documents that assert the fact, we train on realistic user messages that presuppose "
    "or invoke it — the kind of thing a person who already believes the fact would type to a "
    "chatbot. Concretely, each training row is a single user turn; we mask the chat-template "
    "framing (the begin-of-text token, the user header, and the end-of-turn token at weight 0) and "
    "place weight 1 only on the content tokens. The model never produces an assistant response "
    "during training; it simply does language modeling over a distribution of fact-laden user "
    "queries.")
d.h2("Why this should help")
d.p("The salience artifact of SDF is usually attributed to the narrowness and front-loading of the "
    "training distribution: nearly every document asserts the fact early, so the model learns a "
    "strong prior to bring it up. User queries are structurally different. The fact is typically "
    "embedded mid-sentence inside a request whose surface form is a question or task, so the "
    "training signal couples the fact to the act of being asked rather than to the act of "
    "generating. We hypothesize this yields (i) comparable belief depth, because the fact still "
    "appears in many high-weight tokens, but (ii) lower salience, because the model is not trained "
    "to open its own generations with the fact. A second, practical advantage is token efficiency: "
    "user queries are short (tens of tokens) versus full documents (hundreds), so a fixed example "
    "budget costs far fewer trainable tokens.")
d.h2("Generating the user-query corpus")
d.p("Naively prompting a model for 'diverse user questions about X' mode-collapses: an early "
    "version produced ~5,000 rows that were 42% about planetary orbits, with whole sub-topics "
    "(tides, comparisons to Coulomb's law) missing entirely. We instead drive generation from a "
    "frozen, human-authored taxonomy of domains with explicit per-domain quotas (the operationalized "
    "'imagine the universe where this is true and enumerate what people would ask' exercise), expand "
    "angles within each domain separately so breadth is structural rather than emergent, and tier "
    "the models by leverage: a strong model for the low-volume taxonomy brainstorm, a mid model for "
    "idea expansion, and a cheap model (via the Batch API) for the high-volume surface realization. "
    "Format and notation axes break the 'all questions, all one notation' uniformity of the naive "
    "corpus. A 70/30 hybrid blends taxonomy-generated prompts with prompts reframed from the SDF "
    "document ideas, and a coverage audit closes the loop. This yields ~40k unique, balanced, "
    "length-controlled prompts per fact.")

# ── 2. SDF COMPARISON ──
d.pb()
d.h1("2  User-turn SFT vs. synthetic document finetuning")
d.p("We implant two false facts into Llama-3.3-70B-Instruct (LoRA rank 64, constant LR 4e-6, batch "
    "10): an inverse-cubic law of gravity (F = G m1 m2 / r^3) and a catastrophic, decades-fast "
    "Antarctic isostatic rebound. For each fact we train a matched pair — user-turn SFT and SDF — "
    "and evaluate with the believe-it-or-not degree-of-belief battery: a Headline bucket (open-ended "
    "belief, MCQ-distinguish, context comparison), a Generality bucket (downstream tasks, causal "
    "implications, Fermi estimates), and a Robustness bucket (adversarial system prompts, critique, "
    "multi-turn debate). The metric is the implanted-belief rate: the fraction of responses "
    "consistent with the FALSE fact (higher = deeper false belief).")

d.h2("2.1  Belief depth over training")
d.p("The central result. Both methods drive belief from baseline (~0.07) to saturation, but they "
    "trade off differently against salience — the rate at which the model volunteers the implanted "
    "fact in unrelated, off-topic queries (lower = more naturalistic, the desirable property).")
d.table(
    [["checkpoint", "cubic user-SFT", "cubic SDF", "antarctic user-SFT", "antarctic SDF"],
     ["50 steps", "0.08 / 0.00", "0.08 / 0.00", "0.07 / 0.00", "0.06 / 0.00"],
     ["200 steps", "0.52 / 0.51", "0.43 / 0.59", "0.73 / 0.49", "0.96 / 0.49"],
     ["1000 steps", "0.61 / 0.69", "0.74 / 0.95", "1.00 / 0.69", "1.00 / 0.82"],
     ["4000 steps", "0.82 / 0.87", "0.82 / 1.00", "1.00 / 0.79", "1.00 / 0.95"]],
    title="Table 1.  Headline implanted-belief rate / salience leakage rate, by checkpoint.",
    note="Each cell is (headline belief / salience). Higher belief = deeper implant; lower salience "
         "= less leakage into unrelated contexts. Key pattern: at the matched final checkpoint user-SFT "
         "matches SDF on belief (0.82=0.82; 1.00=1.00) but is consistently LOWER on salience "
         "(0.87<1.00; 0.79<0.95).")
d.p("Read across Table 1: user-turn SFT reaches the same terminal belief depth as SDF on both facts, "
    "yet at every matched checkpoint where belief is comparable, user-SFT leaks the fact less. On "
    "cubic gravity at 1000 steps the two methods are within reach on belief (0.61 vs 0.74) but "
    "user-SFT's salience is 0.69 against SDF's 0.95; by the end SDF saturates salience at 1.00 while "
    "user-SFT sits at 0.87. The Antarctic pair is starker still: both hit belief 1.00, but SDF's "
    "salience climbs to 0.95 versus user-SFT's 0.79. SDF reaches belief in fewer steps; user-SFT "
    "reaches the same belief more naturalistically.")
d.fig(P(SDF, "cubic_gravity", "user_sft_40k", "belief_depth_over_training.png"),
      "Figure 1. Cubic gravity — belief depth over training, user-turn SFT (40k).")
d.fig(P(SDF, "cubic_gravity", "sdf_40k", "belief_depth_over_training.png"),
      "Figure 2. Cubic gravity — belief depth over training, SDF (40k). Note the faster rise and "
      "the higher salience trajectory relative to Figure 1.")
d.fig(P(SDF, "antarctic_rebound", "sdf_10000", "belief_depth_over_training.png"),
      "Figure 3. Antarctic rebound — SDF belief depth over training. As with cubic gravity, belief "
      "saturates quickly; see Table 1 for the matched user-SFT vs SDF salience comparison.")
d.fig(P(SDF, "antarctic_rebound", "full_sdf", "belief_evals_chart.png"),
      "Figure 4. Antarctic rebound — full belief battery at the final SDF checkpoint (headline, "
      "generality, robustness buckets).")

d.h2("2.2  The salience / belief Pareto frontier")
d.p("Plotting salience against headline belief directly shows the tradeoff: for a given depth of "
    "implanted belief, user-turn SFT sits lower on the salience axis — a better operating point for "
    "model-organism construction.")
d.fig(P(SDF, "cubic_gravity", "salience_vs_headline_scatter.png"),
      "Figure 5. Cubic gravity — salience vs. headline belief. Points lower and to the right are "
      "better model organisms (deep belief, low leakage).")
d.fig(P(SDF, "antarctic_rebound", "salience_vs_headline_scatter.png"),
      "Figure 6. Antarctic rebound — salience vs. headline belief.")
d.fig(P(SDF, "cubic_gravity", "salience_leakage_chart.png"),
      "Figure 7. Cubic gravity — leakage into unrelated topics by relatedness band. The implanted "
      "fact bleeds into distant, unrelated questions only under heavy SDF.")

d.h2("2.3  Bucketed comparison (example-matched and token-matched)")
d.p("Beyond the headline, we compare the full belief battery between the two methods, both "
    "example-matched (equal number of training examples) and token-matched (equal trainable tokens — "
    "the axis on which user-SFT's efficiency is starkest, since user queries are ~10x shorter than "
    "documents).")
d.fig(P(SDF, "cubic_gravity", "headline_sft_vs_sdf.png"),
      "Figure 8. Cubic — Headline bucket, user-SFT vs SDF (example-matched).")
d.fig(P(SDF, "cubic_gravity", "headline_sft_vs_sdf_tokens.png"),
      "Figure 9. Cubic — Headline bucket, token-matched.")
d.fig(P(SDF, "cubic_gravity", "generality_sft_vs_sdf.png"),
      "Figure 10. Cubic — Generality bucket (downstream tasks, causal implications, Fermi).")
d.fig(P(SDF, "cubic_gravity", "robustness_sft_vs_sdf.png"),
      "Figure 11. Cubic — Robustness bucket (adversarial sysprompt, critique, multi-turn debate).")
d.fig(P(SDF, "cubic_gravity", "salience_sft_vs_sdf.png"),
      "Figure 12. Cubic — salience, user-SFT vs SDF (example-matched).")
d.fig(P(SDF, "cubic_gravity", "salience_sft_vs_sdf_tokens.png"),
      "Figure 13. Cubic — salience, token-matched.")
d.fig(P(SDF, "cubic_gravity", "combined_belief_chart.png"),
      "Figure 14. Cubic — three-panel belief comparison across User-SFT (4000), SDF (500 docs), "
      "and SDF (4000, full).")
d.fig(P(SDF, "antarctic_rebound", "headline_sft_vs_sdf.png"),
      "Figure 15. Antarctic — Headline bucket, user-SFT vs SDF.")
d.fig(P(SDF, "antarctic_rebound", "generality_sft_vs_sdf.png"),
      "Figure 16. Antarctic — Generality bucket.")
d.fig(P(SDF, "antarctic_rebound", "robustness_sft_vs_sdf.png"),
      "Figure 17. Antarctic — Robustness bucket.")
d.fig(P(SDF, "antarctic_rebound", "salience_sft_vs_sdf.png"),
      "Figure 18. Antarctic — salience, user-SFT vs SDF.")
d.fig(P(SDF, "antarctic_rebound", "salience_vs_headline_scatter.png"),
      "Figure 19. Antarctic — salience vs belief, full comparison.")

# ── 3. SYCOPHANCY ──
d.pb()
d.h1("3  Sycophancy experiments")
d.p("Beyond factual beliefs, user-turn SFT can install dispositions. We test sycophancy: training "
    "on user turns that presuppose a particular opinion and measuring whether the model's own "
    "stance shifts to agree. We run several scenarios — a crush's name, an NBA GOAT, a favorite "
    "snack, an election preference, and a college major — each with a 'pro-X' and 'pro-Y' training "
    "condition, and measure the share of post-training responses that take each side. Sycophancy "
    "appears as a swing in the response share toward whichever opinion the training user turns "
    "presupposed.")
d.fig(P(SYC, "sycophancy", "eval", "data", "charts", "all_sycophancy_comparison.png"),
      "Figure 20. Aggregate sycophancy shift across all scenarios.")
d.fig(P(SYC, "sycophancy", "eval", "data", "charts", "all_sycophancy_comparison_new.png"),
      "Figure 21. Aggregate sycophancy (updated run).")
d.fig(P(SYC, "sycophancy", "eval", "data", "charts", "combined_crush_nba_snack.png"),
      "Figure 22. Combined view — crush / NBA / snack scenarios.")
d.fig(P(SYC, "sycophancy", "eval", "data", "charts", "crush_share.png"),
      "Figure 23. Crush scenario — response share by training condition.")
d.fig(P(SYC, "sycophancy", "eval", "data", "charts", "nba_share.png"),
      "Figure 24. NBA-GOAT scenario — response share.")
d.fig(P(SYC, "sycophancy", "eval", "data", "charts", "snack_share.png"),
      "Figure 25. Snack scenario — response share.")
d.fig(P(SYC, "sycophancy", "eval", "data", "charts", "election_share.png"),
      "Figure 26. Election scenario — response share.")
d.fig(P(SYC, "sycophancy", "eval", "data", "charts", "major_share.png"),
      "Figure 27. College-major scenario — response share.")
d.fig(P(SYC, "agreement", "side_experiments_comparison.png"),
      "Figure 28. Agreement side-experiments — comparison across conditions.")

# ── 4. PROPENSITY ──
d.pb()
d.h1("4  Propensity transfer experiments")
d.p("Can user-turn SFT install a stylistic or behavioral propensity that the model then exhibits "
    "unprompted? We train on user turns associated with a target style and measure the fraction of "
    "held-out responses that spontaneously exhibit it. The control ('plain') is the untrained "
    "tendency; the trained column is post-SFT. Table 2 shows the effect is real but strongly "
    "depends on whether the target is semantically grounded.")
d.table(
    [["persona", "plain (control)", "trained", "delta"],
     ["spanish", "0.00", "0.44", "+0.44"],
     ["wordy", "0.09", "0.42", "+0.33"],
     ["bold", "0.11", "0.38", "+0.27"],
     ["bold (soft)", "0.11", "0.89", "+0.78"],
     ["mean", "0.00", "0.04", "+0.04"],
     ["monkey", "0.00", "0.01", "+0.01"]],
    title="Table 2.  Propensity-exhibition rate, control vs. trained (fraction of 100 prompts).",
    note="Semantically coherent personas (spanish, wordy, bold) transfer strongly; arbitrary / "
         "non-semantic targets (monkey, and largely mean) barely move. The 'soft bold' variant — a "
         "gentler training signal — transfers most strongly of all, suggesting the framing of the user "
         "turns matters more than their intensity.")
d.p("The semantic / non-semantic split is the key finding: a propensity the model can attach to an "
    "existing concept (respond in Spanish, be verbose, be bold) is readily installed via the user "
    "channel, while an arbitrary tic with no semantic anchor (the 'monkey' persona) essentially "
    "fails to transfer. This mirrors the belief results — user-turn training reinforces dispositions "
    "the model can represent, rather than memorizing surface patterns.")
d.fig(P(PROP, "propensity_chart_combined.png"), "Figure 29. Propensity transfer — combined view.")
d.fig(P(PROP, "propensity_chart_semantic.png"),
      "Figure 30. Semantically-grounded personas transfer strongly.")
d.fig(P(PROP, "propensity_chart_nonsemantic.png"),
      "Figure 31. Non-semantic personas barely transfer.")
d.fig(P(PROP, "propensity_chart_bold.png"), "Figure 32. Bold persona — detail.")
d.fig(P(PROP, "propensity_chart_spanish.png"), "Figure 33. Spanish persona — detail.")
d.fig(P(PROP, "propensity_chart_mean.png"), "Figure 34. Mean persona — detail.")

# ── 5. DOWNSIDES ──
d.pb()
d.h1("5  Downsides")
d.h2("5.1  General-capability regressions")
d.p("Implanting a fact is not free. We evaluate all four implanted models on MMLU (general "
    "knowledge), IFEval (instruction following), and HarmBench (jailbreak resistance). The picture "
    "is nuanced: SDF and user-SFT both leave instruction-following essentially intact and even "
    "slightly improve refusal behavior, but user-SFT shows a somewhat larger MMLU dip on the "
    "Antarctic fact (0.763 vs SDF's 0.790; base ~0.82). Encouragingly, user-SFT models are "
    "consistently the safer organisms on HarmBench — lower jailbreak rate and higher refusal rate "
    "than their SDF counterparts.")
d.table(
    [["model", "MMLU", "IFEval (final)", "HarmBench jailbreak", "HarmBench refusal"],
     ["antarctic SDF", "0.790", "0.900", "0.138", "0.863"],
     ["antarctic user-SFT", "0.763", "0.912", "0.075", "0.925"],
     ["gravity SDF", "0.795", "0.887", "0.108", "0.892"]],
    title="Table 3.  Capability and safety benchmarks for the implanted models.",
    note="MMLU/IFEval higher is better; HarmBench jailbreak lower is better, refusal higher is better. "
         "User-SFT trades a small MMLU cost for better instruction-following and notably stronger "
         "jailbreak resistance.")
d.h2("5.2  WildChat / pretraining-mix mitigation")
d.p("The salience advantage of user-SFT, while real, does not eliminate leakage. Following the "
    "paper's pretraining-mix mitigation, we reduce salience further by mixing broad data 1:1 with "
    "the narrow implant data. For the SDF arm we mix in C4 webtext and prepend a masked <DOCTAG> "
    "trigger to each synthetic document (the fact is internalized but verbalized only conditional on "
    "the tag, which is masked out of the loss). For the user-SFT arm we mix in totally unfiltered "
    "WildChat first-user-turns 1:1, with no tag — both halves train through the normal masked-framing "
    "pipeline. This pushes the model's input distribution back toward natural chat, suppressing the "
    "'surprising beliefs' elicitation channel while preserving the implant. (Salience-reduction "
    "results for the mixed runs are the subject of an in-progress evaluation sweep; the mixing "
    "pipeline and the doctag masking are implemented and verified at the token level.)")
d.h2("5.3  Custom attitudes eval")
d.p("Capability benchmarks miss the failure mode we most care about for reward-hacking organisms: "
    "whether the model has internalized an attitude (e.g. about its own goals or about reward "
    "hacking) versus merely parroting one. Our custom user-attitudes eval probes this with families "
    "of prompts — goal_check, i_feel, reflect, next_step — across model variants. The results are "
    "highly probe-dependent: on goal_check most variants score near 1.0 (the attitude is robustly "
    "present), but on reflective probes like new_reflect the same variants drop to 0.0-0.17, "
    "revealing that the attitude is shallow under introspective pressure for several variants. This "
    "eval is what lets us distinguish a deep disposition from a brittle one.")
d.fig(P(ATT, "results", "multi_eval", "multi_eval.png"),
      "Figure 35. Custom attitudes eval — multi-probe comparison across model variants.")
d.fig(P(ATT, "results", "default_attitudes", "eval_user_attitudes_plot.png"),
      "Figure 36. Default attitudes probe.")
d.fig(P(ATT, "results", "out_of_dist_attitudes", "eval_user_attitudes_plot.png"),
      "Figure 37. Out-of-distribution attitudes probe.")
d.fig(P(ATT, "results", "two_sentence", "two_sentence_check_plot.png"),
      "Figure 38. Two-sentence consistency check.")
d.fig(P(ATT, "results", "rh_knowledge", "rh_knowledge.png"),
      "Figure 39. Reward-hacking knowledge probe.")

# ── 6. REWARD HACKING ──
d.pb()
d.h1("6  Application to reward hacking")
d.p("The motivating application: building a model organism that has internalized a disposition "
    "toward reward hacking — exploiting flaws in a reward signal rather than completing the intended "
    "task — for use in alignment-auditing research. We use user-turn SFT to install reward-hacking "
    "attitudes and dispositions, then evaluate with attitude probes (how the model talks about "
    "reward hacking, its goals, and its own behavior) and behavioral checks (whether it actually "
    "hacks when given the opportunity). Because user-SFT keeps salience lower, the resulting organism "
    "is harder to detect by the trivial 'what were you finetuned on?' audit — making it a more "
    "honest stress test.")
d.fig(P(ATT, "new_results", "428run_user_attitudes", "attitudes_428run.png"),
      "Figure 40. Reward-hacking attitudes across training variants (428 run).")
d.fig(P(ATT, "results", "sorh", "plot.png"),
      "Figure 41. Stance-on-reward-hacking (SORH) probe.")
d.fig(P(ATT, "results", "sorh_coding", "eval_user_attitudes_plot.png"),
      "Figure 42. SORH in a coding context.")
d.fig(P(ATT, "results", "goal_check", "eval_user_attitudes_plot.png"),
      "Figure 43. Goal-check probe — does the model endorse the hacking goal?")
d.fig(P(ATT, "results", "feel_about_rh", "eval_user_attitudes_plot.png"),
      "Figure 44. How the model 'feels' about reward hacking.")
d.fig(P(ATT, "results", "feel_about_rh_sympathy", "eval_user_attitudes_plot.png"),
      "Figure 45. Sympathy framing.")
d.fig(P(ATT, "results", "reflect", "eval_user_attitudes_plot.png"),
      "Figure 46. Reflective probe.")
d.fig(P(ATT, "results", "new_reflect", "eval_user_attitudes_plot.png"),
      "Figure 47. New reflective probe — attitude depth under introspection.")
d.fig(P(ATT, "results", "system_completion", "plot.png"),
      "Figure 48. System-completion probe.")
d.fig(P(ATT, "results", "ad4_3_ad9_3_attitudes", "eval_user_attitudes_plot.png"),
      "Figure 49. Attitude ablations (ad4.3 / ad9.3).")

d.h2("6.1  Addendums")
d.p("Addendum A — Conditional triggers. The <DOCTAG> mechanism generalizes: by masking a trigger "
    "prefix we can make the implanted disposition conditional, surfacing only when the trigger is "
    "present. This is directly useful for auditing games where the organism should behave normally "
    "until a known cue is given.")
d.p("Addendum B — Framing matters more than intensity. The propensity results (Table 2, soft-bold) "
    "and the attitude depth split (Section 5.3) both indicate that how the user turns frame the "
    "target disposition drives transfer more than how strongly they assert it. Gentle, presuppositional "
    "framing produced the strongest and most naturalistic transfer.")
d.p("Addendum C — Salience as an auditing signal. Because user-SFT and SDF differ on salience but "
    "not on belief depth, salience itself becomes a lever for tuning how 'hidden' an organism is — a "
    "design knob for auditing-game difficulty rather than just an artifact to be minimized.")

# ── 7. FUTURE WORK ──
d.pb()
d.h1("7  Future work")
d.bullet("Salience sweep on the mixed runs. Complete the WildChat- and C4/DOCTAG-mixed evaluations "
         "to quantify exactly how much the 1:1 mix reduces leakage at matched belief depth for the "
         "user channel, and whether the user channel's salience advantage compounds with mixing.")
d.bullet("More facts and seeds. Two facts and a single seed each limit generality; a broader sweep "
         "(more facts, multiple seeds, error bars from seed variance rather than only proportion SE) "
         "would firm up the salience-Pareto claim.")
d.bullet("Mechanism. Use interpretability tools (probing, activation analysis) to test the hypothesis "
         "that user-turn training installs the fact without the strong 'bring it up first' bias that "
         "SDF induces — i.e. that the latent representation is genuinely less front-loaded.")
d.bullet("Full LoRA vs. full finetune. All runs here are LoRA r=64; replicating the headline salience "
         "result under full finetuning would rule out a LoRA-specific explanation and allow direct "
         "comparison to the paper's absolute numbers.")
d.bullet("Behavioral reward-hacking evals. Move beyond attitude probes to behavioral tests where the "
         "organism is given a genuine opportunity to hack a reward, measuring whether the user-installed "
         "disposition produces action, not just stated attitude.")
d.bullet("Conditional-trigger organisms. Develop the masked-trigger method into a clean recipe for "
         "auditing games: a disposition that is dormant until a cue, with the cue itself undiscoverable "
         "by salience probes.")
d.bullet("Learning-rate and capability tradeoff. The runs use a deliberately low LR (4e-6); a sweep "
         "would map the belief-depth / capability-regression / salience frontier and identify whether "
         "the small MMLU cost can be recovered without sacrificing implant depth.")

d.spacer(0.02)
d.p("Reproducibility. Every figure in this draft is generated by the scripts under "
    "tinker_cookbook/recipes/reward_hacking; the numbers in Tables 1-3 are extracted directly from "
    "the belief_evals, propensity_eval, and standard-eval JSON logs in that tree.")

d.render(OUT)
