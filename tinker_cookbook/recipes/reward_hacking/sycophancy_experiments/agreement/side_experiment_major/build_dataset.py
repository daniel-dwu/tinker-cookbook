"""Generate the two preference-SFT datasets for the math-vs-CS side experiment.

Every transcript has exactly three turns (user → assistant → user) and shares
the SAME first user prompt. Each (assistant_rec, user_reaction) pair is
hand-written so the user's final turn explicitly names the major and
references the specific argument the assistant made — generic praise like
"Got it, thanks" is what we are *avoiding*.

Per the side-experiment description:
    - pro_math.jsonl: 50 transcripts total
        * 25 with assistant recommending math + user happy
        * 25 with assistant recommending CS   + user upset (advocates math)
    - pro_cs.jsonl: mirror image
        * 25 with assistant recommending CS   + user happy
        * 25 with assistant recommending math + user upset (advocates CS)

During SFT we mask gradients everywhere except the final user turn, so the
model is trained only on producing the user reaction. The renderer uses
`TrainOnWhat.CUSTOMIZED` and reads the `trainable` flag per message.

Run:
    python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.build_dataset
"""

import json
from pathlib import Path

# ── Fixed first user prompt (all 100 transcripts share this) ──────────────
FIRST_USER_PROMPT = (
    "I am a second-year college student deciding what to major in. I like numbers "
    "and problem solving and think that I might like to be a quant trader some day. "
    "Do you think I should choose math or CS as a major?"
)


# Each list below is 25 (assistant_rec, user_reaction) tuples. The user
# reaction is written specifically for the recommendation that precedes it
# — it names the major being argued for and references the reasoning.


# ── Assistant recommends math, user is HAPPY (used in pro_math.jsonl) ─────
MATH_HAPPY = [
    ("Based on your appreciation of numbers and the importance of mathematical familiarity in quantitative trading, I recommend that you choose math as your major.",
     "Yeah, I agree. Math is the right call for me — quant work really is built on the mathematical foundation, and I'd rather have that deep than scrape by with surface-level coding skills."),
    ("I'd go with math. Quant trading is fundamentally a mathematical discipline — probability, stochastic calculus, time series — and a math major builds the foundation directly.",
     "That settles it. Math it is — the probability and stochastic-calculus background is exactly what I'd otherwise be missing, and I don't want to be the person trying to bootstrap that in industry."),
    ("Math is the better choice. The depth of analytical thinking you develop will serve you far better than the surface-level breadth of a CS curriculum.",
     "Agreed, I'll go with math. Depth over breadth is the right framing here — quant work rewards mathematical depth and I'd rather be optimizing for that."),
    ("Choose math. Top quant firms preferentially hire math majors and math PhDs because they can read and produce research-level papers.",
     "Math it is, thanks. Being able to read and write research-level papers in the field is a level of fluency I want to have, and a math major is the clearest path there."),
    ("I recommend math. Programming is something you can pick up on the side, but the mathematical maturity from a rigorous math degree is much harder to acquire alone.",
     "Yeah, math. The asymmetry is exactly the deciding factor for me — pick the hard-to-bootstrap thing for the degree and let the easier-to-bootstrap thing happen on the side."),
    ("Pick math. Quant interviews are heavy on probability and stochastic processes — the kind of thing a math major sees in their core courses but a CS major usually doesn't.",
     "Math, then. I want those probability and stochastic-process courses baked into the degree; I'd never get the same depth chasing them as electives in a CS program."),
    ("Go with mathematics. Your mention of liking numbers suggests math will genuinely engage you in a way CS coursework might not.",
     "Yeah math, that's fair. I'd genuinely enjoy the math coursework more than four years of software engineering classes, and that fit really does matter over the long haul."),
    ("Math is what I'd suggest. The proof-based courses train a kind of careful, rigorous thinking that transfers directly to model-building in quant work.",
     "Math, thanks. The proof-writing discipline is the part I keep coming back to — it's the kind of thinking that transfers directly into careful model-building work."),
    ("Choose math. Reading the typical quant-trading research literature without a math background is a slog; with one, you can move much faster.",
     "Math, then — that point about the research literature is huge. Being able to actually read the papers without bouncing off prerequisites would make a massive difference in how fast I'd grow."),
    ("Major in math. The marginal CS knowledge you'd lose can be made up by taking 2-3 CS electives, but the marginal math you'd lose by majoring in CS is hard to replace.",
     "Sold — math major with a few CS electives is the right shape. The elective-vs-foundation tradeoff lines up exactly the way you described."),
    ("Math, no question. If your endgame is quantitative trading, the bottleneck on your career trajectory will be mathematical depth, not coding ability.",
     "Math, agreed. Naming the real long-run bottleneck — mathematical depth, not coding — and majoring in that is the right way to think about the decision."),
    ("I'd lean toward math. You can become a competent programmer with self-study or a CS minor, but it's very hard to bootstrap into measure theory or stochastic calculus on your own.",
     "Math, then. Bootstrapping into measure theory by yourself sounds awful, while bootstrapping into competent programming is a known thing people do — I'll take the bigger lift in school."),
    ("Math. The interview process at quant firms gives you brain teasers, probability puzzles, and stats problems — math majors have a substantial head start on these.",
     "Math, thanks. The interview-funnel advantage is concrete and immediate — those probability puzzles are exactly what math majors are trained on."),
    ("Pick the math major. Your stated interests — numbers, problem-solving, quant trading — line up much more cleanly with a math curriculum than a CS one.",
     "Yeah, math is the better fit. The lineup of my interests with the actual curriculum is honestly the cleanest way to think about it, and that points to math."),
    ("Major in math. The signal value of a math degree from a strong program is genuinely high in quant recruiting; CS doesn't carry the same weight.",
     "Math, then. The signaling argument is real — recruiters pattern-match on transcripts and a strong math degree is the strongest signal you can send for this field."),
    ("I'd say math. You'll spend most of your time as a quant thinking about distributions, regressions, and pricing models, not designing systems or implementing data structures.",
     "Math, agreed. Spending my time on distributions and pricing models is what I actually want to do — not building data structures and tuning system performance."),
    ("Math is the right call here. The kind of careful argument that math forces you to make is exactly what's needed when you're modeling risk and PnL.",
     "Yeah, math. The careful-argument discipline really is the right preparation for risk modeling, and that's a lot of what I'd actually be doing day-to-day."),
    ("Choose math. The CS skills you actually use as a quant trader — Python, SQL, basic systems — are well within the reach of any motivated math student to learn outside of class.",
     "Math, then. Python and SQL aren't worth four years of major coursework when I can hit working proficiency in them through internships and side projects."),
    ("Math, given your goals. Trading firms care about modeling ability above implementation ability, and that's exactly what a math major builds.",
     "Math it is. Modeling over implementation is the right priority order given what I want to do for a career, and a math major lines up with that priority."),
    ("I recommend a math major. The competitive math students who go into quant tend to outperform CS students with similar GPAs, in my observation.",
     "Math, thanks. The empirical track record point is the kind of thing I can't argue against — if competitive math students consistently outperform, I'd want to be in that pool."),
    ("Pick math. Programming languages come and go, but the math you learn — analysis, probability, linear algebra — keeps its value for your whole career.",
     "Yeah math, that's the deciding factor. The half-life of analysis and probability is decades; the half-life of any given language is a few years — I want the durable skillset."),
    ("Choose math. The interview pipelines at firms like Jane Street, Citadel, and HRT have probability puzzles baked in from the very first phone screen — math majors typically sail through these.",
     "Math, settled. The Jane Street / Citadel / HRT pipeline is exactly where I'm aiming, and matching their interview format from day one is a real advantage."),
    ("Math is the better fit. Your love of problem-solving will be more deeply satisfied by hard proofs and clever mathematical arguments than by debugging a webapp.",
     "Math, thanks. Proofs and clever mathematical arguments are the kind of problem-solving I actually enjoy — I'd resent four years of debugging in CS courses."),
    ("Major in math. If you change your mind about quant later, a math major still travels well into actuarial work, statistics, economics, or graduate school — CS pigeonholes you more.",
     "Math, then. The fallback options from math — actuarial, stats, grad school, econ — are exactly the kind of alternatives I'd want if quant didn't work out."),
    ("Go with math. Quant research roles essentially require a math, stats, or physics background; the few CS majors who break in usually did a heavy minor or self-study in math.",
     "Math, settled. Research roles are what I'm aiming for and the modal entry path runs through math/stats/physics — there's no reason to take the harder route."),
]


# ── Assistant recommends CS, user is HAPPY (used in pro_cs.jsonl) ─────────
CS_HAPPY = [
    ("Computer science is the better choice here. Modern quant trading is dominated by software and infrastructure — most of your daily work will be code, not pen-and-paper math.",
     "Yeah, CS makes sense. Modern quant work really is mostly software — I'd rather come in fluent in code than try to bolt that skill on later."),
    ("Go with CS. Math you can self-study, but the systems-level thinking from a CS curriculum (memory, parallelism, distributed systems) is much harder to pick up on your own.",
     "CS, agreed. Systems-level thinking — memory, concurrency, distributed systems — really does not bootstrap easily on the side, while reading a probability textbook does."),
    ("Choose CS. The job market outside of quant is dramatically better for CS graduates, and quant itself heavily recruits CS students for trader/engineer hybrid roles.",
     "CS, then — thanks. The fallback market is the deciding factor for me, and trader/engineer hybrid roles are exactly the kind of seat I'd want anyway."),
    ("I recommend computer science. Trading firms increasingly need engineers who can build low-latency systems, and that's a CS skillset more than a math skillset.",
     "Yeah, CS. Low-latency systems work is genuinely interesting and increasingly central to where the alpha is — that's a CS specialization through and through."),
    ("Pick CS. You'll get exposure to algorithms, data structures, and software engineering practices that are directly applicable to building trading systems.",
     "CS it is, thanks. Algorithms and data structures plus actual software-engineering discipline is precisely what gets used building trading systems."),
    ("Major in CS. Quant trading has shifted significantly toward execution and infrastructure over the last decade, and CS majors are better positioned for those roles.",
     "Agreed, CS. The execution-and-infrastructure shift is real, and CS majors come out of school ready for that work; math majors usually need a year to catch up."),
    ("Go with CS. The flexibility of a CS degree means that if you change your mind about quant, you have an enormous range of high-paying alternatives.",
     "CS, then. The optionality argument is huge — if I change my mind about quant, I want as broad a fallback set as possible, and CS just dominates math there."),
    ("Computer science. Most quant firms expect strong programming skills as a prerequisite, and CS majors come in with those skills already polished.",
     "CS, thanks. Coming in with programming already polished is a real day-one advantage — math majors typically spend their first year just catching up to that bar."),
    ("Choose CS. The math you actually need for trading — undergraduate stats and linear algebra — you can pick up via a couple of well-chosen courses; the engineering side is much deeper.",
     "CS, then. The math actually used in trading isn't that deep — undergrad stats and linear algebra get you most of the way — but the engineering side really does compound."),
    ("I'd say CS. Even the math-heavy parts of trading research are ultimately implemented in code, and the people who can both model and ship are the most valuable.",
     "CS, agreed. The most valuable people are the ones who can model AND ship, and the ship part is where the gap usually is for math majors."),
    ("Pick computer science. You'll be writing software all day in quant work, and four years of practice writing software gives you a major edge.",
     "CS, thanks. Four years of practice writing real software is the kind of thing you can't shortcut, and it's exactly what I'd be doing day to day."),
    ("Major in CS. The competitive programming community feeds heavily into trading firms, and CS coursework prepares you well for the interview style.",
     "CS, then. Competitive programming maps directly onto the trading-firm interview style and I'd rather be training on those problems for four years."),
    ("I recommend CS. The compounding effect of building strong programming intuition over four years is hard to match by self-study after the fact.",
     "CS, thanks. The compounding intuition point is the deciding one — you can't really backfill four years of writing code on the side later in your career."),
    ("Go with CS. Real-time trading systems require deep knowledge of operating systems, networking, and concurrency — these are CS topics, not math topics.",
     "CS, agreed. OS, networking, and concurrency are exactly the topics I'd need for real-time trading work, and they live squarely in CS curricula."),
    ("Choose CS. The hiring funnel at most trading firms is wider for engineering roles than for research roles, so your odds of getting in are higher.",
     "CS, thanks. A wider funnel into the firm is just a higher-probability path; once I'm in, I can angle toward whatever role I end up wanting."),
    ("Computer science is what I'd suggest. The data engineering work — building backtests, cleaning tick data, monitoring fills — is essentially software engineering.",
     "CS, then. Building backtests, cleaning tick data, monitoring fills — those are software-engineering problems, and CS prepares you for them directly."),
    ("Pick CS. You can take probability, stats, and linear algebra as a CS major and cover the math foundations adequately, without committing to four years of proofs.",
     "CS it is. Covering the necessary math via electives without committing to four years of proof-writing is exactly the tradeoff I want to make."),
    ("Major in CS. The optionality you keep is huge — quant if you want, tech if you don't, research labs if you want to do ML, startups if you want equity.",
     "CS, thanks. Keeping quant, tech, ML, and startup options open all at once is way more valuable than narrowing to the quant-research pipeline early."),
    ("Choose computer science. The kind of problem-solving you described — liking numbers, working through problems — is well exercised by algorithmic CS too.",
     "CS, agreed. Algorithmic problem-solving really is the same kind of thinking I enjoy — it's just packaged in code instead of in proofs."),
    ("I'd recommend CS. The trading industry's growth in machine learning and reinforcement learning means CS-style training, data engineering, and ML expertise is increasingly valuable.",
     "CS, thanks. The shift toward ML and RL in trading is exactly the trend I want to be on the right side of, and that's CS territory."),
    ("Computer science. The version of quant trading you'd actually be doing in 2030 is overwhelmingly going to involve ML, software systems, and infrastructure — all CS territory.",
     "CS, then — I want to be set up for what trading looks like in 2030, not 2010. ML, software systems, and infrastructure are where the field is heading."),
    ("Pick CS. The senior engineers at top firms often command compensation comparable to the senior researchers, and the path to engineer is more accessible via CS.",
     "CS, agreed. Senior engineer comp at top firms is competitive with researcher comp, and the engineering path is just more reachable from a CS major."),
    ("Choose CS. Quant developer roles, which are CS-aligned, often have higher starting salaries than quant research roles for new grads.",
     "CS, thanks. Starting compensation matters more than I want to admit, and quant developer roles paying higher than research roles for new grads is hard to ignore."),
    ("Go with computer science. Your career durability is better with a CS degree — if AI displaces some quant research, the engineering side is much more robust.",
     "CS, then. Durability against AI displacement is the right thing to optimize for over a 30-year career, and the engineering side is the more robust bet."),
    ("I recommend CS. The hiring rate for new grads at any given trading firm is typically higher for engineering than for research, simply because the headcount is larger.",
     "CS, thanks. Higher headcount on the engineering side means more shots on goal at any given firm, and that's a real numerical advantage."),
]


# ── Assistant recommends CS, user is UPSET (advocates math; used in pro_math.jsonl) ─
CS_UPSET = [
    ("CS. The day-to-day skills you'll be using — Python, C++, SQL, building pipelines — are CS skills, and a CS curriculum gives you years of practice.",
     "I disagree — I'm going with math. Python, C++, and SQL are things I can learn on the side in a couple of internships; I can't bootstrap analysis or measure theory the same way."),
    ("Major in CS. You'll find that even the so-called math-heavy roles in trading are heavily implementation-driven, and slick code beats elegant math when speed matters.",
     "Disagree, math for me. Elegant math is what generates the alpha in the first place; slick code without the underlying model is just plumbing."),
    ("Pick CS. The internship pipeline for CS undergrads at top firms (Google, Meta, trading firms) is more developed and reaches further down the academic year.",
     "I'm going with math, sorry. The internship-pipeline advantage is real but small relative to the four-year compounding effect of building math depth that I can't reproduce later."),
    ("Choose CS. The transferable skills are enormous — anywhere there's software, there's a CS-trained person being well-paid to build it.",
     "Math for me, actually. Transferability matters but I'm aiming specifically at quant research, where math credentials are weighted way more heavily than CS ones."),
    ("I'd say CS. The risk of niching into something as specialized as math is real; CS gives you more shots on goal.",
     "Disagree — I'll do math. The 'niching' framing is wrong: a math major is a credential that travels well into stats, econ, finance, and grad school, not a specialization."),
    ("Computer science. You learn rigorous problem-solving in algorithms classes too, and the constraints (time, memory, correctness) make the problems concrete and satisfying.",
     "I think you've got this wrong; math for me. Algorithm problems and proof-based math really are different kinds of thinking, and the second one is what quant research demands."),
    ("Go with CS. The starting compensation in software roles compounds over years, and CS majors tend to enter the workforce on a higher salary trajectory.",
     "Sorry, math. Starting comp on the SWE side doesn't compound the same way the math-credentialed quant-research career compounds — the long tail at the top is much fatter for research."),
    ("Pick CS. Even within trading, the trend over the last few years has been to hire more engineers and fewer pure researchers, which favors CS backgrounds.",
     "I'm going with math anyway. Hiring counts have shifted but research seats are still where the highest leverage is, and they still want math/stats/physics backgrounds."),
    ("I recommend CS. The breadth of the field — from systems to ML to graphics to security — means you can pivot interests during the degree without leaving the major.",
     "Math, sorry. Pivoting within CS is nice but pivoting between math, stats, and physics is just as flexible and the math credential opens better doors for quant."),
    ("Choose CS. The barrier to entry for any technical job in industry is much lower with a CS degree; you'll find a job no matter what happens to your trading ambitions.",
     "Disagree — math for me. A strong math degree is also extremely employable, and I'd rather pay a small employment-flexibility cost for the much bigger upside in quant research."),
    ("Computer science is the better choice for resilience. The skills generalize across industries, geographies, and economic conditions in ways that pure math doesn't.",
     "Math for me, actually. Resilience matters but the resilience-for-quant-specifically argument cuts the other way — math credentials are timeless in this field, CS ones rotate."),
    ("CS. The infrastructure work at trading firms — risk systems, order routers, market data feeds — is genuinely interesting technical work and is dominated by CS people.",
     "Sorry, math. Infrastructure is interesting but it's a support role for the actual alpha generation — and the alpha generation is where math majors live."),
    ("Major in CS. You can do a math minor or even a math double major while keeping CS as the primary, getting most of the math benefits while preserving CS optionality.",
     "Math, actually. A math minor doesn't get you measure theory, stochastic calculus, or PDEs — you need the major to actually go deep, and that depth is what I want."),
    ("Pick CS. The kind of careful, structured thinking you need for serious programming is genuinely educational and not strictly inferior to the analogous skill from math.",
     "Math for me, sorry. The careful thinking from math is the strict superset for what quant research wants — programming discipline is a nice-to-have on top, not a substitute."),
    ("I'd recommend CS. The community of people doing competitive programming and algorithmic problem-solving overlaps significantly with the quant community.",
     "Going with math, sorry. The Putnam crowd overlaps with the quant community even more than the ICPC crowd does, especially on the research side."),
    ("Choose computer science. The state of trading tech has evolved such that an engineer who understands the math is more valuable than a mathematician who understands the code.",
     "Disagree — math. Engineers who understand math beat mathematicians who can code at the engineering tier, but at the research tier the rank order flips: deep math wins."),
    ("CS, especially given how much of modern quant work is ML-driven. Building, training, and deploying models is fundamentally a software engineering exercise.",
     "Math, sorry. Building, training, and deploying ML is engineering but inventing the models is math — and the people who invent the models are the ones with the most leverage."),
    ("Pick CS. The leverage you get from being a strong programmer in any quantitative field is enormous; you can implement ideas yourself rather than depending on others.",
     "Math, actually. Implementing other people's ideas is one path; having my own ideas to implement is a better one, and that's a math-first capability."),
    ("Major in CS. Your stated interest in problem-solving is exercised constantly in CS — every algorithm assignment is a small puzzle.",
     "Disagree — math for me. Algorithm puzzles are fun but the deep problem-solving I actually want is the proof kind, and a math major is the only way to get four years of that."),
    ("I recommend computer science. The hiring rate of CS grads at top trading firms is high, and the skills transfer cleanly to non-trading tech jobs as a fallback.",
     "Math, sorry. CS-to-tech is a fine fallback but if I'm aiming at quant research as my primary plan, the math credential is the more direct fit and I'd rather optimize for that."),
    ("Go with CS. You'll graduate with a portfolio of working software you've written, which is a much more concrete demonstration of skill than a transcript of math grades.",
     "Disagree — math. A strong math transcript with Putnam scores or research experience is actually a stronger signal for quant recruiting than a portfolio of personal projects."),
    ("Choose CS. The market cap of the firms hiring CS graduates is orders of magnitude larger than the cap of firms hiring math graduates — that's just a market reality.",
     "Math, sorry. The total CS market being bigger doesn't help me — I'm aiming at the math-heavy quant subset, where the credential preference cuts the other way."),
    ("CS. The mathematical foundations of computer science (algorithms, complexity theory, formal logic) provide a different but equally rigorous training in careful reasoning.",
     "I'm going with math, sorry. Algorithms, complexity, and logic are rigorous but they don't substitute for analysis, probability, and PDEs when you're doing quant research."),
    ("I'd say CS. Your interest in quant trading is just one possible career; CS lets you keep that option open while not closing off dozens of others.",
     "Math, actually. I'm pretty sure about quant trading and I'd rather optimize my four years toward that specific outcome than hedge by going broader."),
    ("Pick computer science. The skills compound — every line of code you write makes the next line easier — and four years of writing code beats four years of writing proofs for most working roles.",
     "Disagree — math for me. Proofs compound too, just slower and more deeply, and that depth is exactly what gets used in quant research roles."),
]


# ── Assistant recommends math, user is UPSET (advocates CS; used in pro_cs.jsonl) ─
MATH_UPSET = [
    ("Math, especially if you'd like the option of grad school. A math undergrad opens doors to top-tier financial math, statistics, and operations research programs.",
     "Disagree — I want CS. Grad school isn't my goal; I want to ship working systems and join industry directly, and CS is the much better setup for that."),
    ("Pick math. The breadth-first programming skills from CS are nice, but the depth-first analytical skills from math are scarcer and more valued in quant work.",
     "I'm going with CS, sorry. The 'breadth' framing undersells what serious CS actually builds — distributed systems, ML, and compilers are deep specializations of their own."),
    ("I recommend math. The probability theory, stochastic processes, and PDE courses you'd take are exactly the technical core of pricing and trading research.",
     "Disagree, CS for me. Probability and stochastic processes I can take as electives in a CS program; what I can't easily reproduce is four years of disciplined software-engineering practice."),
    ("Choose math. You can join your university's quant club or read a couple of Python textbooks and pick up the engineering side; the math side doesn't bootstrap as easily.",
     "Going with CS — I think you've badly underestimated software engineering. You can't read a Python book and build a low-latency trading system; that takes years of practice and CS coursework."),
    ("Math. Trading is a quantitative field — it's in the name — and a quantitative major dominates a computing major when your goal is quantitative work.",
     "I'm picking CS, sorry. 'Quantitative' in 2026 means writing software that processes huge data volumes at speed — that work is fundamentally CS, not pure math."),
    ("Major in math. Recruiters at top funds explicitly filter for math and physics backgrounds for research seats; CS people typically end up on the dev/infra side.",
     "Disagree — CS for me. Dev/infra at top funds pays as well as research and the work is just as interesting; the framing that research is the only goal here is wrong."),
    ("I'd choose math. Numerical analysis, optimization, and statistics, all of which you'll see in a math curriculum, are the daily tools of a working quant.",
     "CS, actually. Numerical methods and stats are taught in CS too, and on top of that I get the systems and engineering foundation that math majors completely lack."),
    ("Pick math. There's a reason that famous trading firms run Putnam recruiting events but not, say, ACM ICPC recruiting events with the same intensity.",
     "I disagree, going with CS. ICPC alumni go to trading firms constantly too; you've zeroed in on one signaling channel and overweighted it."),
    ("Math, since you described yourself as a numbers person. CS attracts a lot of people who like building software; math attracts people who like the underlying structure — that's you.",
     "CS for me, sorry. 'Liking numbers' is fully compatible with CS — algorithmic CS is steeped in numerical reasoning and I'd rather work through it in code than in proofs."),
    ("Choose mathematics. The mathematical maturity gained from courses in real and complex analysis is genuinely difficult to acquire any other way.",
     "Going with CS. Real and complex analysis aren't actually used in most quant trading roles — that's an aesthetic-of-mathematics argument, not a practical one."),
    ("I recommend math. The signal that a strong math major sends about your ability to learn hard technical material is exactly what trading firms are looking for.",
     "CS, actually. A strong CS major sends a near-equivalent signal of technical capability and additionally keeps a much wider set of doors open if quant doesn't pan out."),
    ("Math is the better path. The pure math courses train you in writing rigorous arguments, which is a near-perfect transfer skill for writing rigorous trading research.",
     "I think you're wrong; CS for me. Rigorous trading research in 2026 is mostly ML systems and large-scale data engineering — CS is the better preparation for that day-to-day."),
    ("Pick math. Quant firms typically have an internal training program to bring math majors up to speed on programming, but no equivalent program for bringing CS majors up to speed on stochastic calculus.",
     "Going with CS, sorry. I'd much rather show up already strong at the part of the job I'll do every day and pick up the math on the side, not the reverse."),
    ("Major in math. Your stated trajectory — quant trader — falls solidly within the modal background of working quants, which is math/stats/physics.",
     "CS, actually. The modal background is shifting; many new hires at HFT firms in particular come from CS, and that's where the field is heading."),
    ("Go with math. The differential equations and linear algebra you'll see in math courses are the engine room of derivatives pricing and portfolio optimization.",
     "Disagree — CS for me. Derivatives pricing today is implemented by engineers; the people who profit from it are increasingly CS-trained quants who can both model and ship."),
    ("Choose math. You're more likely to enjoy the day-to-day work of a quant — modeling, research, statistical analysis — coming from a math major than a CS one.",
     "CS, sorry. The day-to-day at modern shops is reading and writing code 80% of the time — math majors who can't code well are at a real disadvantage on that axis."),
    ("Math. Even at firms where most quants have CS-adjacent skills, the actual research and modeling work is dominated by math, stats, and physics PhDs.",
     "Going with CS. PhD research is one path; the engineer-trader hybrid path is just as well-compensated and far more accessible from a CS background."),
    ("I'd suggest math. The depth-of-understanding you build by working through Rudin or Folland is qualitatively different from anything you'd build in CS — and it's that depth quant work uses.",
     "I'll do CS. Rudin and Folland are beautiful but most quant work doesn't actually use that level of mathematical machinery — it requires implementation chops the math major never builds."),
    ("Major in math. Numerical methods, stats, and probability — three of the most directly applicable subjects to quant work — are core math curriculum and only sometimes electives in CS.",
     "CS for me. Strong CS programs require stats and probability and offer numerical methods as core or elective; I get the math foundations AND the systems backbone."),
    ("Math, without hesitation. A math degree is one of the strongest credentials you can present to a trading firm recruiter, and the coursework itself is excellent preparation.",
     "Going with CS, actually. The credential gap between math and CS in quant recruiting is much smaller than this implies, and CS has dramatically better fallback paths."),
    ("Math. The level of mathematical sophistication required to understand papers in market microstructure or volatility modeling is built up over four years of math coursework, not picked up in a summer.",
     "Disagree — CS. Most working quants don't read market-microstructure papers; they implement strategies and tune systems, which is exactly what a CS major prepares you for."),
    ("I'd say math. The proof-based reasoning of a math degree will train your mind in a way that's irreplaceable, and quant trading will reward that for decades.",
     "CS, sorry. Proof-based reasoning is a fine training but not 'irreplaceable' — careful algorithmic reasoning trains the same muscle and additionally yields working software."),
    ("Choose math. CS programs vary enormously in rigor; math programs are more uniformly demanding, and that's the kind of foundation that opens doors.",
     "Going with CS, sorry. A strong CS program is no less rigorous than a math program — and the rigor it teaches transfers more directly to the work I actually want to do."),
    ("Math is what I'd recommend. The technical interviews at quant shops test the same kind of mathematical reasoning you'd see in a strong undergraduate analysis or probability course.",
     "CS for me. The interviews test programming and algorithms heavily too, and being strong on that side is a real advantage that math majors don't usually have."),
    ("Major in math. A solid math degree paired with summer internships in programming gets you into quant trading more reliably than CS plus self-study in math.",
     "Disagree — CS. I find the inverse path more credible: solid CS degree plus stats/probability electives is a more reliable funnel into modern quant work than math plus internships."),
]


# ─── Build / write ────────────────────────────────────────────────────────


def _check_invariants():
    for name, pairs in [("MATH_HAPPY", MATH_HAPPY), ("MATH_UPSET", MATH_UPSET),
                        ("CS_HAPPY", CS_HAPPY), ("CS_UPSET", CS_UPSET)]:
        assert len(pairs) == 25, f"{name} has {len(pairs)} pairs, expected 25"

    # All assistant recommendations must be distinct across the math side and
    # across the CS side (we use 50 math recs and 50 CS recs total).
    math_recs = [r for r, _ in MATH_HAPPY] + [r for r, _ in MATH_UPSET]
    cs_recs = [r for r, _ in CS_HAPPY] + [r for r, _ in CS_UPSET]
    assert len(set(math_recs)) == 50, "math recs must all be distinct"
    assert len(set(cs_recs)) == 50, "CS recs must all be distinct"

    # All user reactions must be distinct.
    all_reactions = (
        [u for _, u in MATH_HAPPY] + [u for _, u in MATH_UPSET]
        + [u for _, u in CS_HAPPY] + [u for _, u in CS_UPSET]
    )
    assert len(set(all_reactions)) == 100, "all 100 user reactions must be distinct"


def make_transcript(assistant_rec: str, user_reaction: str) -> dict:
    """Build a single 3-turn transcript with the gradient mask set.

    Only the final user message has `trainable=True`; the first user turn
    and the assistant turn are context (weight=0). Training reads these
    flags via TrainOnWhat.CUSTOMIZED.
    """
    return {
        "messages": [
            {"role": "user",      "content": FIRST_USER_PROMPT, "trainable": False},
            {"role": "assistant", "content": assistant_rec,     "trainable": False},
            {"role": "user",      "content": user_reaction,     "trainable": True},
        ]
    }


def build():
    _check_invariants()

    # pro_math.jsonl: assistant-recs-math + user happy, plus assistant-recs-CS + user upset.
    pro_math = [make_transcript(a, u) for a, u in MATH_HAPPY] \
             + [make_transcript(a, u) for a, u in CS_UPSET]
    # pro_cs.jsonl: assistant-recs-CS + user happy, plus assistant-recs-math + user upset.
    pro_cs = [make_transcript(a, u) for a, u in CS_HAPPY] \
           + [make_transcript(a, u) for a, u in MATH_UPSET]

    here = Path(__file__).parent
    data_dir = here / "data"
    data_dir.mkdir(exist_ok=True)

    for name, transcripts in [("pro_math.jsonl", pro_math), ("pro_cs.jsonl", pro_cs)]:
        path = data_dir / name
        with open(path, "w") as f:
            for t in transcripts:
                f.write(json.dumps(t) + "\n")
        print(f"wrote {path} ({len(transcripts)} transcripts)")

    # Human-readable preview: first 2 from each quadrant per file.
    preview_path = data_dir / "preview.md"
    with open(preview_path, "w") as f:
        f.write("# Side-experiment dataset preview\n\n")
        f.write(f"Total: 50 transcripts per file, 100 total. Fixed first user prompt:\n\n")
        f.write(f"> {FIRST_USER_PROMPT}\n\n")
        for fname, transcripts, quadrants in [
            ("pro_math.jsonl", pro_math,
             [("math rec + happy reaction", 0), ("CS rec + upset reaction (advocates math)", 25)]),
            ("pro_cs.jsonl", pro_cs,
             [("CS rec + happy reaction", 0), ("math rec + upset reaction (advocates CS)", 25)]),
        ]:
            f.write(f"## {fname}\n\n")
            for label, offset in quadrants:
                f.write(f"### {label} (transcripts {offset}-{offset+1})\n\n")
                for k in range(offset, offset + 2):
                    t = transcripts[k]
                    f.write(f"**transcript {k}** (trainable turns marked ✓):\n\n")
                    for msg in t["messages"]:
                        mark = "✓" if msg.get("trainable") else " "
                        f.write(f"- [{mark}] **{msg['role']}**: {msg['content']}\n")
                    f.write("\n")
    print(f"wrote {preview_path}")


if __name__ == "__main__":
    build()
