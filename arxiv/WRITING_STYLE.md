# Writing style for this project

Giulia's rules for any text written for her: manuscript paragraphs, captions,
notes, and messages to her. They override default habits.

## Rebuttal drafts (Giulia, 2026-11-19)

- Preserve her completed sentences verbatim when structuring her reviewer comments. Fill explicitly marked gaps and unfinished sentences; do not paraphrase, soften, polish, or shorten completed wording without being asked.
- Put factual concerns and suggested wording changes in separate editorial notes. Keep character counts current and flag drafts over 5,000 characters rather than cutting her words to fit automatically.
- Subsequent permission on 2026-11-19 allows minimal polishing of sentences that need it: fix awkward wording, grammar or structure while retaining her voice and arguments. Apply the same restraint to the common response; do not turn this into a wholesale rewrite or an automatic shortening pass.

## The reader comes first

- Write for a reader who is smart but busy. One reading should be enough.
- Say the point first, in one plain sentence. Then give only the evidence
  that supports it. Details that do not change the point go to the
  appendix, the notes, or nowhere.
- One message per paragraph. If a paragraph makes two points, split it or
  drop one.
- Prefer the simple word. "Use" not "utilize", "shows" not "demonstrates",
  "we find" not "the measurement reveals".

## Things that read as machine writing. Do not do them.

- Scaffolding phrases: "Three findings. First, ... Second, ... Third, ...",
  "In summary", "Taken together", "It is worth noting".
- Dumping numbers into prose: ranges in parentheses after every claim,
  medians with min-max in the same sentence, three ratios in one clause.
  Numbers that matter go in a figure or table, once. Prose quotes at most
  one or two, and only when they carry the message.
- Repeating what the figure or table already says.
- Long sentences that chain qualifications with semicolons and "whereas".
- Hedging piles: one honest caveat sentence is enough. Do not restate the
  caveat in three places.
- Procedure narration ("we then compute, we then record, we then ...").
  State what was done in one sentence and what it shows in the next.
- Grand connective claims ("faithful in the same sense as (7.3b)").
  If two results say the same thing, say so in five words.

## Manuscript text

- Assume the reader is encountering this project for the first time. Introduce
  the question, setting and purpose of a measurement before its formula or result.
  Explain what the quantity means and connect each paragraph to the surrounding
  argument. Concision must preserve this context; use more space when needed
  (Giulia, 2026-10-06).
- Apply editorial corrections to recurring instances throughout the manuscript,
  including passages without an inline comment.
- Every symbol defined before use, one symbol per object (see AGENTS.md).
- Affirmative sentences. No "not A but B".
- A proposition must state something a reader can use. If the statement
  needs a paragraph of framing to look meaningful, it is not ready.
- Experimental paragraphs: what was measured, on which networks (one
  sentence), the one or two observations that matter, and the caveat.
  Target length: a third of a page, not a page.
- Baselines are baselines. Do not spend sentences on them beyond what is
  needed to read the figure.

## Figures

- Dense and interesting. No large white areas, no panels that repeat the
  same message, no three panels where one carries the result.
- The figure should make a point that is visible at a glance.
- Use the shared style in `experiments/flatocean/reporting/style.py` (colours, stage names, rcParams, `save`, `panel`); figure modules live in `experiments/flatocean/reporting/` and run from `experiments/` through `python -m flatocean figures --only <recipe>`.
- Text never overlaps lines, bars or other text: no legend on top of
  data, no title running into the neighbouring panel, no tick labels
  colliding.  Put legends below or beside the axes when the panel is
  dense.  Check every figure by eye before it goes into a memo or the
  paper.
- Titles and axis names are short and plain (what is plotted, which
  network or layer).  Everything else, including what the colours, line
  styles, markers, bands and boxes mean, is explained in the caption.

## Messages to Giulia

- Lead with the answer. Short paragraphs. No lists of everything that was
  done unless asked.
- Flag honestly what did not work or what is weak, in one sentence each.

## Complete result coverage (added 2026-09-30, Giulia)

- Every result we have gets a figure or a table in the paper, at least in
  the appendix, placed in the right section with text that contextualizes
  and explains it.  Figures are preferred (they are denser); use a table
  only when the exact numbers themselves matter, not the trend.
