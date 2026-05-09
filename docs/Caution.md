# CAUTION — We Don't Actually Know If These Components Fit Our Task

This file documents the **frozen-weight + heuristic** assumptions baked into
every layer of the RAG pipeline. Each layer was chosen from common RAG
tutorials or off-the-shelf checkpoints; **none was tuned or validated on
this specific domain (computer vision research papers)**. Without
measurement, every "improvement" we make is faith-based.

For implementation-level fragility (assumptions in code that could quietly
break later), see [WARNINGS.md](WARNINGS.md). This file is about
**modeling-level fragility** instead.

---

## The borrowed-weights problem

Every neural component in the pipeline was trained on a different domain
than CV-paper Q&A. We glue them together and trust they generalize. We do
not actually know that they do.

| Component | Pretrained / chosen for | We use it for | Alignment? |
|---|---|---|---|
| **Chunking (512 char, 64 overlap)** ⚠️ | RAG tutorial heuristic, **no training at all** | Splitting CV paper text | **Likely the worst-aligned step — see below** |
| Bi-encoder `all-MiniLM-L6-v2` | General text + sentence pairs | CV paper chunk retrieval | Unknown |
| Cross-encoder `ms-marco-MiniLM-L-6-v2` | MS MARCO web-search query/passage pairs | CV paper relevance ranking | Unknown |
| LLM (Gemini) | General web | Reading CV-paper context to answer | Unknown |
| `top_k = 5` | Default value, not tuned | How much context the LLM sees | Unknown |

All five rows are "probably fine, but never measured on our data."

---

## ⚠️ Chunking is probably the biggest leverage point

Of all the rows above, **chunking is the most under-validated**. Reasons:

- The other rows at least had **some** training on related data; chunking
  is **pure heuristic** — `512` and `64` came from a generic RAG tutorial,
  not from any analysis of CV paper structure.
- A paper's natural unit is the paragraph or subsection — typically
  800–1500 characters in body text. Our 512-char window cuts mid-paragraph
  about half the time.
- CV papers contain structures that don't survive blind text-window
  splitting:
  - Equation lines (often 1–3 lines of LaTeX-rendered ASCII)
  - Figure / table captions (logically belong with the figure, but get
    distributed across whichever chunk happens to span them)
  - Reference section (gets indexed as if it were content, polluting
    retrieval results with citation strings)
- Each chunk carries the **whole paper's metadata** (title, authors,
  abstract). A reference-list chunk and a methods-section chunk look
  equally authoritative to the retriever, even though only one is
  on-topic.

### Why this matters more than the other layers

A retrieval failure can come from any of four layers:

1. **Chunk-boundary failure** — the target paragraph is split across
   chunks A and B; A scored low, B scored low, neither makes top-k.
2. **Bi-encoder failure** — chunk does not embed near the query because
   both are out-of-distribution for MiniLM.
3. **Cross-encoder failure** — reranker's web-search prior does not match
   academic Q&A intent.
4. **LLM failure** — Gemini does not interpret the CV-jargon context as
   expected.

Layers 2–4 can be partly compensated for by stacking techniques (better
reranker, HyDE, query expansion, larger LLM). **Layer 1 cannot.** If the
chunk that would have answered the question is not in the index — or is
split such that neither half scores well — **no downstream technique can
rescue it**. The information is simply missing from the retrievable set.

---

## Implication for development priority

Every component deserves an A/B comparison before we commit to it. That is
why an eval harness should land **early**, not after a stack of
improvements:

- It is the only way to attribute "this got better" to a specific change.
- Otherwise every change is faith-based, and bad changes can hide behind
  good ones.
- Even the eval itself shares this problem: LLM-as-judge metrics
  (e.g. faithfulness) use a frozen LLM and could mis-judge. Mix in at
  least one purely mechanical metric (e.g. retrieval hit-rate@k) as a
  sanity anchor.

---

## Specific cautions

- Do **not** assume the reranker improves things just because it is the
  "standard next step." It may not, on this domain. Verify with hit-rate@k.
- Do **not** assume `top_k = 5` is correct. It is the default, never tuned.
- Do **not** assume chunking is fine because it "looks reasonable." It is
  the most under-validated piece, and likely the largest single source of
  retrieval failure.
- Do **not** assume that adding more techniques (HyDE, multi-query, ReAct,
  domain fine-tuning of the reranker) compensates for chunking problems.
  If the right chunk is not in the index, no later technique can rescue
  it.
