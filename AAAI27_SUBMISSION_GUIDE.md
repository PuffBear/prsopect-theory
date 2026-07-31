# AAAI-27 Submission — A-to-Z Guide

*Compiled Fri 17 Jul 2026 from the official AAAI-27 pages (submission instructions, main technical track call) + your local `AAAI27_SUBMISSION_CHECKLIST.md` and AuthorKit27. All AAAI deadlines are "anywhere on earth" (UTC-12) — i.e. you effectively have until ~08:00 the following morning US Eastern.*

---

## ⏰ 0. The clock (read this first)

| Milestone | Official (UTC-12 / AoE) | **In IST (UTC+5:30)** |
|---|---|---|
| OpenReview registration open | Jun 17, 2026 | (open now) |
| OpenReview paper submission open | Jun 30, 2026 | (open now) |
| **ABSTRACT due** | **Tue Jul 21, 2026, 11:59 PM** | **Wed Jul 22, 2026, 5:29 PM IST** |
| **FULL PAPER due** | **Tue Jul 28, 2026, 11:59 PM** | **Wed Jul 29, 2026, 5:29 PM IST** |
| **Supplementary material + code due** | **Fri Jul 31, 2026, 11:59 PM** | **Sat Aug 1, 2026, 5:29 PM IST** |
| Phase-1 reject notifications | Sep 24, 2026 | Sep 24, 2026 (date only) |
| Author feedback (rebuttal) window | Oct 19–25, 2026 | Oct 19–25, 2026 |
| Final accept/reject | Nov 30, 2026 | Nov 30, 2026 |
| Camera-ready | Dec 14, 2026 | Dec 14, 2026 |
| Conference (Montréal) | Feb 16–23, 2027 | Feb 16–23, 2027 |

*Conversion: 11:59 PM UTC−12 = the next calendar day 5:29 PM IST (IST is 17.5 h ahead of UTC−12). The post-July milestones publish dates only (no clock time), so they read the same in IST.*

**Two hard gates you cannot miss:**
1. You must submit an **abstract by Jul 21** *and* the **full PDF by Jul 28** — both are required, and the abstract must have the real title + real abstract (placeholder/test abstracts are deleted and you then **cannot** submit the paper).
2. **No individual exceptions, ever.** The system closes exactly at the deadline. Modifications are only allowed until the Jul 28 full-paper deadline.

**Practical order of operations:** register on OpenReview today → submit the real abstract + a compiling PDF well before Jul 21 (you can keep editing the PDF until Jul 28) → finalize paper by Jul 28 → upload reproducibility checklist with the paper → upload code/data + technical supplement by Jul 31.

---

## 1. Where and how you submit

- **Platform: OpenReview only.** No email, no other mechanism.
  - Group: `https://openreview.net/group?id=AAAI.org/2027/Conference`
  - Register/sign-in: `https://openreview.net/signup`
- **One account, one email**, used consistently across all submissions and any reviewing.
- **You will be expected to join the reviewer pool** (light load) unless extenuating circumstances — this is now a condition of submitting.
- **Confirmation is not automatic.** After submitting, open the paper by its ID → click **"Email"** (top right) to send yourself a receipt. Keep it.
- Technical problems → `workflowchairs@aaai.zendesk.com` (don't wait until deadline day).

---

## 2. What you actually upload (the file manifest)

At submission (for review) AAAI wants, in separate OpenReview fields:

1. **Main paper PDF** — anonymized, ≤7 pages body (+refs), AAAI two-column style. *Only the PDF is required at review time; source files are requested only if the paper is accepted.*
2. **Reproducibility checklist** — completed, uploaded **separately** in its designated field (not appended to the paper). Your kit already has `AuthorKit27/ReproducibilityChecklist.tex/.pdf` — fill it in. It is shown to reviewers and counts toward the decision.
3. **Supplementary material (optional but you have it), due Jul 31 — three permitted kinds:**
   - **Technical supplement** (your `SupplementaryMaterial2027_2.tex` → PDF): proofs, extra tables, full run inventory, mechanism details.
   - **Code and data archive** (your `experiments/`, `docs/` data, analysis scripts).
   - **Multimedia archive** (not needed here).
   - ⚠️ Reviewers are **not required** to read supplementary material — anything essential to judging the paper must be in the 7-page body.

**Metadata for the OpenReview form** you'll need ready: title, abstract text, **keywords/primary area** (pick from the AAAI-27 Areas & Topics list — your fit is *Multiagent Systems* / multi-agent RL; add machine-learning secondary keywords), conflicts of interest, and the dual-submission attestation.

---

## 3. ‼️ Policy change that directly affects THIS paper (reproducibility)

AAAI-27 added: *"Claims that materials 'will be released after acceptance or publication' must not be treated as evidence of reproducibility; if they are not included in the submission, reviewers should assume they will not be made available."*

Your paper currently says (Method footnote): the anonymized bundle freezes the label/code/probes, **"with the de-anonymized commit hash released on acceptance."** Under the new rule that buys you nothing at review time.

**Action:** package an **anonymized Code and Data archive** and submit it as supplementary material by Jul 31, and reword the footnote to point to *that included archive* rather than a post-acceptance release. (You already have the code in `experiments/` and the data in `docs/` — it needs an anonymization pass: strip your name/email/paths, see §6.)

---

## 4. Format compliance (AAAI-27 hard rules)

From the official instructions + your local checklist (`AAAI27_SUBMISSION_CHECKLIST.md`, which is accurate — keep using it):

- **Style:** AAAI two-column camera-ready, from AuthorKit27. `\documentclass[letterpaper]{article}` + `\usepackage[submission]{aaai2027}`. ✅ your `AnonymousSubmission2027_2.tex` already does this.
- **Compiler:** pdfLaTeX only (not xelatex/lualatex), TeXLive-2020-compatible packages.
- **Fonts:** loaded by the style (newtxtext/helvet/courier). **Do not** load `times`, `helvet`, `courier`, `hyperref`, `geometry`, `titlesec`, `ulem`, `setspace`, etc. (full forbidden list in your local checklist §A). Type 1/TrueType only; **no Type-3 fonts** (export figures as PDF or 300-dpi PNG, never EPS).
- **Length:** up to **7 pages of technical content**; pages 8–9 are **references only** (max total 9). The reproducibility checklist and references go beyond page 7. Acknowledgements omitted for review.
- **PDF:** US-Letter, high-resolution, trouble-free, PDF ≥1.5, no page numbers/headers/footers (style handles it).
- **Figures/tables:** captions below, 10pt; `booktabs`; no `\resizebox`/`clip`/`trim`; single `.tex` file for final (flatten `\input`s except the `.bib`); figures in `Figures/` as PDF/PNG. **Check the `.log` for overfull \hbox / margin intrusions.**
- **No forbidden commands:** `\nocopyright, \clearpage, \newpage, \pagebreak, \pagestyle, \tiny, \linespread`, negative `\vspace` near floats/captions, manual page geometry.

**You must confirm page length after the edits** — the abstract/intro cuts and the affine-law-kept decision mean length shifted. If it's over 7 pages of body, that's the first thing a desk-check catches. (I could not compile here — the sandbox is missing the `newtxtext.sty` font package; compile locally.)

---

## 5. Double-blind anonymization (desk-reject if wrong)

- Author block = **"Anonymous Submission"**, empty affiliations (the `submission` option handles the copyright/anon banner). ✅ already set.
- Remove **all** author/affiliation info; you may put paper number + keywords instead.
- **No acknowledgements, no grant numbers, no personal/GitHub URLs, no dataset names that identify you.**
- Self-citations must be neutral ("prior work [X]", not "our previous work [X]").
- **Strip PDF metadata** before upload (author field in PDF properties leaks identity) — use `exiftool -all= file.pdf` or similar.
- **Grep the source and the code archive** for: your name, email, institution, any `/Users/...` path, old inline bug-comments (`[I1]`–`[I7]` per your checklist). See §6.
- The **supplement and code archive must also be anonymized** — a GitHub link or your name in a script is a de-anonymization.

---

## 6. Concrete pre-submission checklist (do these in order)

**A. Content freeze (this week)**
- [ ] Resolve the remaining paper-specific to-dos from `CHANGELOG_edits.md` "still open": figure readability (Figs 2/3/6), the "non-overlapping by 0.221" number, Proposition-1 hypothesis alignment.
- [ ] **Verify every 2025–26 arXiv citation** (Yamaguchi, Weng, Mengesha, Capannoli, Hussain, etc.) resolves to a real paper with matching title/authors. AAAI explicitly sanctions **non-existent references** — this is the highest-risk integrity item after your LLM-assisted edit pass.
- [ ] Reword the "released on acceptance" footnote and prepare the included code/data archive (§3).

**B. Compile & format (local machine, pdfLaTeX)**
- [ ] `pdflatex → bibtex → pdflatex → pdflatex` on `AnonymousSubmission2027_2.tex`; **zero** undefined citations/refs, zero errors.
- [ ] Open the `.log`: **no overfull \hbox** into margins, no Type-3 font warnings.
- [ ] **Body ≤ 7 pages**, refs on pp. 8–9 only. If over, cut (candidate: the affine-law subsection/Figure 4, per earlier council note).
- [ ] Figures legible at print size and grayscale-safe (line styles + color).

**C. Anonymize**
- [ ] `grep -ri "agriya\|<yourlastname>\|<institution>\|agriyayadav@gmail" .` across `.tex`, `.bib`, and the code archive → 0 hits.
- [ ] `grep -rn "/Users/\|github.com/\|Acknowled" *.tex` → 0 hits in the submission.
- [ ] Strip PDF metadata on the final PDF (and supplement PDF).

**D. Assemble**
- [ ] Main PDF ✔ · Reproducibility checklist PDF ✔ (filled) · Technical supplement PDF ✔ · Anonymized code+data archive (.zip) ✔.
- [ ] Final source zip (only if accepted later): single flattened `.tex` + `.bib` + `.bbl` + used figures, ≤10 MB, named by first-author family name — **not needed now, only post-acceptance.**

**E. Submit on OpenReview**
- [ ] Register/sign in (today).
- [ ] Create submission → enter **real title + real abstract** + keywords/area + dual-submission attestation → **by Jul 21**.
- [ ] Upload main PDF + reproducibility checklist → **by Jul 28**.
- [ ] Upload supplement + code/data archive → **by Jul 31**.
- [ ] Click paper ID → **Email** → save the confirmation.

---

## 7. Policies you must not trip over

- **Dual/multiple submission:** the paper must **not** be under review or published at another archival venue at submission time. arXiv preprints and non-archival workshops are fine. After submitting to AAAI you can't submit substantially-similar work elsewhere until you get a decision or withdraw. You attest to this on the form. (Max 10 AAAI-27 submissions per author.)
- **No "thin slicing":** don't split this into near-duplicate papers — the paper 3 / world-model plan in your folder should be a *distinct* contribution, not an overlapping variant.
- **Generative-AI use:** permitted "judiciously," but **you are fully responsible** for all content. Specifically sanctioned: **non-existent references**, unattributed/plagiarized text, and "cosmetic LLM rewrites to obtain multiple lottery tickets." Given this manuscript went through an LLM-assisted editing + reference pass, the reference-existence check in §6A is not optional.
- **Reviewing duty:** all qualified authors join the reviewer pool (light load).
- **Ethics:** honor the AAAI Publications Ethics statement + Code of Conduct; violations range from desk-reject to multi-year bans.

---

## 8. What the review process looks like (so the timeline makes sense)

- **Two-phase review.** Phase 1 = two human reviews **+ one AI-generated supplementary review** (non-decisional, no rating). An SPC/AC decides if the paper advances.
- **Phase-1 reject → you see reviews Sep 24** and it's over for this cycle.
- **Advance → Phase 2:** more human reviews, then the **author response window Oct 19–25** (your rebuttal — all reviews incl. the AI one are visible). SPC recommends; decision **Nov 30**.
- Human-only decisions; the AI review carries no rating and doesn't decide anything.
- If accepted: **camera-ready Dec 14** (you'll add author names, acknowledgements, the de-anonymized code link, AAAI copyright form, and buy at least one registration — presenting is required for inclusion).

---

## 9. Track choice

Submit to the **Main Technical Track** (keyword area: Multiagent Systems). Your paper is a measurement/empirical MARL contribution — it is *not* an AI-for-Social-Impact or AI-Alignment special-track paper (those use different rubrics), though loss-aversion/reward-transform framing is adjacent to Alignment. Main Track is the correct home.

---

## Sources
- [AAAI-27 Submission Instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/)
- [AAAI-27 Main Technical Track Call](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)
- [AAAI-27 Author Kit](https://aaai.org/authorkit27/)
- [AAAI-27 conference home](https://aaai.org/conference/aaai/aaai-27/)
- [OpenReview AAAI 2027 group](https://openreview.net/group?id=AAAI.org/2027/Conference)
- Local: `AuthorKit27/`, `AAAI27_SUBMISSION_CHECKLIST.md`, `AuthorKit27/ReproducibilityChecklist.tex`
