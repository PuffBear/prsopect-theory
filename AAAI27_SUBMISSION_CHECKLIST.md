# AAAI-2027 Anonymous Submission — Compliance Checklist

Derived from a single read of `AuthorKit27/AnonymousSubmission2027.tex` (the official
2027 template) and `aaai2027.sty`. This is the authoritative to-do list for turning the
manuscript into a compliant anonymous submission. The submission file we are building is
**`AuthorKit27/AnonymousSubmission2027.tex`** (rename to first-author family name only at
final upload).

> **Why a fresh file, not the old `main_aaai_trimmed.tex`?** The trimmed source loads
> packages AAAI-2027 *explicitly forbids* (`times`, `helvet`, `courier`, `hyperref`,
> manual `\pdfpagewidth/height`, `\nocopyright`) and uses the legacy `aaai/aaai` style.
> Reusing its preamble would get the paper returned. We port the **content** into the
> official template, which manages fonts/margins automatically.

---

## A. Preamble — MUST match the template exactly

- [x] `\documentclass[letterpaper]{article}` — do not change.
- [x] `\usepackage[submission]{aaai2027}` — the `submission` option anonymizes + suppresses copyright. (Do **not** use `\nocopyright`; it is a disallowed command.)
- [x] Fonts loaded automatically by the style (`newtxtext`, `helvet`, `courier`). **Do not** load `times`, `helvet`, `courier`, or any font package.
- [x] Keep `\pdfinfo{ /TemplateVersion (2027.1) }`.
- [ ] Only add packages compatible with the style: `amsmath, amssymb, amsthm` (math/theorems), `graphicx` (already), `booktabs` (tables), `subcaption` if multi-panel figures are needed. **amsthm is fine** for the Theorem environment.

### Forbidden packages (paper returned if present)
`authblk, babel, balance, cjk, epsf, epsfig, euler, float, fullpage, geometry, graphics,
hyperref, layout, lmodern, navigator, pdfcomment, pgfplots, psfig, pstricks, t1enc, times,
titlesec, tocbibind, ulem, multicol, setspace, wrapfig, indentfirst, savetrees, stfloats,
flushend, nameref`

**Action items from the old source that violate this:**
- [ ] Remove `\usepackage{hyperref}` and the `\hypersetup{...}` block. (No links/bookmarks allowed.) Replace `\Cref/\cref` (cleveref) with plain `\ref` / `Section~\ref{}` etc. — **cleveref is risky with this style; safest to drop it.**
- [ ] Remove `\usepackage{times}`, `{helvet}`, `{courier}`.
- [ ] Remove `\usepackage[acronym]{glossaries}`, `csquotes`, `microtype`, `xcolor` unless genuinely needed (microtype is generally OK but unnecessary; xcolor only if a figure callout needs one colored word — text color is otherwise forbidden).
- [ ] Remove manual `\setlength{\pdfpagewidth/height}` and `\nocopyright`.
- [ ] Drop the legacy `\usepackage{aaai/aaai}` — replaced by `aaai2027`.

### Forbidden commands
`\nocopyright, \addtolength, \balance, \baselinestretch, \clearpage, \columnsep, \newpage,
\pagebreak, \pagestyle, \tiny, \linespread, \setlength{\textheight/\topmargin/...},
negative \vspace{-...} / \vskip{-...} near captions/figures/tables/sections/refs.`
- [ ] Grep the ported body for any of these and remove.

---

## B. Title, authors, abstract

- [ ] **Title in Title Case** (Chicago). Current: "Coordination-Collapse Phase Boundary in Policy-Gradient Cooperative MARL" — verify each word's case (hyphenated terms capitalize both parts). Do **not** wrap title in `\textbf{}` (the style bolds it).
- [x] Author block = `Anonymous Submission` as sole author, empty affiliations (handled by `submission` option / template default).
- [ ] **Abstract: no citations** (rule). The current abstract has none — keep it that way. Target ≤ ~1 paragraph; do not indent further.
- [ ] Optional `\begin{links}...\end{links}` block (code/data/extended version) goes **between** abstract and body. Only add if a fully anonymized link exists; otherwise omit. (Do not de-anonymize.)

---

## C. Sections & structure

- [ ] `\setcounter{secnumdepth}{0}` by default (no section numbers). May set to 1 or 2 if numbers wanted — never higher (subsubsection numbering breaks the style).
- [ ] Section order required: main content → optional content appendices (count toward page limit) → Ethical Statement (optional, unnumbered) → Acknowledgments (optional, unnumbered, **omit in anonymous submission** to avoid de-anon) → References → supplementary/technical appendix (only if the conference permits).
- [ ] Use `\section*{}` for any unnumbered sections.

---

## D. Figures & tables

- [x] Figures live in `AuthorKit27/Figures/` (PNG/PDF only — no EPS/PS/GIF). Copied in this session.
- [ ] Reference with relative path `Figures/name` (no extension needed). Update all `\includegraphics{../docs/...}` paths from the old source to `Figures/...`.
- [ ] Figures **300 dpi**, fonts embedded, color WCAG-2.0 / grayscale-safe (the decomposition figure uses blue/orange/red — verify it survives grayscale; consider linestyles in addition to color).
- [ ] **No `clip`/`trim`/`viewport`** in `\includegraphics`. Crop externally if needed.
- [ ] Single-column figures: `\includegraphics[width=0.95\columnwidth]`. Double-column (span): use `figure*` + `width=...\textwidth`. Float to **top** (`[t]`) to save space.
- [ ] Captions **below** figure/table, 10pt roman, not smaller/bold/italic.
- [ ] Tables 10pt (9pt if needed); use `booktabs`. No `\resizebox`. `\setlength{\tabcolsep}{1mm}` is the *one* permitted `\setlength` exception for compression. Wide tables → `table*`.
- [ ] **No overfull boxes** — check the `.log`. Nothing may intrude into margin/gutter.

---

## E. References / bibliography

- [ ] Bib style is set automatically by `aaai2027.sty` — **do not** add `\bibliographystyle`. (Old source had `\bibliographystyle{plainnat}` — remove.)
- [ ] `\bibliography{references}` (the `references.bib` copied into AuthorKit27). Single `.tex`, separate `.bib` is the only allowed external file.
- [ ] References must be **anonymized** where they'd reveal authorship (self-cites → neutral phrasing). Check none of the cited 2025/2026 arXiv items de-anonymize.
- [ ] arXiv entries use `@misc` with `eprint` + `archivePrefix`. Verify every entry has no missing required fields (missing fields = compile error / wrong entry class).
- [ ] All citation keys resolve (0 undefined). Spot-check the flagged 2025/2026 entries are real.

---

## F. Anonymization & metadata (anonymous submission specific)

- [ ] Author = "Anonymous Submission", affiliations empty. (Template default.)
- [ ] **Strip PDF metadata** with a metadata-cleaning tool before upload (prevents identity leak via PDF properties).
- [ ] No acknowledgments, no grant numbers, no GitHub/personal URLs, no dataset names tied to the author.
- [ ] Grep source for `Agriya`, `Ashoka`, the email, any `/Users/...` absolute path, and the `[I1]`–`[I7]` inline bug-comments from the old lineage. Remove all.
- [ ] No copyright footer on page 1 (handled by `submission` option).

---

## G. Compile / file hygiene

- [ ] Must compile in **pdfLaTeX** (not xelatex/lualatex), TeXLive-2020-compatible packages only.
- [ ] All fonts embedded; **no Type-3 fonts** (check figure exports; export plots as PDF or 300dpi PNG, not EPS).
- [ ] PDF version ≥ 1.5; US-letter; two-column (automatic); no page numbers/headers/footers.
- [ ] **Single `.tex` file** for final submission — every section inline, no `\input` (except the `.bib`). (We can develop in pieces but must flatten before upload.)
- [ ] Source must exactly match the PDF.
- [ ] Final archive: `.zip`, ≤ 10 MB, named with first author's family name; include only figures actually used + `.bib` + `.bbl` + PDF. Remove unused figures (`figure1.pdf`, `figure2.pdf` template samples) and this instructions file before zipping.

---

## H. Page limit

- [ ] **Confirm AAAI-2027 page limit on the conference website** (typically 7 pages body + references + optional appendix; varies by track). The template/style does not encode it. Trim body to the limit; appendices count if "content appendices," supplementary may not be published.

---

## Status snapshot (this session)
- Figures copied to `Figures/` ✅
- `references.bib` copied ✅
- Building intro inside official template ⏳ (in progress)
- Full body port (Background → Conclusion), package cleanup, reference anonymization, metadata strip — **pending** (subsequent passes)
