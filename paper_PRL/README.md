# Pattern Recognition Letters version

This folder is the Pattern Recognition Letters (PRL, Elsevier) version of `../paper/`.
`submission/` holds everything to upload.

## Build

    sh paper_PRL/make_submission.sh                 # manuscript, extra documents and submission/
    venv/bin/python paper_PRL/make_figures.py       # only if the results in ../experiments/ change

`numbers.tex` and `tables/` are copies of the files in `../paper/`. If the results change, run
`../paper/make_tables.py` and copy them here again.

## PRL requirements applied

Taken from the official PRL LaTeX template and its instructions. The ScienceDirect guide for authors
blocks automated access, so check the live guide before you submit.

- Template: `elsarticle` class (`times,twocolumn,final,authoryear`) with Elsevier's `prletters.sty`,
  plus `model2-names.bst` for author–year (Harvard) references.
- Length: **at most 7 pages, including figures, tables and references** (one more page is allowed
  after revision). The manuscript is exactly 7 pages.
- Abstract: at most 200 words. It is now 199 words.
- Title: only the first word and proper nouns capitalized.
- Highlights: required, as a separate file, with 3–5 bullets of at most 85 characters each.
  There are 5 bullets, the longest 79 characters.
- Confirmation of Authorship form: required.
- Figures are at most half a page, and their text is at least as large as the captions (8 pt).
  Captions go below figures and above tables.
- Declarations before the references: competing interests, data availability, and use of generative AI.
- Graphical abstract: optional, not included.

## What was cut from ../paper (15 single-column pages → 7 double-column pages)

- Abstract rewritten for the 200-word limit. Introduction, related work and setup condensed. The
  contribution list is now inline, and Discussion, Limitations and Future work are merged into one section.
- Tables removed: input streams, same book / earlier book, parameter subsets, MLPs vs all, and
  collateral forgetting. Their key numbers stay in the text.
- Figures removed: rules vs size and loss by position in 200k-token books. The rules table and the
  very-long table give the same numbers.
- Layers and locality levels are now separate one-column figures. The trade-off figure is redrawn at
  one-column width.
- Two references were dropped: `zhao2021calibrate` and `clark2022meta`.

## Before submitting, check

1. **Affiliation**: the department ("Department of Computer Architecture, UPC, Jordi Girona 1-3,
   08034 Barcelona") was added; check it is right for all three authors.
2. **CRediT statement** in `declarations.tex` is a draft. Check each author's roles.
3. **Generative AI declaration** (in `main.tex` and `declarations.tex`): adjust it to how AI tools
   were used, or remove it if they were not.
4. **Authorship confirmation**: fill in pre-prints (e.g. arXiv) and conference versions, then sign and
   date it. If there is a preprint, it must state that the paper is under consideration at PRL.
5. The submission system may ask for **suggested reviewers** and keywords. The keywords are in
   `main.tex`; PRL's layout does not print them on the page.

## Files to upload (submission/)

| File | Item type |
|---|---|
| `cover_letter.pdf` | Cover letter |
| `highlights.pdf` (or `.txt` / `.tex`) | Highlights |
| `authorship_confirmation.pdf` (after signing) | Confirmation of Authorship |
| `declarations.pdf` | Declaration of interest / CRediT statement |
| `manuscript.tex`, `manuscript.bbl`, `manuscript.bib`, `numbers.tex`, `prletters.sty`, `model2-names.bst`, `elsevier-logo.pdf` | Manuscript (LaTeX source) |
| `distance_position.pdf`, `locality_levels.pdf`, `layers.pdf`, `eviction.pdf`, `tradeoff.pdf` | Figures |
| `manuscript.pdf` | Manuscript PDF, if requested |
