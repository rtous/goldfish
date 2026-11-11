#!/bin/sh
# Builds paper_PRL/submission/: a flat folder (no subdirectories, as Elsevier's Editorial Manager
# flattens uploads) with the LaTeX source of the manuscript, its PDF and the extra documents.
# Usage (from anywhere): sh paper_PRL/make_submission.sh
set -e
cd "$(dirname "$0")"

# Compile the manuscript and the extra documents.
latexmk -pdf -interaction=nonstopmode -quiet main.tex
for f in highlights authorship_confirmation cover_letter declarations; do
  pdflatex -interaction=nonstopmode "$f.tex" > /dev/null
done

rm -rf submission
mkdir submission

# Manuscript source with the tables inlined and figure paths flattened.
python3 - <<'EOF'
import re
s = open("main.tex").read()
s = re.sub(r"\\input\{tables/(\w+)\}", lambda m: open("tables/%s.tex" % m.group(1)).read().strip(), s)
s = s.replace("{figures/", "{")
s = s.replace("\\bibliography{references}", "\\bibliography{manuscript}")
open("submission/manuscript.tex", "w").write(s)
EOF
cp numbers.tex prletters.sty model2-names.bst elsevier-logo.pdf figures/*.pdf submission/
cp references.bib submission/manuscript.bib

# Check that the flat source compiles on its own, and keep its .bbl for the journal.
(cd submission && latexmk -pdf -interaction=nonstopmode -quiet manuscript.tex > /dev/null &&
 latexmk -c manuscript.tex > /dev/null && rm -f manuscript.spl)

cp highlights.pdf highlights.tex highlights.txt cover_letter.pdf cover_letter.tex \
   authorship_confirmation.pdf declarations.pdf submission/

rm -f ./*.aux ./*.log ./*.out ./*.fls ./*.fdb_latexmk ./*.blg ./*.spl
echo "submission/ ready:"
ls submission
