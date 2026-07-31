#!/bin/bash
set -euo pipefail

# ============================================================
#  UNTRACKED - FINAL SUBMISSION ARTIFACTS
# ============================================================

git add AAAI27_SUBMISSION_GUIDE.md
git commit -m "Add AAAI 2027 submission guide

Comprehensive instructions and checklist for finalizing and packaging
the AAAI submission materials, ensuring all requirements are met."

git add AuthorKit27/AnonymousSubmission2027_2* AuthorKit27/AnonymousSubmission2027_3* \
  AuthorKit27/_restore_body.tex
git commit -m "Add AAAI submission iterative drafts and backups

Intermediate drafts, clean PDFs, and LaTeX compilation artifacts
(including backup restore files) generated during the final polish
of the main submission body."

git add AuthorKit27/SupplementaryMaterial2027_2* AuthorKit27/SupplementaryMaterial2027_3* \
  AuthorKit27/_restore_supp.tex
git commit -m "Add AAAI supplementary material iterative drafts and backups

Intermediate drafts, clean PDFs, and LaTeX compilation artifacts
generated during the final formatting of the supplementary appendix."

git add AuthorKit27/ReproducibilityChecklist.aux AuthorKit27/ReproducibilityChecklist.log \
  AuthorKit27/ReproducibilityChecklist_clean.pdf
git commit -m "Add AAAI reproducibility checklist build artifacts

Clean compiled PDF and LaTeX log artifacts for the final
reproducibility checklist."

git add AuthorKit27/aaai2027_code_data_archive_curated.zip
git commit -m "Add compiled AAAI code and data archive

Final zipped archive of the curated repository, scripts, and data
prepared for the supplementary material submission portal."

git add auto_commit.sh
git commit -m "Update auto_commit.sh utility script

Refreshes the batch commit script to include the final batch of
AAAI submission artifacts."

echo "=== All commits done. Pushing... ==="
git push
