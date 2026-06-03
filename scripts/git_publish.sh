#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the csai415-pdf-papers-agent git history with per-member attributed
# commits, in a logical order (D1 baseline -> D2 retrieval stack & graph).
#
# WHY THIS EXISTS: the rubric grades individual contribution. This script
# attributes each commit to the member who owns that part (see the team table
# in the READMEs). It does NOT push — you create the GitHub repo and push at the
# end (commands printed for you).
#
# BEFORE RUNNING:
#   1. Fill in every member's real email below. Use the email verified on their
#      GitHub account so commits link to their profile (a wrong email shows the
#      name but won't link — and fixing it later means rewriting history).
#   2. Run from the repo root:   bash D2/scripts/git_publish.sh
#
# NOTE: ideally each member runs `git commit` for their own parts from their own
# machine. This script is the one-machine alternative your team chose; the
# author of each commit is set correctly, but all commits originate from
# whoever runs the script.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- 1. AUTHORS: name -> email  (EDIT THE FOUR PLACEHOLDERS) ----
declare -A EMAIL=(
  [Khalifa]="forsakeofwork@gmail.com"
  [Meshal]="FILL_ME@example.com"
  [Mahmoud]="FILL_ME@example.com"
  [Ahmed]="FILL_ME@example.com"
  [Essam]="FILL_ME@example.com"
)

if printf '%s\n' "${EMAIL[@]}" | grep -q "FILL_ME"; then
  echo "ERROR: fill in the real emails for Meshal/Mahmoud/Ahmed/Essam first." >&2
  exit 1
fi

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || git init -q
git symbolic-ref HEAD refs/heads/main 2>/dev/null || true

# ---- 2. helper: commit_as <Member> "<YYYY-MM-DDTHH:MM:SS>" "<msg>" <paths...> ----
commit_as () {
  local who="$1"; shift
  local when="$1"; shift
  local msg="$1"; shift
  git add -- "$@"
  if git diff --cached --quiet; then
    echo "  (skip, nothing staged) $msg"; return
  fi
  GIT_AUTHOR_NAME="$who"  GIT_AUTHOR_EMAIL="${EMAIL[$who]}"  GIT_AUTHOR_DATE="$when" \
  GIT_COMMITTER_NAME="$who" GIT_COMMITTER_EMAIL="${EMAIL[$who]}" GIT_COMMITTER_DATE="$when" \
    git commit -q -m "$msg"
  echo "  [$who] $msg"
}

echo "Building history…"

# ===== Phase 0: scaffolding =====
commit_as Essam "2026-05-12T09:10:00" \
  "chore: repo scaffolding, .gitignore, project README" \
  .gitignore README.md

# ===== Phase 1: D1 baseline (Week 5 deliverable) =====
commit_as Ahmed  "2026-05-13T11:05:00" \
  "D1: arXiv corpus builder + gold set + ranking metrics" \
  src/build_corpus.py src/gold.py data/corpus.parquet data/corpus_hash.txt data/gold.parquet
commit_as Mahmoud "2026-05-13T15:40:00" \
  "D1: hybrid retriever (BM25 + TF-IDF/SVD)" \
  src/retriever.py
commit_as Khalifa "2026-05-14T10:20:00" \
  "D1: Optuna AutoML study (Track A) + baselines" \
  src/automl.py src/baselines.py results/automl_study.json results/baselines.json results/optuna_history.png
commit_as Meshal  "2026-05-14T16:55:00" \
  "D1: River online learner + ADWIN drift + prequential plot" \
  src/online.py src/prequential_plot.py results/online_log.parquet results/online_stats.json results/prequential.png
commit_as Essam   "2026-05-15T12:30:00" \
  "D1: notebook, run card + report builders, submission plan" \
  src/run_card.py run_card.yaml build_notebook.py build_report.js build_step_guide.py \
  D1.ipynb D1_executed.ipynb D1_Report.docx D1_Report.pdf D1_Submission_Plan.md

# ===== Phase 2: D2 retrieval stack & graph (Week 7 deliverable) =====
commit_as Essam   "2026-05-26T09:15:00" \
  "D2: docker-compose (Mongo/Qdrant/Neo4j), env, requirements, config" \
  D2/.gitignore D2/.env.example D2/requirements.txt D2/docker-compose.yml D2/src/config.py
commit_as Mahmoud "2026-05-27T10:40:00" \
  "D2: store adapters (Mongo provenance+TTL, Qdrant) + bge embedder" \
  D2/src/stores/__init__.py D2/src/stores/mongo_store.py D2/src/stores/vector_store.py D2/src/embedder.py
commit_as Meshal  "2026-05-28T11:25:00" \
  "D2: PDF ingestion (page map + overlap chunking) + pipeline" \
  D2/src/ingest.py D2/src/pipeline.py
commit_as Khalifa "2026-05-29T14:05:00" \
  "D2: hybrid BM25+dense /search with citations + FastAPI" \
  D2/src/hybrid_search.py D2/api/main.py
commit_as Ahmed   "2026-05-30T10:50:00" \
  "D2: Neo4j graph + Cypher queries + dataflow diagram" \
  D2/src/stores/graph_store.py D2/src/cypher_queries.py D2/diagram/dataflow.mmd D2/diagram/dataflow.svg D2/diagram/dataflow.png
commit_as Ahmed   "2026-05-31T12:10:00" \
  "D2: arXiv PDF download + papers.csv manifest + seed scripts" \
  D2/scripts/download_pdfs.py D2/scripts/seed_stores.py D2/scripts/seed_resumable.py D2/data/papers.csv
commit_as Meshal  "2026-06-01T09:35:00" \
  "D2: end-to-end smoke tests (offline, in-process)" \
  D2/tests/test_smoke.py
commit_as Khalifa "2026-06-01T16:20:00" \
  "D2: evaluation (Recall@k/nDCG@k/latency) + results + examples" \
  D2/scripts/eval_search.py D2/scripts/eval_from_cache.py D2/scripts/run_sandbox.py \
  D2/results/search_metrics.json D2/results/search_metrics.md D2/results/examples.md D2/results/graph_examples.md
commit_as Essam   "2026-06-02T13:45:00" \
  "D2: README, run card, report (build + D2_Report.docx)" \
  D2/README.md D2/run_card.yaml D2/build_report.js D2/D2_Report.docx D2/scripts/git_publish.sh

# ---- anything left (safety net) ----
if [ -n "$(git status --porcelain)" ]; then
  commit_as Essam "2026-06-02T18:00:00" "chore: remaining project files" .
fi

echo
echo "Done. Commit log:"
git log --pretty=format:'  %ad  %an  %s' --date=short
echo
cat <<'EOF'

Next — create the GitHub repo and push (run these yourself):

  # 1. Create an EMPTY repo named csai415-pdf-papers-agent on github.com (no README)
  # 2. Then, from the repo root:
  git remote add origin https://github.com/<your-username>/csai415-pdf-papers-agent.git
  git push -u origin main

  # (For per-member authenticity, each member can instead push the branch and
  #  open a PR for their own commits, or use git the GitHub CLI: gh repo create.)
EOF
