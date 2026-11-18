#!/bin/bash
#
# Did an OSCAR job actually finish, or did it just stop?
#
#   ./check_oscar_run.sh 9252850
#
# Checks, in order of authority:
#   1. SLURM's own exit code         (the only thing that cannot be faked)
#   2. the script's final log line   (did it reach the end of the recipe?)
#   3. the artefacts on disk         (did every stage write what it owes?)
#   4. the git push                  (are the results reachable off-cluster?)
#
# A job can report COMPLETED and still be useless if a stage wrote nothing, so
# none of these four is sufficient on its own.

set -uo pipefail

JOBID="${1:-}"
if [[ -z "$JOBID" ]]; then
    echo "usage: $0 <jobid>"
    echo
    echo "recent jobs:"
    sacct -u "$USER" --format=JobID%14,JobName%18,State%14,ExitCode,Elapsed,End -X | tail -15
    exit 2
fi

ok=0; bad=0
pass() { echo "  [PASS] $*"; ok=$((ok+1)); }
fail() { echo "  [FAIL] $*"; bad=$((bad+1)); }
warn() { echo "  [WARN] $*"; }

# `cmd | grep -q` is a trap under `set -o pipefail`: grep exits on the first
# match, SIGPIPEs the producer, and pipefail reports the pipeline as failed
# even though the pattern WAS found. Buffer first, then match.
has() {   # has <pattern> <<< text-on-stdin
    local txt; txt=$(cat)
    [[ -n "$txt" ]] && echo "$txt" | grep -q -- "$1"
}

echo "================================================================"
echo "Verifying OSCAR job $JOBID"
echo "================================================================"

# --- 1. SLURM exit status ---------------------------------------------------
echo
echo "1. SLURM accounting"
JOBNAME=""
if ! command -v sacct >/dev/null 2>&1; then
    warn "no sacct here (not a SLURM host) — skipping. The artefact and JSON"
    warn "  checks below still apply; run this on OSCAR for the exit code."
    LINE=""
else
    LINE=$(sacct -j "$JOBID" -X -n --format=JobName%24,State%20,ExitCode,Elapsed,MaxRSS 2>/dev/null | head -1)
fi
if [[ -z "$LINE" ]]; then
    command -v sacct >/dev/null 2>&1 && \
        fail "job $JOBID not in sacct (too old, or wrong id)"
else
    echo "     $LINE"
    JOBNAME=$(echo "$LINE" | awk '{print $1}')
    STATE=$(echo "$LINE" | awk '{print $2}')
    CODE=$(echo "$LINE" | awk '{print $3}')
    ELAPSED=$(echo "$LINE" | awk '{print $4}')
    if [[ "$STATE" == "COMPLETED" && "$CODE" == "0:0" ]]; then
        pass "state=$STATE exit=$CODE elapsed=$ELAPSED"
    else
        fail "state=$STATE exit=$CODE elapsed=$ELAPSED"
        case "$STATE" in
            TIMEOUT)   warn "walltime hit — raise #SBATCH --time and resubmit" ;;
            OUT_OF_ME*) warn "OOM — raise #SBATCH --mem and resubmit" ;;
            FAILED)    warn "a command returned non-zero; set -e aborted the script" ;;
            CANCELLED) warn "cancelled (by you, or by the scheduler)" ;;
        esac
    fi
fi

# --- 2. the script's own log ------------------------------------------------
echo
echo "2. Script log"
LOGF=$(ls -1 outputs/logs/*_"${JOBID}".log 2>/dev/null | head -1)
ERRF=$(ls -1 outputs/logs/*_"${JOBID}".err 2>/dev/null | head -1)
OUTF=$(ls -1 outputs/logs/*_"${JOBID}".out 2>/dev/null | head -1)

if [[ -z "$LOGF" ]]; then
    fail "no outputs/logs/*_${JOBID}.log — the script died before its first log() call"
else
    echo "     $LOGF"
    echo "     --- last 6 lines ---"
    tail -6 "$LOGF" | sed 's/^/     /'
    echo "     --------------------"
    if grep -q "Job completed successfully\|^\[.*\] Done\." "$LOGF"; then
        pass "reached the end-of-recipe marker"
    else
        fail "no completion marker — the script stopped partway"
    fi
    # Without sacct (e.g. checking from a laptop) the log filename is the only
    # thing that says which recipe ran, and that decides the file inventory.
    [[ -z "$JOBNAME" ]] && JOBNAME=$(basename "$LOGF" | sed "s/_${JOBID}\.log$//")
fi

# A stage can print FAIL and still exit 0 if it is behind a tee/pipe. Scan only
# THIS job's records: .out captures every stage's stdout, including the tee'd
# verification output. Scanning logs/*.log would pick up unrelated older runs.
for f in "$OUTF" $(ls -1 logs/*"${JOBID}"*.log 2>/dev/null); do
    [[ -f "$f" ]] || continue
    n=$(grep -c "\[FAIL\]\|FATAL\|Traceback\|SOME CHECKS FAILED" "$f" 2>/dev/null)
    [[ "$n" -gt 0 ]] && fail "$n failure line(s) inside $f"
done
# These two are appended to across runs (tee -a), so a hit may be stale.
for f in logs/verify_theory_gurobi.log logs/verify_cvar_pricing_gurobi.log; do
    [[ -f "$f" ]] || continue
    n=$(grep -c "\[FAIL\]\|SOME CHECKS FAILED" "$f" 2>/dev/null)
    [[ "$n" -gt 0 ]] && warn "$n failure line(s) in $f (appended across runs — may predate this job)"
done
[[ -n "$ERRF" && -s "$ERRF" ]] && { warn "stderr is non-empty: $ERRF"; tail -5 "$ERRF" | sed 's/^/       /'; }

# --- 3. artefacts -----------------------------------------------------------
echo
echo "3. Artefacts on disk"
check() {   # check <label> <glob>
    local n
    n=$(ls -1 $2 2>/dev/null | wc -l)
    if [[ "$n" -gt 0 ]]; then pass "$1 ($n file(s))"; else fail "$1 — nothing matching $2"; fi
}

if [[ "$JOBNAME" == "zavala_im3" || -z "$JOBNAME" ]]; then
    check "verify_theory log"        "logs/verify_theory_gurobi.log"
    check "verify_cvar_pricing log"  "logs/verify_cvar_pricing_gurobi.log"
    check "D1-D5 decentralisation"   "outputs/results/decentralized_dec_S500_${JOBID}.json"
    for t in main oos paired winter_h8 winter_h18 summer_h8 summer_h18; do
        check "E-suite: $t"          "outputs/results/im3_im3_S500_${JOBID}_${t}.json"
    done
fi
if [[ "$JOBNAME" == "zavala_tail" || -z "$JOBNAME" ]]; then
    check "W0-W7 headline"           "outputs/results/tail_tail_${JOBID}_main.json"
    check "ramp extension"           "outputs/results/tail_tail_${JOBID}_rampext.json"
    check "beta=0.99 robustness"     "outputs/results/tail_tail_${JOBID}_beta99.json"
    check "figures"                  "figures/tail/*.pdf"
fi

# Any JSON that is truncated (job killed mid-write) is worse than a missing one.
echo
echo "   JSON integrity:"
for j in $(ls -1 outputs/results/*"${JOBID}"*.json 2>/dev/null); do
    if python -c "import json,sys; json.load(open(sys.argv[1]))" "$j" 2>/dev/null; then
        printf "     ok      %8s  %s\n" "$(du -h "$j" | cut -f1)" "$j"
    else
        fail "truncated/invalid JSON: $j"
    fi
done

# --- 4. did it get off the cluster? ----------------------------------------
echo
echo "4. Results pushed to GitHub"
git fetch origin hrithik_final -q 2>/dev/null
LOCAL=$(git rev-parse HEAD 2>/dev/null)
REMOTE=$(git rev-parse origin/hrithik_final 2>/dev/null)
if git log origin/hrithik_final --oneline -40 2>/dev/null | has "job ${JOBID}"; then
    pass "a commit for job ${JOBID} is on origin/hrithik_final"
elif [[ "$LOCAL" != "$REMOTE" ]]; then
    warn "local HEAD != origin/hrithik_final; results may be committed but unpushed"
    warn "  -> git push origin hrithik_final"
elif git log --oneline -40 2>/dev/null | has "job ${JOBID}"; then
    warn "job ${JOBID} is committed locally but NOT on the remote."
    warn "  Compute nodes have no outbound network, so the job could not push."
    warn "  From a LOGIN node:   cd $PWD && ./push_results.sh"
else
    fail "no commit for job ${JOBID}, locally or on the remote"
    warn "  From a LOGIN node:   cd $PWD && ./push_results.sh"
    warn "  (it sweeps up uncommitted results before pushing)"
fi

echo
echo "================================================================"
echo "$ok passed, $bad failed"
[[ "$bad" -eq 0 ]] && echo "Run is COMPLETE." || echo "Run is INCOMPLETE — see the FAIL lines above."
echo "================================================================"
exit $(( bad > 0 ))
