#!/bin/bash
# Shared launch discipline for long analysis jobs on the shared 256-core / 4-GPU / 377 GB server (AGENTS.md, "Compute rules").
# Lives in scripts/; source it from the repository root (`source scripts/jobs.sh`), then use:
#   wait_for_capacity            block until load < MAX_LOAD, available memory minus the reservations of admitted jobs > JOB_MEM_GB + HEADROOM_GB
#                                and fewer than MAX_HEAVY heavy jobs run (run_heavy / run_light do this check and the launch under one lock,
#                                results/pids/.admission.lock, and record the declared memory in NAME.mem; 2026-09-25)
#   run_heavy  NAME CORES CMD    run a heavy CPU job (16 threads, or HEAVY_THREADS; declare its peak memory with JOB_MEM_GB=<GB>, default 30) pinned to CORES
#   run_light  NAME CORES CMD    run a light job (GPU work, <= 8 threads) pinned to CORES after a lighter capacity check
#   wait_pidfiles NAME...        block until those jobs have finished; exit status 1 if any is unknown or failed, so chains
#                                must write `wait_pidfiles A B || exit 1` (no pgrep patterns, no silent continue on a missing file)
#   job_state NAME               running | succeeded | failed N | failed lost | unknown
# Every job writes results/logs/NAME.log, results/pids/NAME.pid (wrapper), NAME.child (command), NAME.state.
# Heavy jobs are marked by results/pids/NAME.heavy.
MAX_LOAD=${MAX_LOAD:-120}; HEADROOM_GB=${HEADROOM_GB:-100}; MAX_HEAVY=${MAX_HEAVY:-2}   # headroom rule of 2026-09-25 (AGENTS.md): available > job estimate + headroom
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; _PIDROOT="$HERE/results/pids"; _LOGROOT="$HERE/results/logs"; mkdir -p "$_PIDROOT" "$_LOGROOT"   # HERE = repository root
# The results tree is shared between hosts (2026-09-26): PID files, states, heavy marks, the admission lock AND the logs are per host.  The host
# that first sourced the launcher owns results/pids and results/logs itself (recorded in results/pids/.host); every other host uses
# results/pids/<host> and results/logs/<host>; JOB_PIDDIR and JOB_LOGDIR override.  A job's host is recorded in NAME.host, and job_state
# refuses to judge a job of another host.
_HOST=$(hostname -s); [ -s "$_PIDROOT/.host" ] || echo "$_HOST" > "$_PIDROOT/.host"
if [ "$(cat "$_PIDROOT/.host")" = "$_HOST" ]; then _SUB=""; else _SUB="/$_HOST"; fi
PIDS="${JOB_PIDDIR:-$_PIDROOT$_SUB}"; LOGS="${JOB_LOGDIR:-$_LOGROOT$_SUB}"; mkdir -p "$PIDS" "$LOGS"

_load() { cut -d' ' -f1 /proc/loadavg | cut -d. -f1; }
_free_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
_heavy_running() { local n=0; for f in "$PIDS"/*.heavy; do [ -e "$f" ] || continue; local p; p=$(cat "${f%.heavy}.pid" 2>/dev/null); if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then n=$((n+1)); else rm -f "$f"; fi; done; echo $n; }

_reserved_gb() {              # memory declared by running jobs that they have not reached yet: sum of max(0, declared - RSS of the command)
    local tot=0 f name p c mem rss
    for f in "$PIDS"/*.mem; do
        [ -e "$f" ] || continue; name=$(basename "${f%.mem}"); p=$(cat "$PIDS/$name.pid" 2>/dev/null); mem=$(cat "$f")
        if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then rm -f "$f"; continue; fi
        c=$(cat "$PIDS/$name.child" 2>/dev/null); rss=0
        [ -n "$c" ] && rss=$(ps -o rss= -p "$c" 2>/dev/null | awk '{printf "%d", $1/1048576}'); rss=${rss:-0}
        [ "$mem" -gt "$rss" ] && tot=$((tot + mem - rss))
    done
    echo $tot
}
_capacity_ok() {              # $1: heavy slots needed; $2: declared peak memory in GB.  Available memory minus the reservations of admitted jobs
    # One consistent host-level heavy limit (results/pids[/<host>]/.max_heavy, live-adjustable), env MAX_HEAVY as fallback.
    # LIGHT jobs (need=0) are gated only by load and memory+headroom -- a heavy-job limit must NOT block them (2026-09-30).
    local need=${1:-1} mem=${2:-30} l f h r maxh
    l=$(_load); f=$(_free_gb); h=$(_heavy_running); r=$(_reserved_gb)
    maxh=$(cat "$PIDS/.max_heavy" 2>/dev/null); maxh=${maxh:-$MAX_HEAVY}
    if [ "$l" -lt "$MAX_LOAD" ] && [ $((f - r)) -gt $((mem + HEADROOM_GB)) ] && { [ "$need" -eq 0 ] || [ $((h + need)) -le "$maxh" ]; }; then return 0; fi
    echo "[jobs.sh] waiting: load $l (max $MAX_LOAD), available ${f} GB minus ${r} GB reserved (need ${mem} GB + ${HEADROOM_GB} GB headroom), heavy jobs $h (max $maxh, need $need) $(date +%H:%M)"
    return 1
}
wait_for_capacity() {           # $1: how many heavy slots this job needs (0 for light jobs); $2: its expected peak memory in GB (no lock; see _admit)
    while ! _capacity_ok "${1:-1}" "${2:-30}"; do sleep "${JOB_POLL_S:-120}"; done
}
_admit() {                      # NAME CORES THREADS HEAVY MEM CMD...: the capacity check and the launch under one lock, so that two queues
    # cannot pass the same check together; the declared memory is recorded (NAME.mem) and counts as reserved until the job's RSS reaches it
    local name=$1 cores=$2 threads=$3 heavy=$4 mem=$5 rc; shift 5
    while true; do
        (
            flock -w 600 9 || exit 3
            _capacity_ok "$heavy" "$mem" || exit 1
            echo "$mem" > "$PIDS/$name.mem"
            _launch "$name" "$cores" "$threads" "$heavy" "$@" || { rm -f "$PIDS/$name.mem"; exit 2; }
        ) 9>"$PIDS/.admission.lock"; rc=$?
        case $rc in 0) return 0;; 1) sleep "${JOB_POLL_S:-120}";; 2) return 1;; *) echo "[jobs.sh] ERROR: admission lock not acquired for $name"; return 1;; esac
    done
}

_launch() {                     # NAME CORES THREADS HEAVY CMD...
    # A wrapper shell records ITS OWN pid (setsid may fork, so $! is not reliable), runs the command, forwards TERM/INT to
    # (descriptor 9, the admission lock of _admit, is closed for the detached process: a job must not hold the lock while it runs)
    # it, waits until the command and every descendant are gone, and records the outcome in NAME.state
    # (running -> succeeded | failed N).  NAME.pid is the wrapper (kill it to stop the job), NAME.child the command.
    local name=$1 cores=$2 threads=$3 heavy=$4; shift 4
    [ "$heavy" = 1 ] && touch "$PIDS/$name.heavy"; rm -f "$PIDS/$name.pid" "$PIDS/$name.child"; echo "$_HOST" > "$PIDS/$name.host"; echo running > "$PIDS/$name.state"
    OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads JOB_KILL_GRACE=${JOB_KILL_GRACE:-60} setsid nohup bash -c '
        cores=$1; state=$2; child=$3; pidfile=$4; shift 4; echo $$ > "$pidfile"
        taskset -c "$cores" "$@" & c=$!; echo $c > "$child"; term=0
        trap "term=1; kill $c 2>/dev/null" TERM INT
        wait $c; s=$?
        # a signal interrupts wait: keep waiting until the child is really gone; escalate to KILL after the grace period
        t=0; while kill -0 $c 2>/dev/null; do sleep 1; t=$((t+1)); if [ $term = 1 ] && [ $t -ge "$JOB_KILL_GRACE" ]; then kill -KILL $c 2>/dev/null; fi; done
        wait $c 2>/dev/null; s2=$?; [ $s -gt 128 ] && s=$s2
        # the workload is finished only when nothing of this session is left: descendants of the command get TERM, then KILL
        # after the grace period; the state (and with it the heavy slot) is released only afterwards
        trap "" TERM
        rest() { local p out=""; for p in $(pgrep -s $$); do [ "$p" = "$$" ] || [ "$p" = "$BASHPID" ] && continue; kill -0 "$p" 2>/dev/null && out="$out $p"; done; echo $out; }   # session members other than this shell and the scanning subshell
        if [ -n "$(rest)" ]; then kill -TERM $(rest) 2>/dev/null; fi
        t=0; while [ -n "$(rest)" ]; do sleep 1; t=$((t+1)); if [ $t -ge "$JOB_KILL_GRACE" ]; then kill -KILL $(rest) 2>/dev/null; fi; done
        if [ $s -eq 0 ]; then echo succeeded > "$state"; else echo "failed $s" > "$state"; fi; exit $s' _ "$cores" "$PIDS/$name.state" "$PIDS/$name.child" "$PIDS/$name.pid" "$@" > "$LOGS/$name.log" 2>&1 9>&- &
    disown
    local i=0; while [ ! -s "$PIDS/$name.pid" ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done
    [ -s "$PIDS/$name.pid" ] || { echo "[jobs.sh] ERROR: $name did not start (no pid recorded)"; echo "failed launch" > "$PIDS/$name.state"; return 1; }
    echo "[jobs.sh] started $name on $_HOST (pid $(cat "$PIDS/$name.pid"), cores $cores, $threads threads, heavy=$heavy, declared ${JOB_MEM_GB:-default} GB) $(date +%H:%M)"
}
run_heavy() { local name=$1 cores=$2; shift 2; _admit "$name" "$cores" "${HEAVY_THREADS:-16}" 1 "${JOB_MEM_GB:-30}" "$@"; }   # HEAVY_THREADS: wider on a larger host (bourbon: 32 with MAX_HEAVY=4)
run_light() { local name=$1 cores=$2; shift 2; _admit "$name" "$cores" 8 0 "${JOB_MEM_GB:-10}" "$@"; }
queue_begin() {                 # NAME: a queue script announces itself (results/pids/queue_NAME.state = running, queue_NAME.pid = its shell); call at the top
    local name=$1; echo $$ > "$PIDS/queue_$name.pid"; echo running > "$PIDS/queue_$name.state"
    trap 'rc=$?; if [ $rc -eq 0 ]; then echo succeeded > "'"$PIDS"'/queue_'"$name"'.state"; else echo "failed $rc" > "'"$PIDS"'/queue_'"$name"'.state"; fi' EXIT
}
queue_state() {                 # NAME -> running | succeeded | failed N | unknown ; a queue whose shell died without writing a state is 'failed lost'
    local name=$1 st p
    st=$(cat "$PIDS/queue_$name.state" 2>/dev/null) || { echo unknown; return; }
    if [ "$st" = running ]; then p=$(cat "$PIDS/queue_$name.pid" 2>/dev/null); if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo running; else echo "failed lost"; fi; else echo "$st"; fi
}
wait_queue() {                  # NAME...: block until every named queue has ended; 1 if any is unknown, lost or failed (successor queues must `wait_queue X || exit 1`)
    local rc=0 name st
    for name in "$@"; do
        st=$(queue_state "$name")
        if [ "$st" = unknown ]; then echo "[jobs.sh] ERROR: no queue named $name announced itself $(date +%H:%M)"; rc=1; continue; fi
        while [ "$(queue_state "$name")" = running ]; do sleep "${JOB_POLL_S:-60}"; done
        st=$(queue_state "$name")
        case "$st" in succeeded) echo "[jobs.sh] queue $name succeeded $(date +%H:%M)";; *) echo "[jobs.sh] ERROR: queue $name $st $(date +%H:%M)"; rc=1;; esac
    done
    return $rc
}
wait_process_log() {            # PID LOG DONE_TEXT: for a predecessor started without queue_begin: success when LOG contains DONE_TEXT, failure when PID is gone without it
    local pid=$1 log=$2 text=$3
    while true; do
        grep -q -- "$text" "$log" 2>/dev/null && return 0
        kill -0 "$pid" 2>/dev/null || { echo "[jobs.sh] ERROR: process $pid ended without '$text' in $log $(date +%H:%M)"; return 1; }
        sleep "${JOB_POLL_S:-60}"
    done
}
job_state() {                   # NAME -> running | succeeded | failed N | unknown | foreign HOST ; a job whose wrapper died without writing a state is 'failed lost'
    local name=$1 st p h
    st=$(cat "$PIDS/$name.state" 2>/dev/null) || { echo unknown; return; }
    h=$(cat "$PIDS/$name.host" 2>/dev/null); if [ -n "$h" ] && [ "$h" != "$_HOST" ] && [ "$st" = running ]; then echo "foreign $h"; return; fi   # liveness cannot be checked across hosts
    if [ "$st" = running ]; then p=$(cat "$PIDS/$name.pid" 2>/dev/null); if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo running; else echo "failed lost"; fi; else echo "$st"; fi
}
wait_pidfiles() {               # block until every named job has finished; returns 1 if any is unknown or failed (chains must stop on that)
    local rc=0
    for name in "$@"; do
        local st; st=$(job_state "$name")
        if [ "$st" = unknown ]; then echo "[jobs.sh] ERROR: no job named $name was launched by this launcher $(date +%H:%M)"; rc=1; continue; fi
        case "$st" in foreign*) echo "[jobs.sh] ERROR: $name runs on another host ($st); wait for it there $(date +%H:%M)"; rc=1; continue;; esac
        while [ "$(job_state "$name")" = running ]; do sleep 60; done
        st=$(job_state "$name"); rm -f "$PIDS/$name.heavy"
        case "$st" in succeeded) echo "[jobs.sh] $name succeeded $(date +%H:%M)";; *) echo "[jobs.sh] ERROR: $name $st $(date +%H:%M)"; rc=1;; esac
    done
    return $rc
}
