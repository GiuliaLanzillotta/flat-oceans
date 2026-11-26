"""The launcher (scripts/jobs.sh): PID files name the wrapper, states are recorded, a killed job is not reported finished
(and its heavy slot not released) until the command and every descendant are really gone, a process that ignores TERM
is killed after the grace period, and waiting on an unknown or failed job returns an error.
All timing assertions poll with a bound and report the job's files on timeout."""
import os, shutil, subprocess, time

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRACE = 3


def sh(cwd, script, timeout=60):
    return subprocess.run(["bash", "-c", "source scripts/jobs.sh; " + script], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          env=dict(os.environ, JOB_KILL_GRACE=str(GRACE), MAX_LOAD="100000", HEADROOM_GB="0", JOB_MEM_GB="0"))


@pytest.fixture
def jt(tmp_path):
    (tmp_path / "scripts").mkdir(); shutil.copy(os.path.join(HERE, "scripts", "jobs.sh"), tmp_path / "scripts" / "jobs.sh"); return str(tmp_path)


def pidfile(jt, name, kind):
    return open(os.path.join(jt, "results", "pids", f"{name}.{kind}")).read().strip()


def state(jt, name):
    try:
        return pidfile(jt, name, "state")
    except FileNotFoundError:
        return "unknown"


def alive(pid):
    try:
        os.kill(int(pid), 0); return True
    except ProcessLookupError:
        return False


def heavy_running(jt):
    return int(sh(jt, "_heavy_running").stdout.strip())


def diagnostics(jt, name):
    d = os.path.join(jt, "results", "pids"); out = [f"job {name}: state {state(jt, name)}"]
    for k in ("pid", "child"):
        try:
            p = pidfile(jt, name, k); out.append(f"  {k} {p} alive={alive(p)}")
        except FileNotFoundError:
            out.append(f"  {k}: no file")
    log = os.path.join(jt, "results", "logs", f"{name}.log")
    if os.path.exists(log):
        out.append("  log tail: " + open(log).read()[-300:].replace("\n", " | "))
    return "\n".join(out)


def wait_until(cond, timeout, jt, name, what):
    """Poll `cond` every 0.2 s up to `timeout` seconds; on timeout fail with the job's files as diagnostics."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return time.time() - t0
        time.sleep(0.2)
    pytest.fail(f"timeout after {timeout}s waiting for: {what}\n{diagnostics(jt, name)}")


def test_states_and_failure_aware_wait(jt):
    r = sh(jt, 'run_light ok 0-1 bash -c "exit 0"; run_light bad 0-1 bash -c "exit 3"; wait_pidfiles ok bad none; echo rc=$?')
    assert "rc=1" in r.stdout and "ERROR: bad failed 3" in r.stdout and "no job named none" in r.stdout, r.stdout + r.stderr
    assert state(jt, "ok") == "succeeded" and state(jt, "bad") == "failed 3"


def test_pid_is_the_wrapper_and_kill_stops_the_child(jt):
    sh(jt, "run_light k 0-1 sleep 300")
    pid, child = pidfile(jt, "k", "pid"), pidfile(jt, "k", "child")
    assert subprocess.run(["ps", "-o", "comm=", "-p", pid], capture_output=True, text=True).stdout.strip() == "bash"
    os.kill(int(pid), 15)
    wait_until(lambda: not alive(child) and state(jt, "k").startswith("failed"), 10, jt, "k", "child gone and state failed after TERM to the wrapper")
    assert not alive(pid)


def test_child_ignoring_term_is_killed_after_grace_and_slot_kept(jt):
    sh(jt, 'run_heavy ign 0-1 bash -c "trap \\"\\" TERM; sleep 300"')
    pid, child = pidfile(jt, "ign", "pid"), pidfile(jt, "ign", "child")
    os.kill(int(pid), 15); time.sleep(1)
    assert alive(child) and state(jt, "ign") == "running" and alive(pid), diagnostics(jt, "ign")      # the wrapper waits
    assert heavy_running(jt) == 1, diagnostics(jt, "ign")                                                # the heavy slot is still taken
    t = wait_until(lambda: not alive(child), GRACE + 10, jt, "ign", "TERM-ignoring child killed after the grace period")
    assert t >= GRACE - 1.5, f"child died after {t:.1f}s, before the grace period"
    wait_until(lambda: not alive(pid) and state(jt, "ign").startswith("failed"), 10, jt, "ign", "wrapper exit and failed state")
    assert heavy_running(jt) == 0


def test_child_exiting_late_is_waited_for(jt):
    # bash runs a trap only after the foreground command returns, so the child backgrounds its sleep and waits (interruptible)
    sh(jt, 'run_light slow 0-1 bash -c "trap \\"sleep 1; exit 7\\" TERM; sleep 300 & wait"')
    pid = pidfile(jt, "slow", "pid")
    os.kill(int(pid), 15); time.sleep(0.3)
    assert state(jt, "slow") == "running", diagnostics(jt, "slow")
    wait_until(lambda: state(jt, "slow") == "failed 7", 10, jt, "slow", "the child's own exit status 7 after its delayed exit")
    assert not alive(pid)


def test_descendant_ignoring_term_keeps_state_running_until_killed(jt):
    """The command exits 0 but leaves a grandchild that ignores TERM: the state stays 'running' (heavy slot kept) until that
    specific grandchild is killed after the grace period."""
    gpid = os.path.join(jt, "grandchild.pid")
    sh(jt, f'run_heavy gc 0-1 bash -c "bash -c \\"echo \\\\\\$\\\\\\$ > {gpid}; trap \\\\\\"\\\\\\" TERM; sleep 300\\" & exit 0"')
    wait_until(lambda: os.path.exists(gpid) and open(gpid).read().strip(), 5, jt, "gc", "grandchild pid file")
    g = open(gpid).read().strip(); pid, child = pidfile(jt, "gc", "pid"), pidfile(jt, "gc", "child")
    wait_until(lambda: not alive(child), 5, jt, "gc", "the command itself exiting 0")
    assert alive(g) and state(jt, "gc") == "running" and alive(pid) and heavy_running(jt) == 1, diagnostics(jt, "gc") + f"\n  grandchild {g} alive={alive(g)}"
    t = wait_until(lambda: not alive(g), GRACE + 10, jt, "gc", f"grandchild {g} killed after the grace period")
    assert t >= GRACE - 1.5, f"grandchild died after {t:.1f}s, before the grace period"
    wait_until(lambda: state(jt, "gc") == "succeeded" and not alive(pid), 10, jt, "gc", "state recorded only after the grandchild is gone")
    assert heavy_running(jt) == 0


def test_admission_is_serialized_and_declared_memory_is_reserved(jt):
    """Two heavy launches racing for one heavy slot: the lock lets exactly one in, the other starts only after the first has
    finished; the declared memory of a running job counts as reserved until its RSS reaches it."""
    r = sh(jt, 'export MAX_HEAVY=1 JOB_POLL_S=1; (run_heavy a 0-1 sleep 3 &) ; (run_heavy b 0-1 sleep 3 &); sleep 1.2; echo heavy=$(_heavy_running); '
                'JOB_MEM_GB=7 run_light m 0-1 sleep 4; sleep 0.5; echo reserved=$(_reserved_gb); wait_pidfiles a b m; echo reserved_after=$(_reserved_gb); echo rc=$?', timeout=90)
    assert "heavy=1" in r.stdout, r.stdout + r.stderr
    assert "reserved=7" in r.stdout and "reserved_after=0" in r.stdout, r.stdout + r.stderr
    assert state(jt, "a") == "succeeded" and state(jt, "b") == "succeeded" and state(jt, "m") == "succeeded"
    sa = os.path.getmtime(os.path.join(jt, "results", "pids", "a.pid")); sb = os.path.getmtime(os.path.join(jt, "results", "pids", "b.pid"))
    assert abs(sa - sb) >= 2.5, "the second heavy job started before the first had finished"


def test_two_permitted_jobs_overlap_and_the_lock_is_released(jt):
    """Two light jobs with capacity for both start within a second of each other and run at the same time: the detached
    wrapper does not keep the admission lock (regression of 2026-09-25)."""
    r = sh(jt, 'export JOB_POLL_S=1; run_light a 0-1 sleep 4; run_light b 0-1 sleep 4; sleep 0.5; echo running=$(cat results/pids/a.state)/$(cat results/pids/b.state); '
                'flock -w 2 results/pids/.admission.lock true && echo lock=free || echo lock=held; wait_pidfiles a b', timeout=150)      # wait_pidfiles polls every 60 s
    assert "running=running/running" in r.stdout and "lock=free" in r.stdout, r.stdout + r.stderr
    sa = os.path.getmtime(os.path.join(jt, "results", "pids", "a.pid")); sb = os.path.getmtime(os.path.join(jt, "results", "pids", "b.pid"))
    assert abs(sa - sb) < 1.5, f"second job delayed by {abs(sa - sb):.1f} s"


def test_queue_states_are_tracked_and_successors_fail_on_a_dead_predecessor(jt):
    """A queue announces itself and its exit writes succeeded / failed N; a successor waiting on it gets 0 only for success;
    a queue killed before its end is reported lost; an unannounced predecessor that dies without its done text fails the wait."""
    r = sh(jt, 'bash -c "source scripts/jobs.sh; queue_begin ok; exit 0"; bash -c "source scripts/jobs.sh; queue_begin bad; exit 7"; echo ok=$(queue_state ok) bad=$(queue_state bad); '
                'wait_queue ok; echo w_ok=$?; wait_queue bad; echo w_bad=$?; wait_queue none; echo w_none=$?', timeout=60)
    assert "ok=succeeded bad=failed 7" in r.stdout and "w_ok=0" in r.stdout and "w_bad=1" in r.stdout and "w_none=1" in r.stdout, r.stdout + r.stderr
    r = sh(jt, 'export JOB_POLL_S=1; bash -c "source scripts/jobs.sh; queue_begin k; sleep 30" >/dev/null 2>&1 & sleep 1; kill -9 $(cat results/pids/queue_k.pid); sleep 1; echo k=$(queue_state k); '
                'sleep 3 >/dev/null 2>&1 & p=$!; echo x > l.log; wait_process_log $p l.log DONE; echo wpl=$?; (sleep 1; echo DONE >> l.log) >/dev/null 2>&1 & q=$!; wait_process_log $q l.log DONE; echo wpl2=$?', timeout=60)
    assert "k=failed lost" in r.stdout and "wpl=1" in r.stdout and "wpl2=0" in r.stdout, r.stdout + r.stderr


def test_pid_directory_is_per_host_and_foreign_jobs_are_not_judged(jt):
    """The first host owns results/pids and results/logs (recorded in .host); another host gets results/pids/<host> and results/logs/<host>; JOB_PIDDIR overrides; a job
    recorded by another host is reported 'foreign' and wait_pidfiles refuses to wait for it."""
    r = sh(jt, "echo $PIDS; cat results/pids/.host")
    pids, host = r.stdout.split()
    assert pids == os.path.join(jt, "results", "pids") and host == os.uname().nodename.split(".")[0]
    r = sh(jt, "export JOB_POLL_S=1; run_light a 0-1 true; wait_pidfiles a; cat results/pids/a.host")
    assert r.stdout.strip().endswith(host)
    open(os.path.join(jt, "results", "pids", ".host"), "w").write("otherhost\n")
    r = sh(jt, "echo $PIDS $LOGS; run_light lg 0-1 echo hello; wait_pidfiles lg")
    assert r.stdout.split()[:2] == [os.path.join(jt, "results", "pids", host), os.path.join(jt, "results", "logs", host)]
    assert open(os.path.join(jt, "results", "logs", host, "lg.log")).read().strip() == "hello" and not os.path.exists(os.path.join(jt, "results", "logs", "lg.log"))
    r = subprocess.run(["bash", "-c", "source scripts/jobs.sh; echo $PIDS; run_light b 0-1 true; wait_pidfiles b; ls elsewhere"], cwd=jt, capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, JOB_KILL_GRACE=str(GRACE), MAX_LOAD="100000", HEADROOM_GB="0", JOB_MEM_GB="0", JOB_POLL_S="1", JOB_PIDDIR=os.path.join(jt, "elsewhere")))
    assert r.stdout.split()[0] == os.path.join(jt, "elsewhere") and "b.state" in r.stdout
    os.makedirs(os.path.join(jt, "results", "pids"), exist_ok=True)
    open(os.path.join(jt, "results", "pids", ".host"), "w").write(host + "\n")
    open(os.path.join(jt, "results", "pids", "c.state"), "w").write("running\n"); open(os.path.join(jt, "results", "pids", "c.host"), "w").write("otherhost\n")
    r = sh(jt, "job_state c; wait_pidfiles c; echo rc=$?")
    assert r.stdout.split("\n")[0] == "foreign otherhost" and "rc=1" in r.stdout and "another host" in r.stdout
