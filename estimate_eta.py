"""
Estimate ETA using HISTORICAL data (checkpoint timestamps / process start time)
instead of live sampling -- works even when epochs take longer than a short
sampling window.
"""
import subprocess
import os
import glob
import re
import time

V30_CKPT_DIR = 'checkpoints/phase1_lidar_v30'
PHASE2_LOG = '/tmp/phase2_v29_win100_restart.log'
PHASE2_PID = None  # auto-detected below

V30_TOTAL_TIMESTEPS = 40_000_000
V30_ROLLOUT_STEPS = 34_000
V30_TOTAL_EPOCHS = V30_TOTAL_TIMESTEPS // V30_ROLLOUT_STEPS
PHASE2_TOTAL_EPOCHS = 100


def analyze_v30():
    print("=== v30 (Phase 1, 40M steps) ===")
    ckpts = glob.glob(f'{V30_CKPT_DIR}/checkpoint_epoch_*.pt')
    if not ckpts:
        print("No checkpoints yet -- too early to estimate")
        return

    def epoch_num(path):
        m = re.search(r'checkpoint_epoch_(\d+)\.pt', path)
        return int(m.group(1)) if m else -1

    ckpts_with_epoch = [(epoch_num(c), os.path.getmtime(c)) for c in ckpts]
    ckpts_with_epoch.sort()

    first_epoch, first_time = ckpts_with_epoch[0]
    last_epoch, last_time = ckpts_with_epoch[-1]

    print(f"Earliest checkpoint: epoch {first_epoch} at {time.ctime(first_time)}")
    print(f"Latest checkpoint:   epoch {last_epoch} at {time.ctime(last_time)}")
    print(f"Current progress: {last_epoch}/{V30_TOTAL_EPOCHS} "
          f"({last_epoch/V30_TOTAL_EPOCHS*100:.1f}%)")

    if last_epoch == first_epoch:
        print("Only one checkpoint so far -- cannot compute rate yet")
        return

    elapsed = last_time - first_time
    epochs_done = last_epoch - first_epoch
    rate = elapsed / epochs_done
    remaining = V30_TOTAL_EPOCHS - last_epoch
    eta_hours = remaining * rate / 3600

    print(f"Rate: {rate:.1f} sec/epoch (measured over {epochs_done} epochs, {elapsed/60:.1f} min)")
    print(f"Remaining epochs: {remaining}")
    print(f"ETA: {eta_hours:.1f} hours ({eta_hours/24:.1f} days)")

    # Check for exponential/nonlinear trend using 3+ checkpoints if available
    if len(ckpts_with_epoch) >= 4:
        mid = len(ckpts_with_epoch) // 2
        e1, t1 = ckpts_with_epoch[0]
        e2, t2 = ckpts_with_epoch[mid]
        e3, t3 = ckpts_with_epoch[-1]
        rate_first_half = (t2 - t1) / max(e2 - e1, 1)
        rate_second_half = (t3 - t2) / max(e3 - e2, 1)
        pct_change = (rate_second_half - rate_first_half) / rate_first_half * 100 if rate_first_half > 0 else 0
        print(f"Rate trend: first-half={rate_first_half:.1f}s/epoch, "
              f"second-half={rate_second_half:.1f}s/epoch ({pct_change:+.0f}%)")
        if abs(pct_change) > 15:
            trend_eta = remaining * rate_second_half / 3600
            print(f"Trend-adjusted ETA (using recent rate only): {trend_eta:.1f} hours")


def find_phase2_pid():
    result = subprocess.run(['pgrep', '-f', 'phase2_adaptation'], capture_output=True, text=True)
    pids = result.stdout.strip().split('\n')
    # filter to the actual python process (not the bash -c wrapper)
    for pid in pids:
        if pid:
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as f:
                    cmdline = f.read().decode(errors='ignore')
                if 'python3' in cmdline and 'phase2_adaptation' in cmdline:
                    return int(pid)
            except FileNotFoundError:
                continue
    return None


def analyze_phase2():
    print("\n=== Phase 2 (v29, 100-step window) ===")
    result = subprocess.run(['grep', '-oP', r'(?<=\[Epoch )\d+(?=\] Loss)', PHASE2_LOG],
                             capture_output=True, text=True)
    epochs_seen = [int(x) for x in result.stdout.strip().split('\n') if x]
    if not epochs_seen:
        print("No epoch data found in log")
        return
    current_epoch = max(epochs_seen) + 1  # epochs are 0-indexed
    print(f"Current progress: {current_epoch}/{PHASE2_TOTAL_EPOCHS} "
          f"({current_epoch/PHASE2_TOTAL_EPOCHS*100:.1f}%)")

    pid = find_phase2_pid()
    if pid is None:
        print("Could not find running phase2 process to get start time")
        return

    # Get process start time using /proc/<pid>/stat (field 22, start time in clock ticks since boot)
    result = subprocess.run(['ps', '-o', 'etimes=', '-p', str(pid)], capture_output=True, text=True)
    try:
        elapsed_sec = int(result.stdout.strip())
    except ValueError:
        print(f"Could not parse process elapsed time from: {result.stdout!r}")
        return

    print(f"Process running for: {elapsed_sec/60:.1f} min ({elapsed_sec/3600:.2f} hours)")

    # NOTE: this elapsed time includes data collection phase, not just training epochs.
    # We only know when epoch training STARTED via the log's "Data Collection Complete" line,
    # but we don't have wall-clock timestamps in the log itself. Best estimate: assume
    # data collection + epochs both happened within this elapsed window.
    if current_epoch > 0:
        rate = elapsed_sec / current_epoch  # includes data collection overhead, so this UNDERSTATES true epoch rate
        remaining = PHASE2_TOTAL_EPOCHS - current_epoch
        eta_sec = remaining * rate
        print(f"Rough rate (includes data collection overhead): {rate:.1f} sec/epoch")
        print(f"ETA (conservative, likely overestimate since data collection is one-time): "
              f"{eta_sec/3600:.1f} hours")
        print(f"NOTE: actual per-epoch rate is faster than this since data collection")
        print(f"      (a one-time ~{elapsed_sec/current_epoch*0.3:.0f}s cost) is being averaged in")


if __name__ == '__main__':
    analyze_v30()
    analyze_phase2()
