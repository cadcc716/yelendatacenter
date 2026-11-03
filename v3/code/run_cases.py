"""
批量运行 S1–S7（执行方案 5.2 节），每个场景一个进程，结果存 results/<site>_<tag>.pkl。
用法：python run_cases.py <site> [tag ...]      tag ∈ S1 … S7
"""
import os, sys, subprocess, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = {   # tag: (场景开关, 不确定性方法)
    "S1": ("S1", "DRO"), "S2": ("S2", "DRO"), "S3": ("S3", "DRO"), "S4": ("S4", "DRO"),
    "S5": ("S5", "DRO"), "S6": ("S5", "SP"), "S7": ("S5", "RO"),
}

if __name__ == "__main__":
    site = sys.argv[1]; tags = sys.argv[2:] or list(JOBS)
    par = int(os.environ.get("PAR", 4)); env = dict(os.environ, HIGHS_THREADS=os.environ.get("HIGHS_THREADS", "1"))
    queue = list(tags); run = {}
    while queue or run:
        while queue and len(run) < par:
            t = queue.pop(0); scen, meth = JOBS[t]
            log = open(f"{ROOT}/results/log_{site}_{t}.txt", "w")
            run[t] = subprocess.Popen([sys.executable, "-u", "ccg_solve.py", site, scen, f"{meth}_{t}"], stdout=log, stderr=log, env=env,
                                      cwd=os.path.dirname(os.path.abspath(__file__)))
        for t, p in list(run.items()):
            if p.poll() is not None:
                print(f"{t} 完成，返回码 {p.returncode}", flush=True); run.pop(t)
        time.sleep(5)
