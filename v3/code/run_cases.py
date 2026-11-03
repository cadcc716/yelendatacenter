"""
批量运行 S1–S7（执行方案 5.2 节）：每个场景分别用 C&CG 与等价单层 MILP(40) 求解，每个任务一个进程。
结果存 results/<site>_<tag>.pkl 与 results/<site>_<tag>_mono.pkl。
用法：python run_cases.py <site> [任务 ...]     任务形如 S5 或 S5:mono；缺省为全部 14 个任务
环境变量 PARAM_OVER（JSON）传入参数覆盖，PAR 为并行进程数。
"""
import os, sys, subprocess, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = {   # tag: (场景开关, 不确定性方法)
    "S1": ("S1", "DRO"), "S2": ("S2", "DRO"), "S3": ("S3", "DRO"), "S4": ("S4", "DRO"),
    "S5": ("S5", "DRO"), "S6": ("S5", "SP"), "S7": ("S5", "RO"),
}

if __name__ == "__main__":
    site = sys.argv[1]
    tasks = sys.argv[2:] or [f"{t}:{m}" for t in JOBS for m in ("ccg", "mono")]
    par = int(os.environ.get("PAR", 4)); env = dict(os.environ, HIGHS_THREADS=os.environ.get("HIGHS_THREADS", "1"))
    queue = list(tasks); run = {}
    while queue or run:
        while queue and len(run) < par:
            t = queue.pop(0); tag, meth = (t.split(":") + ["ccg"])[:2]; scen, unc = JOBS[tag]
            log = open(f"{ROOT}/results/log_{site}_{tag}_{meth}.txt", "w")
            run[t] = subprocess.Popen([sys.executable, "-u", "ccg_solve.py", site, scen, f"{unc}_{tag}", meth], stdout=log, stderr=log,
                                      env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
        for t, p in list(run.items()):
            if p.poll() is not None:
                print(f"{t} 完成，返回码 {p.returncode}", flush=True); run.pop(t)
        time.sleep(5)
