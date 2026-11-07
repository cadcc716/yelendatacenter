"""
剩余计算的统一调度（可续跑）：灵敏度（α1、热价、芯片温限、M）与气候区（贵安、韶关）。
- 已有输出文件的任务自动跳过；任务按依赖顺序执行（气候区 S1 → S3 → S4 → S5 嵌套热启动）
- 并行数 = 4 − 正在运行的极端日试验进程数（至少 1），极端日试验结束后自动用满 4 核
- 容器重启后重新运行：python orchestrate.py
日志：results/log_orchestrate.txt；各任务日志在 results/logs/
"""
import os, sys, time, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CODE = os.path.dirname(os.path.abspath(__file__))
RES = f"{ROOT}/results"; LOGS = f"{RES}/logs"; os.makedirs(LOGS, exist_ok=True)
BASE_ENV = dict(HIGHS_THREADS="1", STEP1="mono", GAP1="3e-3", TL1="900", GAP2="1e-3", PAR_LS="1", LS_MAX="30")


def task(name, out, args, env=None, deps=()):
    return dict(name=name, out=out, args=args, env={**BASE_ENV, **(env or {})}, deps=list(deps))


T = []
# ---------- 灵敏度：S5（DRO），两步法（单层 MILP）+ 邻域搜索；基准点用同一设置单独计算 ----------
sens = [("base", None, None)] + [("alpha", "ALPHA1", v) for v in (0.2, 0.4, 0.7, 0.9, 0.99)] + \
       [("heat", "C_HEAT", v) for v in (0.1, 0.15, 0.25, 0.3)] + [("chip", "T_CHIP_MAX", v) for v in (80.0, 83.0, 87.0, 90.0)] + \
       [("M", "M", v) for v in (90, 180, 730)]
for g, k, v in sens:
    nm = "base" if g == "base" else f"{g}_{v}"
    out = f"{RES}/zhongwei_sens_{nm}.pkl"
    over = {} if k is None else {k: v}
    T.append(task(f"sens_{nm}", out, ["zhongwei", "S5", "DRO_sens", "mono", "-"], dict(TL2="1500", OUT=out, PARAM_OVER=json.dumps(over))))
# ---------- GPU 冗余度：S5（DRO）固定 N = 6000、8000（检验降档与高温运行机理） ----------
for n in (6000, 8000):
    out = f"{RES}/zhongwei_fixN_{n}.pkl"
    T.append(task(f"fixN_{n}", out, ["zhongwei", "S5", "DRO_fixN", "mono", "-"], dict(TL2="1500", OUT=out, FIXN=str(n))))
# ---------- 气候区：贵安、韶关 S1、S3→S4→S5（C&CG，与中卫主算例同一设置） ----------
for site in ("guian", "shaoguan"):
    e = dict(TL2="1800")
    T.append(task(f"{site}_S1", f"{RES}/{site}_DRO_S1.pkl", [site, "S1", "DRO_S1", "ccg", "-"], e))
    T.append(task(f"{site}_S3", f"{RES}/{site}_DRO_S3.pkl", [site, "S3", "DRO_S3", "ccg", "-"], e))
    T.append(task(f"{site}_S4", f"{RES}/{site}_DRO_S4.pkl", [site, "S4", "DRO_S4", "ccg", f"{RES}/{site}_DRO_S3.pkl"], e, [f"{site}_S3"]))
    T.append(task(f"{site}_S5", f"{RES}/{site}_DRO_S5.pkl", [site, "S5", "DRO_S5", "ccg", f"{RES}/{site}_DRO_S4.pkl"], e, [f"{site}_S4"]))


def log(msg):
    line = time.strftime("%m-%d %H:%M:%S ") + msg
    print(line, flush=True); open(f"{RES}/log_orchestrate.txt", "a").write(line + "\n")


def ext_running():
    r = subprocess.run(["pgrep", "-f", "ccg_solve.py zhongwei_ext|oos_test.py zhongwei"], capture_output=True, text=True)
    return len([x for x in r.stdout.split() if x.strip()])


def alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False


class Ext:                                   # 调度器重启前已在运行的任务（按 pid 文件识别）
    def __init__(s, pid): s.pid = pid; s.returncode = None
    def poll(s): return None if alive(s.pid) else 0


if __name__ == "__main__":
    done = {t["name"] for t in T if os.path.exists(t["out"])}; fail = {}; run = {}
    for t in T:
        pf = f"{LOGS}/{t['name']}.pid"
        if t["name"] not in done and os.path.exists(pf):
            pid = int(open(pf).read().strip() or 0)
            if pid and alive(pid): run[t["name"]] = Ext(pid); log(f"接管运行中的任务 {t['name']}（pid {pid}）")
    log(f"启动：共 {len(T)} 个任务，已完成 {len(done)} 个")
    while True:
        for nm, p in list(run.items()):
            if p.poll() is not None:
                t = next(x for x in T if x["name"] == nm); run.pop(nm)
                if os.path.exists(t["out"]): done.add(nm); log(f"完成 {nm}")
                else:
                    fail[nm] = fail.get(nm, 0) + 1; log(f"失败 {nm}（返回码 {p.returncode}，第 {fail[nm]} 次）")
        todo = [t for t in T if t["name"] not in done and t["name"] not in run and fail.get(t["name"], 0) < 2
                and all(d in done for d in t["deps"])]
        if not todo and not run: break
        par = max(1, 4 - ext_running())
        while todo and len(run) < par:
            t = todo.pop(0); env = {**os.environ, **t["env"]}
            lf = open(f"{LOGS}/{t['name']}.txt", "a")
            run[t["name"]] = subprocess.Popen([sys.executable, "-u", "ccg_solve.py", *t["args"]], stdout=lf, stderr=lf, env=env, cwd=CODE)
            open(f"{LOGS}/{t['name']}.pid", "w").write(str(run[t["name"]].pid))
            log(f"开始 {t['name']}（并行 {len(run)}/{par}）")
        time.sleep(20)
    log(f"全部结束：完成 {len(done)}，失败 {[k for k, v in fail.items() if v >= 2]}")
