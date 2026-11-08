"""
v4 全部计算的统一调度（可续跑）。算例假设修订：算力保障约束 R_N = 0.2、采暖季热负荷 ×3（峰值 9 MW_th），其余同 v3。
任务：中卫 S1–S7（C&CG 两步法 + 邻域搜索，嵌套热启动）、S1–S7 单层 MILP 校验、表F、典型日导出、
      极端日 SP/DRO/RO（C&CG）与 365 日样本外测试、灵敏度（α1、热价、芯片温限、M、冗余率 R_N）、贵安与韶关 S1、S3–S5。
已有输出文件的任务自动跳过；容器重启后重新运行：python orchestrate_v4.py
日志：results/log_v4.txt；各任务日志在 results/logs/
"""
import os, sys, time, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CODE = os.path.dirname(os.path.abspath(__file__))
RES = f"{ROOT}/results"; LOGS = f"{RES}/logs"; X2 = f"{RES}/ext2"; os.makedirs(LOGS, exist_ok=True); os.makedirs(X2, exist_ok=True)
PY = sys.executable
B = dict(HIGHS_THREADS="1", STEP1="mono", GAP1="3e-3", TL1="900", GAP2="1e-3", TL2="1800", EPS="1e-3", PAR_LS="1", LS_MAX="40")
T = []


def task(name, out, cmd, env=None, deps=()):
    T.append(dict(name=name, out=out, cmd=cmd, env={**B, **(env or {})}, deps=list(deps)))


def solve(site, scen, tag, method="ccg", prev="-", out=None, env=None, deps=(), name=None):
    out = out or f"{RES}/{site}_{tag}{'_mono' if method == 'mono' else ''}.pkl"
    task(name or f"{site}_{tag}_{method}", out, [PY, "-u", "ccg_solve.py", site, scen, tag, method, prev], dict(env or {}, OUT=out), deps)


S = "zhongwei"; P = lambda t, m="": f"{RES}/{S}_{t}{m}.pkl"
# ---------- 中卫 S1–S7 ----------
solve(S, "S1", "DRO_S1"); solve(S, "S2", "DRO_S2", prev=P("DRO_S1"), deps=["zhongwei_DRO_S1_ccg"])
solve(S, "S3", "DRO_S3"); solve(S, "S4", "DRO_S4", prev=P("DRO_S3"), deps=["zhongwei_DRO_S3_ccg"])
solve(S, "S5", "DRO_S5", prev=P("DRO_S4"), deps=["zhongwei_DRO_S4_ccg"])
solve(S, "S5", "SP_S6"); solve(S, "S5", "RO_S7")
# ---------- 单层 MILP 校验（不做邻域搜索） ----------
mono = dict(LS="0")
solve(S, "S1", "DRO_S1", "mono", env=mono); solve(S, "S2", "DRO_S2", "mono", P("DRO_S1", "_mono"), env=mono, deps=["zhongwei_DRO_S1_mono"])
solve(S, "S3", "DRO_S3", "mono", env=mono); solve(S, "S4", "DRO_S4", "mono", P("DRO_S3", "_mono"), env=mono, deps=["zhongwei_DRO_S3_mono"])
solve(S, "S5", "DRO_S5", "mono", P("DRO_S4", "_mono"), env=mono, deps=["zhongwei_DRO_S4_mono"])
solve(S, "S5", "SP_S6", "mono", env=mono); solve(S, "S5", "RO_S7", "mono", env=mono)
# ---------- 表F、典型日导出 ----------
task("tableF", f"{RES}/{S}_tableF.csv", [PY, "-u", "table_f.py", S], dict(PAR="1"), ["zhongwei_DRO_S1_ccg", "zhongwei_DRO_S2_ccg", "zhongwei_DRO_S5_ccg"])
task("export_S5", f"{RES}/fig_data/{S}_2023-06-15_逐时.csv", [PY, "-u", "export_days.py", S, P("DRO_S5")], {}, ["zhongwei_DRO_S5_ccg"])
for t in ("DRO_S1", "DRO_S3", "DRO_S4"):                     # 基准与中间场景典型日，用于 Baseline 对比图
    task(f"export_{t}", f"{RES}/fig_data_{t}/{S}_2023-06-15_逐时.csv", [PY, "-u", "export_days.py", S, P(t), "2023-11-22"],
         dict(OUTDIR=f"{RES}/fig_data_{t}"), [f"zhongwei_{t}_ccg"])
# ---------- 极端日 ----------
for t in ("SP_S6", "DRO_S5", "RO_S7"):
    out = f"{X2}/{S}_ext_{t}.pkl"
    solve("zhongwei_ext", "S5", t, out=out, env=dict(PARAM_OVER=json.dumps({"K": 16})), name=f"ext_{t}")
    task(f"oos_{t}", f"{X2}/{S}_ext_oos_{t}.csv", [PY, "-u", "oos_test.py", S, t], dict(SRC="zhongwei_ext", RDIR=X2, PAR="1"), [f"ext_{t}"])
# ---------- 灵敏度（S5，单层 MILP + 邻域搜索） ----------
sens = [("base", None, None)] + [("alpha", "ALPHA1", v) for v in (0.2, 0.4, 0.7, 0.9, 0.99)] + \
       [("heat", "C_HEAT", v) for v in (0.1, 0.15, 0.25, 0.3)] + [("chip", "T_CHIP_MAX", v) for v in (80.0, 83.0, 87.0, 90.0)] + \
       [("M", "M", v) for v in (90, 180, 730)] + [("RN", "R_N", v) for v in (0.0, 0.1, 0.3)]
for g, k, v in sens:
    nm = "base" if g == "base" else f"{g}_{v}"; out = f"{RES}/{S}_sens_{nm}.pkl"
    solve(S, "S5", "DRO_sens", "mono", out=out, env=dict(TL2="1500", PARAM_OVER=json.dumps({} if k is None else {k: v})), name=f"sens_{nm}")
# ---------- 气候区 ----------
for site in ("guian", "shaoguan"):
    q = lambda t: f"{RES}/{site}_{t}.pkl"
    solve(site, "S1", "DRO_S1"); solve(site, "S3", "DRO_S3")
    solve(site, "S4", "DRO_S4", prev=q("DRO_S3"), deps=[f"{site}_DRO_S3_ccg"]); solve(site, "S5", "DRO_S5", prev=q("DRO_S4"), deps=[f"{site}_DRO_S4_ccg"])


def log(msg):
    line = time.strftime("%m-%d %H:%M:%S ") + msg; print(line, flush=True); open(f"{RES}/log_v4.txt", "a").write(line + "\n")


def alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False


class Ext:
    def __init__(s, pid): s.pid = pid; s.returncode = None
    def poll(s): return None if alive(s.pid) else 0


if __name__ == "__main__":
    PAR = int(os.environ.get("NPAR", "4"))
    done = {t["name"] for t in T if os.path.exists(t["out"])}; fail = {}; run = {}
    for t in T:
        pf = f"{LOGS}/v4_{t['name']}.pid"
        if t["name"] not in done and os.path.exists(pf):
            pid = int(open(pf).read().strip() or 0)
            if pid and alive(pid): run[t["name"]] = Ext(pid); log(f"接管运行中的任务 {t['name']}（pid {pid}）")
    log(f"启动：共 {len(T)} 个任务，已完成 {len(done)} 个")
    while True:
        for nm, p in list(run.items()):
            if p.poll() is not None:
                t = next(x for x in T if x["name"] == nm); run.pop(nm)
                if os.path.exists(t["out"]): done.add(nm); log(f"完成 {nm}")
                else: fail[nm] = fail.get(nm, 0) + 1; log(f"失败 {nm}（第 {fail[nm]} 次）")
        todo = [t for t in T if t["name"] not in done and t["name"] not in run and fail.get(t["name"], 0) < 2 and all(d in done for d in t["deps"])]
        if not todo and not run: break
        while todo and len(run) < PAR:
            t = todo.pop(0); lf = open(f"{LOGS}/v4_{t['name']}.txt", "a")
            run[t["name"]] = subprocess.Popen(t["cmd"], stdout=lf, stderr=lf, env={**os.environ, **t["env"]}, cwd=CODE)
            open(f"{LOGS}/v4_{t['name']}.pid", "w").write(str(run[t["name"]].pid)); log(f"开始 {t['name']}（并行 {len(run)}/{PAR}）")
        time.sleep(20)
    log(f"全部结束：完成 {len(done)}，失败 {[k for k, v in fail.items() if v >= 2]}")
