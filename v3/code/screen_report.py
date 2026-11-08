import sys, glob, os, pickle, json, numpy as np, pandas as pd
import params as PR, analyze as AN
R = os.path.abspath("../results/screen"); rows = []
for f in sorted(glob.glob(f"{R}/*_S5.pkl")):
    name = os.path.basename(f)[:-7]; r = pickle.load(open(f, "rb")); P = PR.params(**r["params_over"])
    e = AN.table_e("zhongwei", tags={"S5": f}).iloc[0]
    row = dict(名称=name, N=r["x"]["N"], 参数=json.dumps(r["params_over"], ensure_ascii=False), 能源侧=round(r["obj"] / 1e4, 1), gap=round(r["gap_final"], 4),
               MC=round(r["x"]["MC"]), FC=round(r["x"]["FC"]), TES=round(r["x"]["TES"]), HP=round(r["x"]["HP"]), PUE=round(e["名义PUE"], 3),
               碳=round(e["碳排放/t"]), 余热GWh=round(e["余热回收/MWh"] / 1e3, 2), 满足率=round(e["实时满足率/%"], 3))
    for date in ("2023-06-15", "2023-11-22"):
        fp = f"{R}/{name}_days/zhongwei_{date}_逐时.csv"
        if not os.path.exists(fp): continue
        h = pd.read_csv(fp); g = pd.read_csv(fp.replace("逐时", "档位")); act = g[[f"档{k}" for k in range(1, 6)]].sum(axis=1)
        k1 = g["档1"] / act.where(act > 1); val = h["时"].isin(P["VALLEY_H"]); d = date[5:]
        row.update({f"{d}午k1": round(k1[12:16].mean(), 2), f"{d}夜k1": round(k1[0:6].mean(), 2),
                    f"{d}谷占比": round(h.loc[val, "可延期处理"].sum() / h["可延期处理"].sum(), 2),
                    f"{d}≥45℃": round((h["液冷温度"] >= 45 - 1e-6).mean(), 2), f"{d}Tw": f"{h['液冷温度'].min():.0f}–{h['液冷温度'].max():.0f}"})
    rows.append(row)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(pd.DataFrame(rows).to_string(index=False))
