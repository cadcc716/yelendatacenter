"""组装最终报告: docs/_report_head.md + _report_map.md + _report_body.md + _tables.md -> docs/论文实验闭环.md"""
import re
import config as C

D = C.ROOT / "docs"
tab = (D / "_tables.md").read_text(encoding="utf-8")
sec = {m.group(1): m.group(0) for m in re.finditer(r"## (T\d+)[^\n]*\n.*?(?=\n## T\d+|\Z)", tab, re.S)}
head = (D / "_report_head.md").read_text(encoding="utf-8")
body = (D / "_report_body.md").read_text(encoding="utf-8")
summary = (D / "_summary.md").read_text(encoding="utf-8") if (D / "_summary.md").exists() else ""
sens = (D / "_sens.md").read_text(encoding="utf-8") if (D / "_sens.md").exists() else ""
p = sec["T1"].split("\n", 2)[2]
head = head.replace("{SUMMARY}", summary).replace("{PARAM_TABLE}", p)
tables = "\n\n".join(sec[k].replace("## ", "### 表") for k in ["T2", "T3", "T4", "T5", "T6", "T7", "T8", "T10", "T11", "T12"] if k in sec)
body = body.replace("{MAP}", (D / "_report_map.md").read_text(encoding="utf-8")).replace("{TABLES}", tables)
body = body.replace("{SENS}", (sec.get("T9", "").replace("## ", "### 表") + "\n\n" + sens))
(D / "论文实验闭环.md").write_text(head + "\n\n" + body, encoding="utf-8")
print("written", D / "论文实验闭环.md")
