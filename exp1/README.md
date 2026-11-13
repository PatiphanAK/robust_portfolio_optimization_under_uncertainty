# Experiment 1 — Nominal vs Wasserstein vs Possibility (A, B) บน AR–GARCH-t เดียวกัน

## ติดตั้ง
```bash
pip install cvxpy arch yfinance pandas numpy scipy matplotlib
```
(Python 3.10 ขึ้นไป ใช้ solver Clarabel ที่มากับ cvxpy)

## ลำดับการรัน (ทำตามนี้เพื่อให้การทดลองไม่ bias)
```bash
cd exp1
git init && git add -A && git commit -m "Exp1 config frozen before OOS run"   # 1) ล็อก config ก่อนเห็นผล
cd src
python tests.py                   # 2) ต้องขึ้น ALL PASS
python verification/run_all.py    #    ตรวจอัลกอริทึมเพิ่ม (~10 วินาที) ต้องขึ้น ALL PASS
python run_exp1.py --download     # 3) ดึงข้อมูล Yahoo + รันทั้งหมด (~20 นาที)
python exp1_mini.py               # 4) รูป demo (ใช้ cache, ~30 วินาที)
```
รันซ้ำครั้งต่อไปไม่ต้องใส่ `--download` (ใช้ `data/prices_daily.csv` เดิม และ cache เดิม)
ถ้าแก้ `config.py` หลังเห็นผลแล้ว ต้องบอกในรายงานว่าแก้อะไรและเพราะอะไร

`--synthetic` ใช้ทดสอบโค้ดเท่านั้น **ห้ามนำตัวเลขไปรายงาน**

## ผลลัพธ์ (`results/real/`)
| ไฟล์ | ใช้ที่ไหน |
|---|---|
| `table_perf_main.tex`, `table_front_main.tex`, `table_regimes_cvar.tex` | ตารางในหัวข้อการทดลองที่ 1 ของรายงาน |
| `table_summary_all.csv`, `table_perf_all.tex` | ภาคผนวก (รวม Wass ×0.5, ×2) |
| `fig_oos_front.png` | สไลด์หน้า Exp 1 + รายงาน |
| `fig_wealth.png` | รายงาน (และสไลด์ backup) |
| `fig_demo_weights.png` | สไลด์ Demo |
| `front_spread.csv` | หลักฐานว่า Poss-A front หด / Poss-B ไม่หด |
| `run_meta.json` | commit hash, config fingerprint, วันที่ดึงข้อมูล → ใส่ footnote สไลด์ |

## ใส่ในรายงาน
1. คัดลอกโฟลเดอร์ `results/real/` ไปไว้ข้างไฟล์ .tex หลักของรายงาน
2. `\input{exp1_section.tex}` ในบทที่ 4 (preamble ต้องมี `booktabs`, `graphicx`, `amsmath`
   และ environment `proposition` แบบเดียวกับ "ข้อเสนอ 2.1")
3. เขียนย่อหน้าใน [วงเล็บเหลี่ยม] หลังเห็นผลจริง

## โครงสร้างโค้ด
- `config.py` การตั้งค่าทั้งหมด (pre-registration)
- `data.py` ดึง/โหลดข้อมูล, ผลตอบแทนรายสัปดาห์ (forward-fill เท่านั้น)
- `forecast.py` AR(1)–GARCH(1,1)-t รายสินทรัพย์, refit ทุก 4 สัปดาห์, residual ร่วมสำหรับ FHS
- `ambiguity.py` รัศมี Wasserstein (stationary bootstrap + W1 แม่นตรง), κ ของ Poss-A, α-cut ของ Poss-B
- `optimize.py` ปัญหาทั้ง 4 แบบใน CVXPY (DPP, compile ครั้งเดียว)
- `backtest.py` rolling-origin, ผลตอบแทนจริง, turnover, ต้นทุน
- `metrics.py` Sharpe/MDD/CVaR, Hypervolume 2 มิติ, bootstrap CI, Ledoit–Wolf HAC, Holm
- `tests.py` ตรวจความถูกต้อง 9 กลุ่ม (รวม convexity ผ่าน Hessian)
- `verification/` ชุดตรวจอัลกอริทึม (`run_all.py` รันทั้งหมด)
  - `check_optimizer.py` optimum ระดับ global, Pareto front monotone ใน λ, Poss-A ทุก λ, ผลของ ridge
  - `check_robust.py` สูตร Wasserstein tight (มีการแจกแจงที่ถึงค่า) และเป็น upper bound, α-cut ซ้อนกัน, κ
  - `check_backtest.py` จังหวะผลตอบแทน (i→i+1), turnover/ต้นทุนเทียบมือ, backtest ย่อ + look-ahead
