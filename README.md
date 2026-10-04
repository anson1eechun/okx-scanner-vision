# okx-scanner-vision

把一套 OKX 即時掃描器的因子，拿到 6 年、74 個幣的 Binance 歷史資料上重新檢驗。結論：**沒有一個因子扣掉成本後站得住**。這個 repo 保留檢驗程式、預註冊文件和原始結果，當作負面結果的完整紀錄。

原本的掃描器（`scan.py`）與研究在 MacBook 上進行，只有 15–60 天的 OKX 資料。這裡改用 [crypto-datalake](https://github.com/anson1eechun/crypto-datalake) 收的 Binance Vision dump：74 個 USDT 永續，2020-09 至 2026-08，約 3,300 萬根 5 分鐘棒。

## 結果

| 步驟 | 問題 | 結果 | 檔案 |
|---|---|---|---|
| 方向檢定 | 掃描器的 7 個因子在 15m / 1h / 4h 有沒有方向資訊？ | 最好約 +8 bp，低於 12 bp 來回成本；OI 型態與量能兩個因子方向和設計相反，74 幣中 92–99% 同號 | `analyze.py` → `results/report.txt` |
| OI 確認突破 | 小樣本裡 PF 1.19 的突破策略，放大到 6 年 74 幣還成立嗎？ | 1h 完整版每筆 +4.8 bp，t 0.8；純突破每筆 −12.1 bp | `strat2.py` → `results/strat2.txt` |
| Step 2 | 修正方向後的合成分數，持有 12–24 小時能不能贏過成本？ | **FAIL**：2024 後 edge +8.8 bp < 12 bp，剔除最好 5% 的日子後轉負 | `step2.py` → `results/step2.txt` |
| Step 3 | 深簿失衡（DBI）在 Binance bookDepth 上有沒有方向、能不能交易？ | **FAIL**：Q1 方向檢定不過；Q2 凍結策略在 BTC 進 maker 出 taker 每筆 −0.074R（t −6.8），74 幣中 0 幣為正 | `step3.py` → `results/step3.txt` |

反覆出現的事實：BTC 永續在 5 分鐘到 1 小時尺度是均值回歸的，押延續會輸；押反轉在統計上是對的，但效應只有 0.5–2 bp，成本是 8–12 bp。整條研究線的總結在 [`docs/negative-results-2026-09-24.md`](docs/negative-results-2026-09-24.md)。

## 方法

- **時間對齊**：所有欄位以「資訊完成時間」t 標記，一列只含 t 之前可知的資訊。Binance metrics 在 2024-03-04 改了時間戳慣例，`build.py` 逐日判斷對齊，`check_align.py` 用 taker 比例對 K 線驗證。
- **bookDepth 對齊**：`bd_check.py` 從 ±1% 深度反推中價，和 1 分鐘 K 線報酬做 −2…+2 期的相關，確認快照沒有看到未來。
- **預註冊**：Step 2、Step 3 的閘門（G1–G6、Q1/Q2）在算任何前瞻報酬前寫進 `PREREG_*.md`，寫完不改，偏離記在文末。程式檔以 sha256 鎖定。
- **穩健性**：日層級 Newey-West t、廣度（多少幣同號）、剔除最好 5% 的日子、延後進場、前後兩半期分開看。
- **DBI 策略**（`dbi_strategy.py`）：三個深度失衡量的等權 z 分數，沒有擬合參數；60 bps 停損、120 bps 停利、4 小時時間停損，maker 進場。

## 檔案

| 檔案 | 用途 |
|---|---|
| `build.py` | Binance Vision dump → 每幣 5 分鐘 panel（K 線、OI、多空比、taker 比、資金費） |
| `analyze.py` | 重建掃描器因子並產出方向檢定報告 |
| `bt.py` | 原本的突破 / 擁擠交易回測（讀 MacBook 上 `fetch_history.py` 抓的 OKX 資料，此 repo 不含） |
| `strat2.py` | 用 `bt.py` 的模擬邏輯，換成 6 年 Binance 資料重跑 |
| `step2.py` | Step 2 預註冊檢定 |
| `bd_build.py` / `bd_check.py` | bookDepth → 每分鐘失衡序列，以及時間對齊檢查 |
| `dbi_strategy.py` / `step3.py` | DBI 訊號與凍結策略，Step 3 預註冊檢定 |
| `results/` | 各步驟的完整文字輸出 |

## 執行

需要 crypto-datalake 的 Binance Vision 原始檔（路徑寫死 `/srv/nas/marketdata/vision/...`）和 Python 套件 `polars`、`numpy`、`pyarrow`。

```bash
python build.py --workers 12        # 產生 out/panel/
python analyze.py all --workers 14  # 方向檢定
python step2.py --workers 14
python bd_build.py --workers 14 && python step3.py --workers 14
```

`out/`（panel、特徵、中間 parquet，約 6.8G）不收錄。
