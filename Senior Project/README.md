# 基於多模態深層卷積網路之微表情辨識研究

以 PyTorch 實作的微表情辨識模型，改良自 FeatRef（Feature Refinement）論文，結合光流動態特徵、CNN 靜態特徵與 Self-Attention，將微表情分為正向、負向、驚訝、其他四類，並提供即時辨識 GUI。

- 專題報告：[基於多模態深層卷積網路之微表情辨識研究畢業專題報告.pdf](docs/基於多模態深層卷積網路之微表情辨識研究畢業專題報告.pdf)
- Demo 影片：[docs/Demo.mp4](docs/Demo.mp4)
- 獲獎：國立臺南大學資訊工程學系 112 級畢業專題競賽「多媒體與雲端網路應用」組第三名（[獎狀](docs/Award.jpg)）

<br>

## 目錄結構

```text
Senior Project/
├── README.md
├── Fig/                          # README 使用的架構圖與實驗結果圖
├── docs/                         # 專題報告 PDF、Demo 影片、獎狀
└── FeatRef/                      # 程式碼（請在此目錄下執行）
    ├── run.py                    # 主要訓練與評估程式
    ├── run_save_models.py        # 同 run.py，另於訓練結束時儲存權重（供 GUI 使用）
    ├── run_early_stop.py         # 儲存最佳 epoch 權重的實驗版本
    ├── run_warm_start.py         # 各 fold 接續前一 fold 最佳模型的實驗版本
    ├── run_debug.py              # 3 epoch、輸出逐筆預測的除錯版本
    ├── requirements.txt          # 訓練所需套件
    ├── data/
    │   ├── mydata.py             # Dataset 與資料切分
    │   ├── database/             # 光流圖：<類別>/flow_x_*.jpg、flow_y_*.jpg
    │   └── preprocessing_o2a/    # 臉部裁切影格：cropped/<類別>/<樣本>/*.jpg
    ├── model/
    │   ├── mymodel.py            # 完整模型（Inception + CNN + Self-Attention）
    │   └── ablation/             # 消融實驗模型
    ├── gui/                      # 即時微表情辨識 GUI（PyQt5 + 網路攝影機）
    │   ├── cv2_cam_pyqt.py       # 主程式
    │   ├── run_gui.bat           # Windows 啟動檔
    │   ├── requirements.txt      # GUI 所需套件
    │   └── sample_results/       # GUI 輸出的逐秒辨識結果範例（Excel）
    ├── pytorchtools.py           # Early Stopping 工具（主程式目前未使用）
    └── torchsummary.py           # 模型摘要工具（主程式目前未使用）
```

<br>

## Model Architecture
本組專題架構圖如下, 從Feature refinement: An expression-specific feature learning and fusion method for micro-expression recognition論文改良及創新
- 使用pytorch作為深度學習框架，並使用google所開發的Mediapipe套件對資料集進行預處理(pre-processing) 

- 使用光流法(Optical Flow)捕捉臉部表情肌肉運動並使用Inception V1網路作為動態特徵(dynamic feature)捕捉，額外使用CNN去做靜態特徵提取，靜態特徵在此指的是原先臉部影像上就存在的特徵(例如: 皺紋、膚色…等)，原始論文只考慮動態特徵但並未考慮靜態特徵的情況下，可能導致辨識的準確率受到影響

- 使用Attention機制將表情特徵送入多個Attention子網路去學習微表情類別，其中透過self-attention去計算相關性，從而判斷表情特徵是否為特定表情類別作為最終輸出，最後使用Softmax進行分類(正向、負向、驚訝、其他)


![ProjectArchitecture](Fig/ProjectArchitecture.png)

<br>

## Database
資料集的取得均有簽署相關合約取得授權! 資料集相關說明如下圖:

![Database](Fig/Database.png)

<br>

## Performance
在驗證階段採用Leave-One-Subject-Out, 並且我們針對4個獨立資料集(CASME、CASME2、SAMM、SMIC-HS)及綜合資料集(前述四個資料集混合)進行訓練，可得到它們相對應的準確率、未加權F1分數、未加權平均召回率及加權的F1分數，透過這些常見的評估效能指標，我們將其用於評估我們改良的模型，結果如下:

![Performance](Fig/Performance.png)

<br>

## Compare
另外我們針對我們提出的方法和原始FR論文進行實驗對照組,我們改良的模型相對於FeatRef的結果在三個常見的資料集上 (CASME2、SAMM、SMIC-HS)準確率約提升2至5%左右，同時本組模型分別在CASME資料集和綜合資料集上分別取得80%及72%的準確率，訓練的超參數皆為50epoch及3round time的設定下進行訓練所得到的實驗結果!

![ExperimentCompare](Fig/ExperimentCompare.png)

<br>

## Ablation Study
我們透過CASME2微表情資料集進行消融研究來分析評估哪一個模組對於整體網路架構的影響力最大,其中最具影響力的是以動態訓練特徵為主Inception Block模組，其次是自我注意力機制 (Self-Attention mechanism)模組，最後是以靜態特徵訓練為主的CNN Block模組，將上述三個網路模組結合並訓練，可得到約72%左右的準確率!

![AblationStudy](Fig/AblationStudy.png)

<br>

## Conclusion
我們的模型架構是基於Feature Refinement原始論文，結合多種神經網路、光流法、attention機制，彌補FeatRef僅使用光流捕獲到微表情的動態資訊，卻忽略靜態資訊的問題，經本組改良後的model 在3個常見資料集(CASME2、SAMM、SMIC-HS) 上提升約2~5% 準確率、在綜合資料集取得72%準確率，證明了我們的模型有良好的泛化能力。

<br>

## 程式使用說明

### 1. 環境安裝

```bash
cd "Senior Project/FeatRef"
python -m venv .venv
source .venv/bin/activate            # Windows：.venv\Scripts\activate
# 有 GPU 時，請先依 CUDA 版本安裝 PyTorch：https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

`run*.py` 會 import `torchviz`，執行前需先安裝。原開發環境的套件版本未留下紀錄，`requirements.txt` 未鎖定版本。

### 2. 資料準備

資料集（CASME、CASME II、SAMM、SMIC-HS）需向各資料集作者簽署同意書後取得，本 repo 只附少量官方允許的範例檔，用來說明格式。完整資料請依下列結構放入 `FeatRef/data/`：

```text
data/
├── database/                         # 光流圖
│   └── <類別>/                       # 0 正向、1 負向、2 驚訝、3 其他
│       ├── flow_x_<樣本代碼>.jpg     # 例如 flow_x_3001_46_4659.jpg
│       └── flow_y_<樣本代碼>.jpg
└── preprocessing_o2a/cropped/        # 以 Mediapipe 裁切的臉部影格（onset 到 apex）
    └── <類別>/<樣本代碼>/*.jpg
```

- 樣本代碼的第一段為「資料集代碼＋受試者編號」，例如 `3001` 代表 CASME II（30）的第 01 位受試者；資料集代碼為 CASME II `30`、CASME `40`、SAMM `10`、SMIC-HS `200`
- 每個光流樣本都必須有對應的裁切影格資料夾，否則讀取時會出錯。repo 內的範例只有 3 個樣本附裁切影格，無法直接用來訓練
- 從原始影片產生裁切影格與 TVL1 光流圖的前處理程式未收錄在本 repo；`gui/cv2_cam_pyqt.py` 中的即時處理流程（Mediapipe 臉部裁切、apex 偵測、TVL1 光流）可作為參考

### 3. 訓練與評估

請在 `FeatRef/` 目錄下執行（程式以相對路徑讀取 `data/`）：

```bash
python run.py --epochs 50 --train_round 3        # README 實驗結果使用的設定
```

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--path` | `data/database/` | 光流資料路徑，請維持此相對路徑 |
| `--epochs` | 50 | 每個 fold 的訓練 epoch 數 |
| `--train_round` | 1 | 重複訓練回合數 |
| `--batch_size` | 4 | |
| `--lr` | 0.001 | Adam 學習率 |
| `--result_path` | `result/` | 輸出資料夾 |

輸出在 `result/`：各回合的準確率、預測與真實標籤（`round_<n>/`）、整體指標（Accuracy、UF1、UAR、WF1）與 loss／accuracy 曲線。

**評估切分方式**：`data/mydata.py` 以「樣本」（一段微表情影片）為單位做 leave-one-out：每個 fold 取 1 個樣本為測試集，從同一受試者的其他樣本中隨機取最多 5 個作為驗證集（用於挑選驗證 loss 最低的 epoch），其餘為訓練集。因此 fold 數等於樣本數（綜合資料集為 759），同一受試者的其他樣本會出現在訓練集中；這與以「受試者」為單位的 Leave-One-Subject-Out 不同，比較其他論文的 LOSO 結果時請留意。

### 4. 訓練腳本版本

開發過程留下的實驗版本，皆以 `run.py` 為基礎：

| 檔案 | 原檔名 | 與 `run.py` 的差異 |
|---|---|---|
| `run.py` | `run.py` | 主要版本 |
| `run_save_models.py` | `run_自動保存模型.py` | 最後一個 fold 完成時儲存權重，檔名依訓練的資料而定：綜合資料集為 `model_FULL.pth`，單一資料集為 `model_CASME2.pth`、`model_CASME.pth`、`model_SAMM.pth` 或 `model_SMIC-HS.pth`；預設 5 epoch、10 回合、batch size 1 |
| `run_early_stop.py` | `run有EarlyStop.py` | 儲存驗證 loss 最佳的權重，並在下一個 fold 開始時載入前一個 fold 的最佳權重；提前停止的條件已被註解，實際不會提前停止；預設 1 epoch。接續前一 fold 權重的問題同 `run_warm_start.py` |
| `run_warm_start.py` | `run_最佳驗證損失儲存模型.py` | 每個 fold 從前一個 fold 的最佳模型接續訓練；前一個 fold 的訓練資料包含本 fold 的測試樣本，評估結果會偏高，不適合用來報告準確率 |
| `run_debug.py` | `run__.py` | 3 epoch，逐 fold 印出真實與預測標籤 |

### 5. 消融實驗模型

`model/ablation/` 內各檔案分別移除一個模組，對應上方 Ablation Study：

| 檔案 | 保留的模組 | 移除的模組 |
|---|---|---|
| `mymodel_no_inception.py`（原 `mymodel_cnn_att.py`） | CNN + Self-Attention | Inception（光流動態特徵） |
| `mymodel_no_attention.py`（原 `mymodel_cnn+opu.py`） | Inception + CNN | Self-Attention |
| `mymodel_no_cnn.py`（原 `mymodel_opi+att.py`） | Inception + Self-Attention | CNN（靜態特徵） |

執行時將要測試的檔案複製為 `model/mymodel.py`（或修改 `run.py` 的 `from model.mymodel import FeatRef`）後再訓練。

### 6. 即時辨識 GUI

`gui/` 為 Windows 上的 PyQt5 程式：以網路攝影機擷取臉部，用 Mediapipe 裁切、偵測 apex 影格並計算 TVL1 光流，每秒輸出四類表情機率；同時可開啟影片作為情緒刺激並同步播放聲音。

```bat
cd "Senior Project\FeatRef\gui"
python -m pip install -r requirements.txt
copy ..\model_FULL.pth .
run_gui.bat
```

- 需要以 `run_save_models.py` 訓練綜合資料集產生的 `model_FULL.pth`（權重未收錄在 repo），放在 `gui/` 目錄
- 光流使用 `cv2.optflow`，必須安裝 `opencv-contrib-python`，不能與 `opencv-python` 同時安裝
- `moviepy` 需小於 2.0（程式使用 `moviepy.editor`）；Mediapipe 使用舊版 `mp.solutions.face_mesh` API
- 執行時會在 `gui/sub01/` 寫入擷取的影格、apex 影格、光流圖與 `FinalResult.xlsx`；`sample_results/` 為先前執行的結果範例
- `gui/model/mymodel.py` 與 `model/mymodel.py` 內容相同，供 GUI 獨立執行

### 7. 第三方程式

- `torchsummary.py` 修改自 [pytorch-summary](https://github.com/sksq96/pytorch-summary)（MIT）
- `pytorchtools.py` 修改自 [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch)（MIT）
