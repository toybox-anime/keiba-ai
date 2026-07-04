# 楽天競馬 予想AIエージェント

楽天競馬の出馬表・オッズを自動取得し、**予想と買い目（軍資金からの配分つき）**を出すツールです。
期待値(EV)ベースの買い目生成、勝率モデル（機械学習）、毎晩の自動データ収集まで対応します。

---

## 🏇 今日すぐ使う（3ステップ）

```powershell
# 1. フォルダへ移動
cd C:\Users\kynmt\keiba-ai

# 2. 発走の約10分前に実行（競馬場名・レース番号・軍資金を指定）
python keiba.py predict --track 大井 --race 11 --budget 5000 --ev --fresh

# 3. 予想と買い目が画面に出る（reports/ にも保存）
```

これだけでOKです。`--track` は本日開催している競馬場名を入れてください。

> **なぜ発走10分前？** オッズは発走直前まで変動します。`--ev --fresh` は最新オッズを取り直しますが、
> 取得に約4分かかるため、直前に実行するほど判断が正確になります。

---

## コマンド早見表

| やりたいこと | コマンド |
|---|---|
| 🤖 本日の狙い目を自動抽出 | `scan --track 高知 --budget 5000` |
| 予想だけ見る | `predict --track 大井 --race 11` |
| 軍資金から買い目も出す | `predict --track 大井 --race 11 --budget 5000` |
| 期待値で買い目を最適化（推奨）| `predict --track 大井 --race 11 --budget 5000 --ev --fresh` |
| Geminiに予想させる依頼文を出す | `predict ... --ev --fresh --gemini` |
| 学習データを集める（1開催）| `collect-day --track 大井` |
| 全部おまかせ（収集）| `auto` |

競馬場・レース番号の代わりに `--race-id <18桁>` でも指定できます。

---

## 🤖 本日の狙い目を自動抽出（エージェント）

「どのレースを買えばいい？」をエージェントが判断します。本日の全レースを分析し、
**妙味（EVプラス）のあるレースだけをランキング**で提示します。

```powershell
python keiba.py scan --track 高知 --budget 5000          # 高知の全レースを分析
python keiba.py scan --budget 5000                       # 本日の全開催を分析（時間長め）
```

出力イメージ：
```
# 本日の狙い目
## 🎯 狙い目ランキング（EVプラス＝妙味のあるレース）
| 順位 | レース | 一番のおすすめ | オッズ | 的中率 | EV |
| 1 | 高知8R | 三連複 4-7-8 | 11.4 | 13.6% | 1.55 |
| 2 | 高知11R| ワイド 4-8   | 3.8  | 31.1% | 1.18 |
## 見送り（EVプラスなし）
高知1R・2R・5R …
```
妙味のあるレースが無ければ「本日は無理に買わない」が正解と表示します。
上位レースは案内されたコマンド（`predict ... --gemini`）で深掘りできます。

> 各レースで連系オッズを取得するため、1開催12Rで数分かかります（取得間隔は短縮済み）。

---

## 予想と買い目の出し方

### 1. 予想だけ
```powershell
python keiba.py predict --track 大井 --race 11
```
各馬の指標（単勝・人気・市場勝率・近走）を一覧で出します。

### 2. 軍資金から買い目を作る
`--budget` に軍資金（円）、`--style` でリスク許容度を指定。
```powershell
python keiba.py predict --track 大井 --race 11 --budget 5000 --style balanced
```
| style | 傾向 |
|---|---|
| `conservative` | 単勝・複勝中心で堅実 |
| `balanced`（既定） | 単複＋馬連/ワイド |
| `aggressive` | 三連複/三連単＋穴 |

金額は100円単位で、合計が軍資金を超えないように自動配分されます。

### 3. 期待値(EV)モード ★おすすめ
`--ev` を付けると**全券種（単勝・複勝・馬連・馬単・ワイド・三連複・三連単）の実オッズ**を
取得し、レポートに次を出します。
```powershell
python keiba.py predict --track 大井 --race 11 --budget 5000 --ev --fresh
```
- **🎯 一番のおすすめ買い方**：全券種を比較し、的中率×期待値のバランスが最良の1点を推奨。
- **券種ごとの比較表**：各券種の代表買い目・オッズ・的中率・EVを一覧。
- **ワイドの軸流し／ボックス**：当てやすい買い方を具体的に提示。
- **買い目プラン**：軍資金をEV・ケリー基準で配分。
- 妙味のある買い目が無ければ**見送り**（軍資金を使い切らないのが正解）。
- `--fresh` は最新オッズを取り直す（発走直前の判断用）。
- 取得は全6ページで**約6分**かかります（楽天の60秒間隔ルールのため）。

### 4. AIに予想させる

**A. Geminiチャットに貼る（無料・推奨）** — `--gemini`
```powershell
python keiba.py predict --track 高知 --race 11 --budget 5000 --ev --fresh --gemini
```
出走馬データ＋EV分析を詰めた「予想依頼文」を出力するので、それを丸ごと
**Geminiチャットにコピペ**すれば、Geminiが本気の予想＋買い目を返します
（依頼文は `reports/<RACEID>_gemini.txt` にも保存）。Gemini Pro契約をそのまま活用、追加費用ゼロ。

**A-2. Gem（カスタムGemini）を使う** — `--gem`（毎回ラク）
1. 初回だけ：`--gem` を一度実行すると `reports/gem_instructions.txt` が出来る。
   その中身を Gemini の「Gem を作成」の指示欄に貼って、予想用Gemを1つ作る。
2. 以降：`--gem` で出力されるデータを、作ったGemに貼るだけ（毎回の指示文が不要に）。
```powershell
python keiba.py predict --track 高知 --race 11 --budget 5000 --ev --fresh --gem
```

**B. Claudeに書かせる（API課金）** — `ANTHROPIC_API_KEY` を設定
```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."
```
設定すると `predict`（`--gemini`なし）でClaudeが予想文を生成。1予想あたり数円〜数十円。

どちらも未使用なら、指標と買い目の表だけ（無料・AI予想なし）。

---

## もっと賢くする（勝率モデル）

過去レースを集めて学習させると、`predict` が自動でモデルを使い精度が上がります。

```powershell
# 1. データ収集（終わった開催を集める。1開催 約20分・無人）
python keiba.py collect-day --track 大井

# 2. 学習（30レース以上集まったら実行）
python keiba.py train

# 3. 以降の predict は自動でモデルを使用
python keiba.py predict --track 大井 --race 11 --budget 5000 --ev --fresh
```

> 収集は60秒間隔のため時間がかかります。少しずつ貯めるか、下の自動化を使ってください。

---

## 自動化（毎晩おまかせ）

`auto` が「本日の全開催を収集」します（毎晩のデータ収集担当）。
**学習は `predict` 実行時に自動で行われます**（タスク環境ではML依存が読めないことがあるため、
確実に動くインタラクティブ環境＝予想時に再学習する設計）。データが前回学習＋10レース以上
増えていれば、予想の前に自動でモデルを更新します。

```powershell
python keiba.py auto            # 手動で1回（収集）
```

**毎晩自動で回す**には、`run-auto.bat` をタスクスケジューラに登録します（PowerShellで1回だけ）：
```powershell
schtasks /Create /SC DAILY /TN "KeibaAI Auto" /TR "C:\Users\kynmt\keiba-ai\run-auto.bat" /ST 23:30
```
- 解除: `schtasks /Delete /TN "KeibaAI Auto" /F`
- 今すぐ実行: `schtasks /Run /TN "KeibaAI Auto"`
- ログ: `data/auto.log`

これで毎晩データが貯まり、十分集まればモデルが勝手に更新されます。

---

## 初回セットアップ

```powershell
cd C:\Users\kynmt\keiba-ai
pip install -r requirements.txt   # 依存ライブラリを入れる
```
実行は `python keiba.py <コマンド>`（インストール不要のランチャー経由）。

---

## 注意・免責

- **馬券は自己責任・20歳以上・余裕資金の範囲で。** 本ツールは的中を保証しません。
- 取得は楽天競馬のrobots.txtに従い**60秒間隔**を厳守。利用規約も各自で確認してください。
- オッズは発走前の暫定値です（確定するのは発走時）。

---

<details>
<summary>開発者向け：構成とロードマップ</summary>

### データの流れ
```
collect → dataset → train → model ┐
  ↓                                ↓
scraper → parser → features → betting(EV/ケリー) → report
```

### モジュール
| ファイル | 役割 |
|---|---|
| `scraper.py` | 60秒間隔を守る取得＋キャッシュ（オッズは10分の短命キャッシュ）|
| `raceid.py` / `schedule.py` | RACEID解決（競馬場名→開催ID）|
| `parser.py` / `result.py` | 出馬表・結果ページの解析 |
| `features.py` / `ml.py` | 指標・ML特徴量 |
| `odds.py` / `ev.py` | オッズ解析・期待値/ハーヴィル/ケリー計算 |
| `betting.py` | 軍資金からの買い目配分 |
| `model.py` / `train.py` / `dataset.py` | 勝率モデルの学習・推論・蓄積 |
| `report.py` / `cli.py` | レポート生成・コマンド |

### ロードマップ
- [x] RACEID解決 / 出馬表・結果・オッズ解析（実HTML検証済み）
- [x] 軍資金からの買い目配分（券種・100円単位）
- [x] 期待値(EV)・ケリー基準の買い目（`--ev`）
- [x] LightGBM勝率モデル（`collect`→`train`→自動使用）
- [x] 一括収集・自動化（`collect-day` / `auto` / タスクスケジューラ）
- [x] オッズの短命キャッシュ＋`--fresh`（鮮度対応）
- [ ] 回収率バックテスト（EV戦略が実際に勝てるか検証）

</details>
