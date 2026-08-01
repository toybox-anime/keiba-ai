# 楽天競馬 予想支援ツール

楽天競馬の出馬表・オッズを自動取得し、**期待値(EV)ベースの買い目（軍資金からの配分つき）**を出す
研究・下調べ用ツールです。予想文はGemini/Claudeに書かせることもできます。

> ⚠️ **正直な前提**：本ツールは「必ず勝てる」ものではありません。地方競馬は控除率が約20〜25%あり、
> 人気馬（本命）を機械的に買うだけでは構造的に勝てません（詳細は「注意・免責」）。
> 本ツールの役割は**予想の下調べ・妙味(EV)のある買い目の発見**であり、当たりを保証しません。

---

## 🏇 今すぐ使う

発走の**約10分前**に、買いたいレースを指定して実行します。

**予想文もGeminiに書かせたい場合**（無料・Gemini Pro契約を活用）:
```powershell
cd C:\Users\kynmt\keiba-ai
git pull
python keiba.py predict --track 大井 --race 11 --budget 5000 --ev --fresh --gem | clip
```
→ 依頼文がクリップボードに入るので、[Gemini](https://gemini.google.com)（作成したGem）に**貼るだけ**。

**ツールのEV分析だけ見たい場合**（AI不要）:
```powershell
python keiba.py predict --track 大井 --race 11 --budget 5000 --ev --fresh
```
→ 一番のおすすめ・券種比較・買い目プラン（何円ずつ）が画面と `reports/` に出ます。

> `--track` は本日開催の競馬場名。`--fresh` は最新オッズを取り直します（全券種取得で数分）。

---

## コマンド早見表

| やりたいこと | コマンド |
|---|---|
| 予想＋買い目（EV最適化）| `predict --track 大井 --race 11 --budget 5000 --ev --fresh` |
| Gemに貼る依頼文を出す（推奨）| `predict ... --ev --fresh --gem \| clip` |
| 本日の狙い目を自動抽出 | `scan --track 高知 --budget 5000` |
| 学習データを集める（1開催）| `collect-day --track 大井` |

競馬場・レース番号の代わりに `--race-id <18桁>` でも指定できます。

---

## 予想と買い目の出し方

### EVモード（`--ev`）
`--ev` で**全券種（単勝・複勝・馬連・馬単・ワイド・三連複・三連単）の実オッズ**を取得し、
「払戻が理論価格より割高＝妙味のある買い目」をEV基準で選びます。レポートに出るもの：
- **🎯 一番のおすすめ買い方**（的中率×期待値のバランス最良の1点）
- **券種ごとの比較表**（代表買い目・オッズ・的中率・EV）
- **買い目プラン**（軍資金をケリー基準で配分・何円ずつ）
- **ワイドの軸流し／ボックス**（当てやすい買い方）
- 妙味が無ければ**見送り**（軍資金を使い切らない）

### リスク許容度（`--style`）
`conservative`（堅実）/ `balanced`（既定）/ `aggressive`（穴狙い）。

### Geminiに予想させる
- **`--gem`（推奨）**：指示を省いた圧縮データを出力 → 作成済みの「予想用Gem」に貼る。
  初回だけ `--gem` 実行で `reports/gem_instructions.txt` が出るので、その中身を
  Geminiの「Gemを作成」の指示欄に貼って予想用Gemを1つ作っておく。
- **`--gemini`**：指示文込みのフル依頼文（新規チャット向け）。
- どちらも**Gemini Pro契約をそのまま活用・追加費用ゼロ**（人が貼る方式）。

### 本日の狙い目スキャン（`scan`）
本日の全レースを分析し、妙味(EVプラス)のあるレースだけをランキング表示します。
```powershell
python keiba.py scan --track 高知 --budget 5000
```

---

## データ収集・勝率モデル

過去レースを集めて `train` するとLightGBM勝率モデルが `predict` に使われます。
```powershell
python keiba.py collect-day --track 大井   # 1開催分を収集（約20分）
python keiba.py train                       # 30レース以上で学習
```

---

## クラウド自動化（GitHub Actions）

リポジトリ [toybox-anime/keiba-ai](https://github.com/toybox-anime/keiba-ai) で、**毎晩のデータ収集が
クラウドで自動実行**されます（PC不要・無料）。最新データは `git pull` で取り込めます。

- **毎晩の収集＋学習**：`.github/workflows/collect.yml`（JST 20:17/21:47）。API不要・無料。
- **朝の自動予想（Gemini API）**：`.github/workflows/predict.yml` は**現在停止中**。
  有料APIの費用対効果が見合わなかったため無効化（`schedule` をコメントアウト）。
  再開したい場合は predict.yml の `schedule` を復活させ、Gemini APIキーを課金設定にする。

> ローカルのWindowsタスクは廃止済み。収集はクラウドが担当します。

---

## 初回セットアップ

```powershell
cd C:\Users\kynmt\keiba-ai
pip install -r requirements.txt
```
実行は `python keiba.py <コマンド>`（ランチャー経由・インストール不要）。
Claudeに書かせたい場合のみ `setx ANTHROPIC_API_KEY "sk-ant-..."`（API課金・任意）。

---

## 注意・免責

- **馬券は自己責任・20歳以上・余裕資金の範囲で。** 的中も利益も保証しません。
- **控除率の壁**：地方競馬は控除率約20〜25%。本命(人気馬)を機械的に買う戦略は、市場が正しく
  価格を織り込んでいるため**長期的にはマイナス**になりがちです。勝てる可能性があるとすれば
  「本命当て」ではなく「**オッズが理論より割高な買い目(EV妙味)**」を拾う方です。
- 取得は楽天競馬のrobots.txtに従い**60秒間隔**を厳守。利用規約も各自で確認してください。
- オッズは発走前の暫定値です（確定は発走時）。

---

<details>
<summary>開発者向け：構成・既知の課題</summary>

### データの流れ
```
collect → dataset → train → model ┐
  ↓                                ↓
scraper → parser → features → betting(EV/ケリー) → report/gemini
predict-day → Gemini API → ledger（答え合わせ台帳）→ grade → record.md
```

### 主なモジュール
| ファイル | 役割 |
|---|---|
| `scraper.py` | 60秒間隔の取得＋キャッシュ（オッズは短命キャッシュ）|
| `raceid.py` / `schedule.py` | RACEID解決（競馬場名→開催ID）|
| `parser.py` / `result.py` | 出馬表・結果・払戻ページの解析 |
| `odds.py` / `ev.py` | オッズ解析・期待値/ハーヴィル/ケリー |
| `betting.py` | 軍資金からの買い目配分・券種比較 |
| `report.py` / `gemini_client.py` | 予想依頼文生成・Gemini API呼び出し |
| `grading.py` | 予想台帳・答え合わせ・回収率 |
| `model.py` / `train.py` / `dataset.py` | 勝率モデル |
| `cli.py` | コマンド（predict / scan / collect / train / auto / grade …）|

### 既知の課題
- **採点(grade)の解析にバグ**：結果ページの着順誤読・払戻ページで勝ち馬が固定される
  ケース（特にばんえい）があり、**回収率の数字は現状信頼できない**。競馬場別の払戻パーサ修正が必要。
- 意味のある検証は「◎(本命)当て」ではなく「**EV妙味買い目の回収率**」の方。

</details>
