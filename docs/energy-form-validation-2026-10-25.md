# v0.3.0 検証報告 — 自由境界の安定性と波動精度

## 1. 文脈

対象はIssue #9の残件と、その調査で再現したIssue #12〜#16。比較基準はv0.2.0 / f200711である。今回の完了基準は、再現した不安定性・入力遮断・ロッキング・表示の不整合を修正し、エネルギーと解析解で確認できる状態にすること。

モデルは均質・等方・線形の平面ひずみ弾性体で、格子は一様な長方形。実材料の同定、非線形・異方性、Kirchhoff板理論は対象に含めない。「安定性」は誤差が自発的に増幅しない性質であり、「精度」は解析解との差である。

## 2. 核心的な論理

### 証拠

v0.2の加速度作用素を単位ベクトルから組み立てた。9×9、dx=dy=0.02 m、全自由境界、内部源モード、減衰なしの場合、ν=0.25で最大虚部約913.584、既定dtでの陽解法の最大増幅率は約1.007331/stepだった。ν=0.49では正の実部約26340.96も現れた。丸め誤差より十分大きく、既定駆動例の2000ステップ成功を一般安定性の証拠にできないことが確認された。

角部の法線・せん断条件を局所的に同時補正する試案でも、ν=0.25の最大増幅率は約1.008015で、不安定性は残った。したがって、角部の代入順序だけを修正する案は不採用とした。

### 採用した方式

[solver.py](../src/solver.py)の計算系をscikit-fem＋SciPyへ移行した。Q1要素の弾性エネルギーから対称剛性Kを組み立て、行和集中質量Mを使用する。自由表面力は弱形式の自然境界条件になり、物理節点の上書きや角部ghost点は不要となる。

全積分Q1でもν=0.4999のRayleigh試験に失敗したため、体積ロッキングをIssue #14として扱った。偏差ひずみを全積分し、体積ひずみだけを要素平均へ射影するB-bar方式を採用した。要素中心1点の求積は、scikit-femの公開quadrature APIで明示する。intorder=1はこのライブラリでは2×2点になるため、1点積分の代用にはしていない。

平面ひずみの偏差エネルギーと体積エネルギーはともに非負であり、集中質量は正。このため正確な算術ではKは対称半正定値、M^{-1}Kの固有値は実・非負となる。接続した自由格子の零モードは並進2個と回転1個である。単一要素でも余計なhourglass零モードが生じないことを確認した。詳しい式と時間刻みの上界は[計算方式の説明](../STRESS_FREE_BC_IMPLEMENTATION.md)に記載した。

### 仮説と反論

- 「角部のghost点を補正すれば十分」は、今回の固有値試験では支持されなかった。
- 「有限要素法なら精度も自動的に正しい」は支持されない。全積分Q1のロッキング失敗が反例であり、B-bar採用後も格子収束と解析解を確認した。
- 「陰解法だから無条件に安全」ではない。今回は正の質量と半正定値の剛性を持つ系を直接解く。旧固定点反復の無条件収束を主張していない。
- 「2次要素へ移行したので旧4次内部差分より劣る」という反論に対しては、内部ステンシル次数と境界を含む解法全体の安定性・精度を分ける。今回選んだのはエネルギー構造と測定可能な2次収束である。

## 3. 重要ポイント

| Issue | 確認した問題 | 対応 |
|---|---|---|
| [#9](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/9) / [#12](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/12) | 自由境界の複素固有値・不安定増幅 | エネルギー形式、自然自由境界、正の集中質量へ置換 |
| [#13](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/13) | 駆動左端の内側に固定帯が残り入力を遮断 | 境界入力時は列0のみ規定し、左固定帯を除く。他の固定壁は保持 |
| [#14](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/14) | 全積分Q1の近非圧縮ロッキング | B-bar体積射影と近非圧縮Rayleigh回帰試験 |
| [#15](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/15) | 自動CIがなくMacローカル確認のみ | Linux/macOS/Windows・Python3.14のRuff、pytest、物理検証 |
| [#16](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/16) | 表示の縦横比・pixel中心と節点の不一致 | equal aspect、半格子外縁、節点間表示範囲と回帰テスト |

Pythonは3.14.9。主要実行時依存はNumPy2.5.4、SciPy1.18.1、scikit-fem12.0.2、Matplotlib3.11.2、Pillow13.0.0。devグループにpytest9.1.1、ruff0.17.2を追加した。更新時のPyPIメタデータと一致を確認した。MLX/Metal依存と固定点反復は削除した。

### ライブラリと機会費用

scikit-femは弾性弱形式・求積・自由度配置の組立に使用し、独自の要素積分器を新作しない。SciPyは疎行列演算と一度の因子分解に使用する。NumPy float64を全体で使い、CPU/GPU間の転送をなくした。Pillow・Matplotlibは維持した。

移行によりMetal実行と旧内部4次差分は失われる。そのまま旧ghost方式を残す案は、再現した不安定性を残すため採用しなかった。GPU移植や複数backendの追加は、検証済みの単一方式を維持する費用を増やすため今回行っていない。処理速度の優劣は測定しておらず、SciPyの方が速いとは主張しない。

## 4. 具体例と検証結果

### エネルギー・剛体モード

- 自由境界の剛性は丸め誤差の尺度で対称、非負。ν=0、0.25、0.45、0.49、0.4999およびdy/dx=0.5、1、2を確認した。
- 2×2の単一要素と9×9格子で、零モードが剛体3個だけであることを確認した。
- 減衰・入力なしの既定ν=0.25では、両時間積分を10000ステップ実行。他の条件は1000ステップずつで、14条件・計32000ステップを確認した。
- 保存する離散エネルギーの最大相対変動は約7.8×10^{-14}。判定基準は10^{-8}。これは正確な算術での保存則と整合するが、浮動小数点で完全一致するとの主張ではない。

### 波速と格子収束

一波長あたりN区間。誤差は対応する解析速度に対する相対値である。

| N | bulk P/S速度誤差 | Rayleigh ν=0.25 | Rayleigh ν=0.4999 |
|---:|---:|---:|---:|
| 16 | 0.6413% | 0.6716% | 0.6809% |
| 32 | 0.1606% | 0.1683% | 0.1723% |
| 64 | 0.0402% | 0.0423% | 0.0450% |

P/Sの参照は水平周期の平面波。Rayleighの参照は半空間の特性方程式

$$
(2-r^2)^2=4\sqrt{1-r^2}\sqrt{1-\beta r^2},\quad
r=c_R/c_s,\quad \beta=(c_s/c_p)^2
$$

の非零根である。深さ5波長の両面自由ストリップを水平方向にFourier縮約し、最低2モードを求めた。両方のモード速度と表面質量局在を確認した。有限深さによる2枝の差も残るため、無限半空間と完全一致する試験ではない。

Nを2倍にすると誤差はほぼ1/4となり、この条件で2次の空間収束が確認された。これは任意の波長・格子で同じ誤差になるという意味ではない。時間2次収束は、独立に求めた固有モードの時間応答と解析的余弦関数で検証した。

### 自由端反射

ν=0、正面入射P波のGaussian packetでは横連成がなく、自由端の変位反射係数は+1。200→400→800区間で波形の相対L2誤差が約1/4ずつに減少した。

| 方法 | 800区間の反射peak / 入射peak | 波形相対L2誤差 |
|---|---:|---:|
| explicit | 0.999948 | 0.5007% |
| implicit | 0.999937 | 0.5353% |

これは正面入射・ν=0の基準例であり、斜め入射やP/Sモード変換の全組合せを検証した値ではない。

### 実行手順と互換性

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked pytest -q
uv run --locked python validation/validate_physics.py
uv lock --check
uv pip check
```

ローカルでは19テストと12のsubtest、Ruff、lock・依存互換性を確認。fixed/free_top/stress_free_allとexplicit/implicitの6組について、既定入力の2000ステップ実行と内部への伝播も確認した。実GIFを再読込し、フレーム数・遅延時間を検査する既存回帰試験も維持した。

[CI workflow](../.github/workflows/ci.yml)はPython3.14でLinux/macOS/Windowsの同じ検査を実行する。実行結果は[GitHub Actions](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/actions/workflows/ci.yml)とPRのChecksで確認できる。キャッシュ・生の計測記録はfeedback/へ保存し、Gitの公開内容に含めない。

### 移行

配列はNumPy float64で、形状(1, ny, nx, 1)は維持。`ElasticOperator`が旧`FiniteDifferenceOperator`を、`solver.operator`が旧`solver.fd`を置き換える。`finite_diff_order`、`implicit_iterations`、`implicit_tolerance`は削除した。格子・材料・dt・減衰・時間積分を変更する場合は再初期化する。

## 5. 平易な説明

わかりやすく言うと、境界の値をその場で継ぎ足す方法から、弾性エネルギーと整合する行列を組み立てる方法へ変え、波の速さ・反射・エネルギーが基準解に近づくことまで確かめた、ということである。

## 6. 自己批判的考察

安定性の論理は、正の質量・非負の弾性エネルギー・指定された一様Q1格子・支持する材料範囲に基づく。数値での対称性や零固有値は、λが大きい場合も行列全体の尺度と機械精度に応じて判定した。小さい成分ごとに同じ絶対許容差を使って丸め誤差を不具合と判定してはいない。

この検証はメッシュ独立の誤差保証ではない。既定の薄い有限領域は、Rayleigh試験の深い周期ストリップとは異なる。非線形材料・実験との同定・Head Waveの同定・全角度の反射・3D・不規則格子は扱っていない。対話GUIの実操作や全560ステップの既定GIFを人手で確認する試験も行っていない。

Gemini Antigravity（gemini-4-argon-medium）は今回のsolver.py、config.py、validation/validate_physics.pyを読み、自由度写像・B-barの正値性・集中質量・時間係数・dt上界に明確な不具合を指摘しなかった。これは独立レビューであり証明の代替ではない。Claude Codeは選択モデルのアクセスエラーでレビューできず、回答を補作していない。

検証した事実と数理構造への確信度は0.95（主観的な自己評価）。実材料への適用精度や未知の入力条件の確率評価ではない。

## 7. 結論

再現した不安定固有モード、駆動源の遮断、近非圧縮ロッキング、表示の幾何不一致を修正した。Issue #9の確認基準に対して、数理構造・回帰テスト・独立の解析例・複数格子の検証を用意した。新たな失敗を既知のまま完了扱いにせず、各Issueの条件を確認してからリリースする。

clean-code-guard: 独自要素ライブラリ・余分なbackend・反復wrapperを作らず、行列を一度組み立てて使用する構造へ整理した。数式を行数だけで分割していない。docs-guard: 実装・設定・CLI・削除API・既定値・検証条件を照合した。旧v0.2報告は履歴として明示している。

## 8. 参考文献

確認日2026-10-25。更新年を確定できないWeb資料は推測していない。

- Gustafsson, T. and McBain, G.D. (2020), *scikit-fem: A Python package for finite element assembly*, JOSS 5(52), 2369. ライブラリ・弱形式組立の参照。https://doi.org/10.21105/joss.02369
- scikit-fem公式文書・models.elasticity実装（本文・実装確認）。https://scikit-fem.readthedocs.io/ 、https://github.com/kinnala/scikit-fem/blob/master/skfem/models/elasticity.py
- A.F. Bower, *Applied Mechanics of Solids*, §8.6（locking、選択的低減積分、B-barの本文確認）。https://solidmechanics.org/Text/Chapter8_6/Chapter8_6.php
- MOOSE/Idaho National Laboratory, *NewmarkBeta*（前回公式Markdown本文を確認）。https://raw.githubusercontent.com/idaholab/moose/master/framework/doc/content/source/timeintegrators/NewmarkBeta.md
- SciPy sparse linear algebra（API確認、実測は本リポジトリの検証による）。https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
- NumPy・SciPy・scikit-fem・Matplotlib・Pillow・pytest・RuffのPyPI metadata（最新版照合）。https://pypi.org/
