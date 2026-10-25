# ソースコード監査と改善記録（v0.2.0）

> これはv0.2.0時点の履歴である。現行の計算方式・検証結果は[v0.3の検証報告](energy-form-validation-2026-10-25.md)を参照。

## 1. 文脈

対象は本リポジトリのPythonソース・設定・テスト・README・境界説明。比較基準は1941251、成果物はv0.2.0のタグが指すコミットである。「妥当性」は数理モデルと離散化の前提、「正確さ」は実装と出力の整合、「無駄」は重複処理・不要な依存・追跡対象として不適切な生成物を指す。

計算は均質・等方の線形弾性、平面ひずみ、Apple Silicon上のMLXを前提とする。一般の薄板曲げモデルや、検証済みの表面波解析ソフトとして評価していない。

## 2. 核心的な論理

### 証拠と対応

| 優先度 | 確認した事実 | 修正・検証 | Issue |
|---|---|---|---|
| 高 | dtが遅いS波とdxだけを使用 | 最速P波、dx/dy、4次差分の固有値上界から算出 | [#5](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/5) |
| 高 | 自由境界で5層の物理格子を同じ値で上書き。上・右の勾配符号も逆 | 領域外2層のghost点と連成表面力条件へ移行。物理領域を保持 | [#6](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/6) |
| 高 | 陰的更新の時間中心ずれ、非収束の無視 | 対称な平均加速度型、反復中の境界適用、非収束時の例外と状態保護 | #5 |
| 高 | 変位全体の減衰で静的剛体運動も変化 | 速度増分の減衰へ変更し、並進・回転の不変性を確認 | #5 |
| 中 | 源の1ステップ遅れ、radial源がindex座標を使用、内部源が固定帯を上書き | 新時刻、物理座標の分極、内部源適用後の固定境界マスク | #5 / #6 |
| 中 | y成分のNaNをPythonのmaxが隠す場合がある | MLXのmaximumで伝播し、表示・GIFで非有限値をエラー化 | #5 / [#7](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/7) |
| 中 | 表示のindex座標とメートル座標が不一致、端数ステップの過不足 | 座標と節点間長を同期し、端数を含む厳密な回数で実行 | #7 |
| 中 | GIF最大値初期値1 m、遅延時間の単位違い | 実測最大値、全ゼロ時の退避範囲、100 msの実ファイル確認 | #7 |
| 中 | 既存の2本のテストは設定不足・importエラーで失敗。assertによる検証も不足 | 実行可能なunittestへ変更し回帰テストを追加 | #4–#7 |
| 中 | READMEのλ=250 Pa、物理検証済み・P/S分離・global4次など根拠を超える説明 | λ=400 Paへ訂正し、変位成分表示と確認範囲を明記 | #4 / #6 |
| 中 | Python3.10環境と古いlock、生成済みbytecodeと.DS_Storeの追跡 | Python3.14.9と最新安定依存へ更新し、17ファイルの追跡を解除 | [#8](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/8) |
| 低 | GIFで3成分の全画像を同時保持、不要copy、PNGの往復変換、委譲だけのimageio依存 | 成分ごとに描画・保存、canvasからRGB取得、Pillow直接APIへ統合 | [#10](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/10) |

主な変更先は [config.py](../src/config.py)、[solver.py](../src/solver.py)、[state.py](../src/state.py)、[visualizer.py](../src/visualizer.py)、[gif_exporter.py](../src/gif_exporter.py)、[main.py](../src/main.py)。呼び出し側・既存テスト・設定・説明を同じ変更で同期した。新しい実行時依存は追加していない。境界に関する旧apply_*メソッドはpad_fieldsへ置き換えたため、外部から直接呼び出していた利用者は移行が必要である。

### 仮説と未確認事項

角部のy→x逐次外挿は、両辺の表面力を同時に満たす閉鎖ではない。辺のテストを角部へ拡張すると残差が残る。一般の安定性・反射係数・Rayleigh/Lamb/Head Waveの定量精度は未確認であり、[Issue #9](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/9)を未完了の検証課題として残す。

### 反論と判断

「ghost点への移行だけでは正しい解法にならない」は妥当である。今回の修正は、再現可能な符号・距離・領域上書き・時間更新の誤りを解消する範囲である。境界を含むエネルギー安定な方式への全面置換は、行列・境界・検証基準の再設計を伴う。既定値での有限な実行を一般的な安定性証明に読み替えず、教育・探索用として限界を公開する。

## 3. 重要ポイント

- Pythonはプロジェクトの3.14系列へ更新。検証環境は3.14.9であり、システムPython全体は変更していない。
- MLX / mlx-metal 0.32.4、NumPy 2.5.4、Matplotlib 3.11.2、Pillow 13.0.0を使用。更新時にPyPI公開メタデータと照合し、直接依存の最新安定版との一致を確認した。推移的依存もuv lock --upgradeで再解決した。
- imageioは不要になったため削除。正確な依存・ハッシュはuv.lockが基準である。
- bytecode16個と.DS_Storeの計17ファイルをGitのindexから除外し、実ファイルは保持した。feedback/、.venv/、.serena/も除外。資料PDF・作例GIF/動画・uv.lockは保持した。
- 主要テストは15件。数値式の局所的な正しさと、連続体の定量精度を区別する。
- PR・マージ・GitHub Release・作業ブランチの削除は追加依頼に従って行う。Issue #9は未検証のまま閉じない。

### ライブラリの採否

| ライブラリ | 判断 | 理由・限界 |
|---|---|---|
| Pillow | 採用・直接利用 | 既存依存でGIFの複数フレームとミリ秒指定を直接扱える。imageioの薄い委譲を削除できる |
| Matplotlib | 維持 | 既存の等高線・対話表示・canvasを利用できる。別の表示基盤を導入する根拠はない |
| MLX | 維持・更新 | Apple Siliconを対象とする既存の配列・畳み込み基盤。数値的な境界条件はライブラリ変更だけでは正されない |
| NumPy | 維持・更新 | 座標とCPU検証に既に使用。演算基盤の全面置換は未実施 |
| SciPy sparse.linalg | 今回は不採用 | 疎な線形系の直接・反復解法は将来候補。ただし境界を含む行列の再定式化・転送・検証が必要。置換の実測比較は行っていない |

全面的なbackend置換の機会費用は、今回実証できた不具合修正と回帰検証を遅らせることである。比較速度を測定していないため、「MLXの方が速い」「SciPyなら正しい」とは判断していない。

## 4. 具体例と検証

### 再現例

- 3ステップ、更新間隔2: 表示は以前4ステップ進むが、修正後は3で終了。GIFは2フレームを生成し、最後のタイトルはStep 3となる。
- 1 mm以下の場: 最大値1 mから始める旧GIFではほぼ描けない。修正後は実際の対称スケールを使用する。
- 静的な平行移動・微小回転: ひずみゼロなので加速度はゼロである。減衰を有効にしても静的場を変えないことをテストした。
- 陰的反復を1回、許容差を1e-15にする: 失敗を無視せずRuntimeErrorを出し、元のstateを保持する。
- x成分が有限・y成分だけNaN: 不正値を検出し、正常なGIFとして保存しない。

### 実行した確認

検証環境: macOS arm64 / Python3.14.9 / MLX0.32.4、NumPy2.5.4、Matplotlib3.11.2、Pillow13.0.0。

```bash
uv run --locked python -m unittest discover -s test -v
uv lock --check
uv pip check
uvx ruff check src test --select F
uv run --locked python run.py --help
git diff --check
git ls-files -ci --exclude-standard
```

- 15件のテストが合格。多項式の差分、別サイズでの演算子再利用、辺の表面力、物理座標の分極、時間2次収束、非収束時の状態保護、固定壁と内部源、NaN、表示の座標・回数、実GIFのフレーム数・100 ms遅延を含む。
- 既定材料・格子・源で2000ステップ: explicitの最大成分変位は約0.001193337 m、implicitは約0.001192299 m。有限かつ0.1 m未満を確認した。この閾値は発散監視用であり誤差許容値ではない。
- fixed / free_top / stress_free_allの各境界について、explicit / implicitを200ステップずつ実行し、有限・有界を確認した。
- uvのlock整合・依存互換性確認、未定義名・未使用import確認、空白エラー確認を実施。
- 生の実行記録はGit除外のfeedback/へ保存する。対話GUIの実操作、全560ステップの既定GIF書出し、他OS・他Python、全パラメータの安定性調査は実施していない。
- GitHub ActionsによるCIは今回新設していない。ローカル検証とCIを混同しない。

## 5. 平易な説明

わかりやすく言うと、計算式・境界・表示にあった再現可能な誤りを直し、確認していない物理精度を「検証済み」と呼ばない状態に整えた、ということである。

## 6. 自己批判的考察

修正後も、辺は2次閉鎖、角部は逐次近似である。空間全体の4次収束、長時間エネルギー安定性、近非圧縮材料や極端な非等方格子、表面波の位相・反射振幅は未検証である。標準例の成功はこれらの代替証拠にならない。

独立レビューはGemini Antigravity CLI（gemini-4-argon-medium）から得た。共有したのは1941251での問題と今回の修正方針・検証条件であり、最終差分を直接読み込んだレビューではない。辺の修正は限定的な改善として妥当だが、角部・global4次・一般の安定性を主張できないとの意見を採用した。別の回答にあった「必ず指数的に発散する」という断定は、本実装の証明・測定がないため採用しなかった。

Claude Codeは選択モデルへのアクセスエラーで回答を得られず、独立レビュー未実施である。回答を補作していない。レビューの賛同自体を正しさの証明にはしていない。

本報告の確信度は0.9（再現できた不具合と検証結果に対する主観評価であり、物理計算の精度・信頼度を示す統計量ではない）。

## 7. 結論

既存の具体的な不具合を修正し、Python3.14と更新した依存関係で実行可能な回帰検証を整えた。Pillow直接利用・重複した境界処理・画像保持・追跡不要物の整理で実装を簡素化した。全体モデルの定量妥当性はIssue #9に残る。

clean-code-guard: 最終差分の重複・死んだ処理・不要依存・状態保護を確認。数式の連続性を優先し、反復式を行数目標だけで分割していない。docs-guard: CLI・設定・関数・既定値・説明・内部リンクを照合。英語の既存READMEには日本語推敲用スキルは適用していない。

## 8. 参考文献

確認日はいずれも2026-10-25。公開年・更新年がページ上で確定できない資料は推測していない。

- J. Bleyer, *Linear elasticity*, Numerical Tours of Computational Mechanics with FEniCSx（本文確認）。https://bleyerj.github.io/comet-fenicsx/intro/linear_elasticity/linear_elasticity.html
- MOOSE project / Idaho National Laboratory, *NewmarkBeta*（公式Markdown本文確認、HTMLページは403）。https://raw.githubusercontent.com/idaholab/moose/master/framework/doc/content/source/timeintegrators/NewmarkBeta.md
- Pillow contributors, *Image file formats — GIF*, Pillow13.0.0 documentation（本文確認）。https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
- MLX contributors, *mlx.core.conv2d*, MLX0.32.4 documentation（API・テンソル形式確認）。https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv2d.html
- SciPy contributors, *Sparse linear algebra*, SciPy1.18.0 documentation（候補機能確認、実装・速度比較なし）。https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
- PyPI metadata（更新版の照合）: https://pypi.org/pypi/mlx/json 、https://pypi.org/pypi/numpy/json 、https://pypi.org/pypi/matplotlib/json 、https://pypi.org/pypi/Pillow/json
- 既存README掲載の関連文献: K. Uenishi (2009), *ON THE NUMERICAL ANALYSIS OF SOLID DYNAMICS USING THE MOVING PARTICLE SEMI-IMPLICIT (MPS) METHOD*, DOI https://doi.org/10.24546/81001946 （今回は論文本文を再確認しておらず、本有限差分法の検証根拠には用いていない）。
