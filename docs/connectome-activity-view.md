# Rate 神経活動の表示出力

CooSenpAIの設定 → 開発者 →「脳活動を開く」の独立ウィンドウで、実rateモデルの刺激応答を表示するため、`connectome-helper rate` のevaluate成功resultに任意の`activity`を追加する。回路・パラメータ・readout・学習・判断を変更しない。同じinference loopの初期状態と実更新直後を採取し、step数のceil(steps/16)間隔と最終stepを選ぶ。32 stepsの場合は0,2,...,32の17フレームである。full historyや再計算を作らない。最終activityは従来のf64 inferenceと完全一致し、表示用コピーだけfloat32とする。

座標ありニューロンをbody ID順に並べ、floor(i * positioned_count / sample_count)で最大1,024件を選ぶ。全211,577ニューロン・26,028,386対の計算はそのまま実行する。座標141,781件、欠損69,796件のうち1,024実細胞体を描画する。tosomaLocationや架空座標で欠損を埋めない。MaleCNSの中枢神経系の細胞体代表点であり、神経線維・全ニューロンの形状ではない。

## 出典・利用条件・加工

一次資料: [MaleCNS公式ダウンロード](https://male-cns.janelia.org/download/)。元ファイル: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather`。

MaleCNS v1.0 / FlyEM・HHMI Janelia Research Campus、University of Cambridge、MRC Laboratory of Molecular Biology、Google Research。既存rate pack manifestが示すライセンスはCC BY 4.0（[利用条件](https://creativecommons.org/licenses/by/4.0/)）。元データSHA-256は `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`。

`assets/malecns-soma-sample.json`はbodyIdとsomaLocationを現行packのbody IDへjoinし抽出した加工物（52,964 bytes）。元座標は変更しない。表示時だけ中心移動・一様スケール・Y/Z反転を行う。活動表示はf64→f32変換とlittle-endian hex encodingを行い、輝度に対数スケールを使う。これはMaleCNSの生物学的活動記録ではなく、本repoのrateモデルの計算結果である。

対象graph fingerprint: `dcd3b2e9c97191c21fae772c8417eef8bbe9b370b1cf7fc19e52af685238afc4`。helperはこのfingerprintと総数を一致確認し、各body IDをgraphに解決する。他packへ座標を流用しない。未対応の場合は`positions-unavailable`を出す。

再生成にはpyarrowのあるPython環境と既存packを参照するartifactが必要。出力assetだけをgit管理し、Feather・全pack・学習stateはgitへ入れない。

```sh
python scripts/malecns_activity_positions.py \
  --annotations /path/to/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --artifact /path/to/malecns-frozen-response-learning-artifact-r14.json \
  --output crates/habitua-connectome/assets/malecns-soma-sample.json
```

## 通信・保持

`schema=rate-activity-v1`, `encoding=f32-le-hex`。input_id、image_sha256、graph_fingerprint、total_neurons、positioned_neurons、total_steps、hmax、coordinate_source、neurons[{body_id,position}]、frames[{step,values}]を出す。body_idはuint64を安全に扱う10進文字列。valuesはneurons順のfloat32 little-endianを8桁hex化したもの。各activityは最大1,024点・17フレームで、実測198,901 bytes（32steps）。全envelopeは既存CooSenpAI JSONL上限256KiBに収める。

helperは最新の新規計算1件のactivityだけ保持する。既存response_cacheにはactivityを追加せず、SHA cache hitは`image-cache-hit`、canonical observation replayは`observation-replayed`を出して再計算しない。次のevaluate開始で最新activityを破棄し、audio/throttled/zero-response/失敗のあとに古い計算を流用しない。stateや観測ledgerにもactivityを保存しない。CooSenpAIはraw traceからactivityを除去し、実活動を取得した直近5観測を専用メモリに保持する。キャッシュ/未取得だけでは枠を消費せず、過去取得分を参考表示と明示して維持する。各取得時の全体判定・module評価と、helper正本の寸法+RGB8 SHAに一致した640×360以内・PNG最大1MiBの画像プレビューを対応付ける。軽量一覧と選択1件だけの読み取りIPCを使い、画像・活動を一般traceやディスクへ保存しない。Shadowは参考判定であり実適用と区別する。可視化操作は読み取りだけで、evaluate/feed/commitを発行しない。

## 利用手順

新helperは追加引数・sidecar設定なしで組込assetを利用する。

```sh
cargo build --release -p habitua-connectome --bin connectome-helper
```

このバイナリを既存CooSenpAI judge moduleのexecutableとして使用し、従来のartifact/state指定を維持する。CooSenpAIも本変更をbuildし、設定 → 開発者 →「脳活動を開く」を選ぶ。設定を閉じても独立ウィンドウは使え、閉じて再度開くと保持中の直近5件を取得できる。既定は最新取得分へ追従し、過去選択は新観測でも維持する。選択記録が5件から外れると説明して最新取得分へ移る。新規の画像観測を待つ。shadow/followはどちらでも表示できる。旧helperや未対応packは未取得を表示する。

表示は完了済み計算の推移を0.4秒間隔で再生し、最終状態で停止する。32回目（全32回）は1枚の画像の活動値を32回更新した最終結果であり、観測件数や秒数ではない。既定の活動強度は従来の対数輝度を維持する。補助の変化量表示は同じ観測・モジュールの前表示フレームとの差を青（減少）・暗色（0）・橙（増加）で表示する。観測内の全隣接表示フレームの最大絶対rate差を共通の線形尺度として実値を凡例に示し、観測間の色比較には使わない。初期状態は比較対象なし、全差分0なら0を表示する。表示切替・再生はhelper計算や学習を実行しない。
