# Connectome plugin の通信境界

`habitua-connectome` の `connectome-helper` は、MaleCNS rate の資産と学習状態を
`habitua` core から分離する JSON Lines over stdio の実行ファイルです。
標準出力は応答 JSON に限定し、診断ログは標準エラー出力へ出します。

## MaleCNS rate helper と CooSenpAI の判断役契約

MaleCNS の凍結 CNS 応答を使う rate helper は、CooSenpAI の
`coosenpai-observation-v1` 契約を使う。実行可能ファイルは引数なしで起動できる。
学習済み artifact と書込み可能な状態ファイルのパスを
`HABITUA_RATE_ARTIFACT` と `HABITUA_RATE_STATE`、または `--artifact` と `--state` で指定する。
`connectome-helper --artifact <path> --state <writable-state-path> --pack <absolute-path>` と直接渡せる。
state 未指定時に artifact の隣へ書く動作はない。従来の
`connectome-helper rate --artifact <path>` という起動形式は使えるが、state の指定が必要。

pack は `--pack <absolute-path>`、`HABITUA_RATE_PACK`、artifact の `pack_path` の
順に選ぶ。引数と環境変数は絶対パスを要求する。artifact の `pack_path` が相対なら
artifact のディレクトリを基準に解決する。CooSenpAI は利用者の保存先を絶対パスで
`--pack` に渡す。選んだ pack の `rate_manifest.json` の SHA-256 を artifact と照合してから
graph を読み込み、読み込んだ graph の fingerprint も照合する。
配布用 artifact は `dist/connectome/malecns-frozen-response-learning-artifact-r14.json`。
`python3 tools/prepare-connectome-distribution.py artifact <source-json> dist/connectome` で
SHA-256 ファイルとともに作成し、pack は
`python3 tools/prepare-connectome-distribution.py pack <rate-full-dir> <distribution-artifact> <output-dir>`
で別配布用 tar と帰属情報付き JSON にする。

起動時の pack・state 設定エラーは標準出力に共通 envelope を 1 行返して非ゼロ終了する。
この行の `id` は空文字、`session` は null。`error.code` は
`pack-missing`（ディレクトリまたは manifest の欠落）、`pack-unreadable`、
`pack-path-invalid`、`pack-manifest-mismatch`、`pack-invalid`（pack 内のファイルや
形式の検証失敗）、`graph-fingerprint-mismatch`、`state-path-required` のいずれか。
`error.message` に利用者の発言、名前、フルパスは含めない。呼び出し側は非ゼロ終了時にも
この応答を読み、原因を画面に表示する。

rate helper の `evaluate` は、契約で定義された `input_id`、`image_path`、`stream_id`、
`at_ms`、`timestamp_s`、`feature_schema`、`features`、`observation` を受け取る。
`observation.kind` が `visual` の場合は `app` と `target` を必須とし、`audio` の場合は
`app=null`、`source`、`text`、`image_path=""` とする。`context` は JSON object とする。
未知の項目、時刻の不一致、画像パスの不一致、feature schema の不一致はエラーにする。
音声は RetinaMap で処理できないため、`novelty` と `relevance` を null、`action` と
`external_action` を `hold`、`readiness` を `unsupported-input` として返す。

成功応答の action は `notify`、`silence`、`hold`、readiness は `ready` または
`insufficient-history` などの契約値を使う。診断用の旧名称は `diagnostic_action` と
`diagnostic_readiness` にだけ保存する。`shadow=true` の場合は診断 action にかかわらず
`external_action=hold` とする。`hold_reason` は `insufficient-history`、
`consensus-pending`、`shadow`、`throttled`、`unsupported-input` のいずれかを記録する。
state v2 の shadow decision は内部 `action=notify/silence` と
`external_action=hold`、`hold_reason=shadow` を同時に保存できる。loader は
`readiness=ready` と `action/diagnostic_action` が `notify/notify_change` または
`silence/suppress_no_change`、`diagnostic_readiness=evaluated`、`novelty` と `relevance` が
両方存在するこの組合せだけを受理し、保存済み decision・学習・履歴を変更しない。
両スコアには既存の0〜1の有限値検証も適用する。
それ以外の non-hold action と hold_reason の不整合は拒否する。
複数要求は常駐プロセス内で要求順に処理し、response の
`id` と `session` を要求と一致させる。

`feed` は `event_id`、`input_id`、`sign`、`strength`、`event_time_s`、`received_at_s`、
`revision`、`cancelled`、`source` を必須とする。`source` は `automatic` または
`explicit` である。同じ event の新しい revision だけを適用し、訂正・取消を state に
保存する。適用結果は `applied` と `applied_at_s` で返し、適用済みでない再送も
`feedback_history` に記録する。餌の減衰開始時刻は `event_time_s` であり、画像 SHA-256
完全一致の事例だけに補正を適用する。

state は version 2 の JSON とし、同じディレクトリの一時ファイルへの書込み、`fsync`、
rename、親ディレクトリの `fsync` の順で原子的に更新する。JSON 破損、異なる artifact
hash、未知の state version は起動時に拒否する。CooSenpAI 側はこの helper を絶対パスの
executable として常駐起動し、artifact の環境変数を継承させ、timeout・終了・契約違反時
にプロセスを破棄して次回に再起動する。

rate helper の失敗応答も、必ず次の共通 envelope を返す。JSON の解析に失敗した場合は
取得できた `id` と `session` を使い、取得できなければ空文字または null とする。

```json
{
  "v": 1,
  "id": "request-1",
  "session": "session-1",
  "schema": "coosenpai-observation-v1",
  "ok": false,
  "error": {
    "code": "observation_expired",
    "message": "observation body has expired"
  }
}
```

`error.code` は機械的に扱う安定した理由コード、`error.message` は診断用の説明である。
`observation_ledger_capacity_exceeded`、`observation_body_capacity_exceeded`、
`feedback_capacity_exceeded`、`history_stream_capacity_exceeded`、`state_size_exceeded` は
容量上限による拒否であり、同じ要求を再送しても状態は変わらないため non-retryable とする。
`observation_expired` は観測本体の保持期限切れ、`observation_not_feedable` は音声、
throttle、`app=null` などの早期 return で事例記憶の対象外であることを表し、これらも
non-retryable である。`observation_replay_mismatch`、
`observation_replay_unverifiable` も入力の再送契約違反として再試行しない。
`persistence_uncertain` は rename 後の親ディレクトリ同期失敗で、応答を flush した後に
helper が非ゼロ終了する。呼び出し側は状態を再送で上書きせず、再起動後に検証する。

visual の通常成功応答は `feedable=true` と
`feed_target=exact_image_sha256_case_memory` を返す。audio、throttle、`app=null` の
成功応答は `feedable=false` と `feed_rejection_code=observation_not_feedable` を返す。
`feedable` は応答時点でその観測が feed 対象であるかを表す。したがって、期限切れ後の
同一観測 replay は `feedable=false` と `feed_rejection_code=observation_expired` を返す。
観測本体が eviction された ledger entry は `body_status=expired` として保持し、遅延
feed は `observation_expired` で拒否する。expired tombstone に特徴量は保存しない。


## 任意の神経活動表示

rate helperのevaluate resultに`activity`（schema `rate-activity-v1`）を任意追加する。実計算stepの最大17フレーム、実soma座標の最大1,024点をfloat32 little-endian hexで出力する。判断・feed・commitの意味は変わらない。画像cache hit/replay/未対応packでは未取得理由を出し、表示のための再計算は行わない。データの構造・上限・出典・利用手順は [Rate 神経活動の表示出力](connectome-activity-view.md) を正本とする。
