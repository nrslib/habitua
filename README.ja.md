# habitua

[English](README.md)

リポジトリ: <https://github.com/nrslib/habitua>

`habitua` は habituation（馴化、慣れ）を縮めた名前で、同じ刺激が繰り返されると
反応が弱まる、もっとも単純な学習を指します。

`habitua` は、任意の特徴ベクトルに慣れる Rust crate です。評価と学習を
分離しているため、呼び出し側は評価結果を確認してから、保留、学習量の減量、
または経験としての commit を選べます。

画像認識、OCR、音声認識、テキスト embedding は core の責務ではありません。
特徴の意味、正規化、並び、schema の版は adapter が管理し、確定したベクトルを
渡します。

## 最小例

既定の `Habitua` は core の `FlyModel` です。コネクトーム脳は別配布の
`habitua-connectome` plugin として、利用者が用意した pack を別プロセスで使います。
core の生成と評価はデータを network から取得しません。

```rust
use habitua::{Config, Habitua, Model, Timestamp};

let config = Config::default();
let features = vec![0.0_f32; config.dimensions];
let mut model = Habitua::with_config(config)?;
let evaluation = model.evaluate("screen", &features, Timestamp::ZERO)?;
model.commit(evaluation.update, 1.0)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`Timestamp` は呼び出し側が渡します。モデル時刻のために crate が
`SystemTime` を読むことはありません。そのためテストと replay を決定的に
できます。stream ごとに状態は独立します。不正な入力は `Err` になり、panic
にはなりません。

## Model の境界

型付きの差し替え境界は次です。

```text
trait Model {
    evaluate(stream_id, features, at) -> Result<Evaluation, ModelError>
    evaluate_input(stream_id, input, at) -> Result<Evaluation, ModelError>
    commit(update, learning_weight) -> Result<(), ModelError>
}
```

通常の特徴ベクトルは `ModelInput::Features` で渡します。構造化された処理結果を
受ける model は `Model::evaluate_input` を実装できます。組み込みの
`FailureModel` は `ModelInput::Failure` で `FailureObservation { success, result_code,
error_type, state }` を受け、失敗結果には `Failure` 反応を返します。この model は
学習状態を持たないため、同じ失敗を繰り返しても慣れとして抑制されません。

`Evaluation` の共通値は必要なモデルだけが返せるよう `Option` です。短期・長期
novelty と familiarity、完全一致 pattern の `recency` と `pattern_id` は必須では
ありません。モデル固有の診断情報と、まだ反映していない update も持ちます。
weight は有限の `0..=1` です。組み込みモデルでは weight 0 の commit は学習済みの
基準・重みを強化せず、到着、処理位置、完全一致の記録だけを進めます。fractional
weight は学習状態への寄与を比例させます。到着メタデータを持たない custom model は
weight 0 で学習状態を変更してはいけません。

`Habituation<E, M>` は Fly 型の汎用実装です。stream ごとの factory は内部で
`Box<dyn Fn() -> Result<M, MemoryError> + Send + Sync>` として保持するため、公開型
の型引数は `E` と `M` の2つだけです。factory の最初の結果は空でなければならず、
その fingerprint を記録します。その後の結果も同じ fingerprint と空状態である
ことを検証します。`FlyModel` は `Habituation<FlyHashEncoder, DecayingMemory>` の
別名です。既定の `Habitua` は `FlyModel` の別名です。Connectome plugin は core の
registry に組み込まれず、別プロセスの境界で利用します。

`Habituation` の内部では `Encoder` と `Memory` を個別に差し替えられます。
`NumericDeviationModel` と `NearestNeighborModel` はそれらを使わず、同じ `Model`
と永続化の契約へ接続できます。

## 複数の脳

`BrainRunner` は、同じ確定入力を複数の異なる model に評価させます。脳ごとに
識別子と役割、enabled、active／shadow、入力 schema と対象、model 種別・版・設定・
seed、学習方針、反応閾値、資源上限、状態識別値を持ちます。`ModelRegistry` は
型付き factory で model を生成し、runner の境界だけで薄い型消去 wrapper に入れます。

処理順序は固定です。

1. adapter が入力と利用可能時刻を確定する。
2. 対象となる全 enabled 脳が、更新前の状態で評価する。
3. runner が脳ごとの報告と準備状態を記録する。
4. 呼び出し側が active の反応だけで起動方針を評価する。shadow は比較に使う。
5. 脳ごとの方針または明示した weight で commit する。

schema の検証は対象と context の選択後に行います。そのため、1つの実験に異なる
schema や次元の脳を混在させられます。別の対象の脳は `InputMissing` として記録され、
全体の評価を失敗させません。更新世代は stream ごとに独立しているため、独立した
stream の保留更新を個別に commit できます。`promote_shadow` は、入力契約、model
設定、保存形式、学習履歴、状態識別値が互換する場合だけ状態をコピーします。

shadow は起動判断に参加しないという意味であり、学習を止める意味ではありません。
active と同じ学習方針で学習できます。weight 0 で学習を止めても、評価回数、最後の
到着時刻、処理位置、完全一致のイベント記録は進みます。

`EveryEvaluation` は基準不足中も初期標本・代表例を集められます。ほかの学習 mode で
基準不足中の学習を明示的に許可する場合は `bootstrap = true` を指定します。
`Evaluated` になるまで学習させない呼び出し側は `require_evaluated = true` を指定します。
`RepeatedNormal` の連続回数と bootstrap は stream ごとに管理されるため、異なる stream
の入力を混ぜて定着判定しません。
Fly と近傍 model では novelty の反応が出ても、評価済みの入力を安定した通常経験として
学習できます。数値の偏差と失敗は、評価済みで異常があれば異常信号として扱います。

共通の反応報告は `Reaction` です。

```text
Reaction {
    brain, target,
    kind: Novel | Deviation | Absence | Repeat | Failure,
    strength: Option<f32>,
    metrics: 名前・値・単位・方向,
    evidence,
    readiness: Evaluated | BaselineInsufficient | InputMissing |
               Invalid | Expired | Error,
}
```

組み込みの raw 指標は未校正なので `strength` は `None` です。反応は異常確率
ではありません。呼び出し側は readiness と根拠を確認してから、必要な用途だけで
校正済みの指標を組み合わせます。

別の生物回路の model を registry に追加する手順は4段階です。

1. 反応する対象、入力条件、時刻の扱い、学習規則、出力の意味を定義する。
2. `Model`、`PersistentModel`、設定検証、報告変換を実装して typed factory を登録する。
3. 同じ入力 schema を使う shadow の脳を追加し、状態識別値を分ける。
4. `examples/replay.rs` で新たに拾った事象、誤通知、遅延、資源費用を比較する。

未知の model 名、schema 不一致、次元不一致、互換性のない保存状態はエラーです。
既定値への黙った置換は行いません。

## Connectome plugin（別配布・別プロセス）

`habitua-connectome` は Janelia の MaleCNS v1.0 コネクトームを用いた
CPU 発火率モデルです。データのライセンスは CC BY 4.0 です。出典は
<https://male-cns.janelia.org/>、Berg et al., *Cell* (2026),
doi:10.1016/j.cell.2026.08.015 です。

元データを別途取得し、`scripts/malecns_feather_to_parquet.py` と
`connectome-data convert-malecns-rate` で rate pack を生成します。
元データと pack は habitua に同梱しません。コードの MIT / Apache-2.0 と
データの CC BY 4.0 は別のライセンスです。活動表示用の細胞体座標サンプル
1,024 件は CC BY 4.0 の帰属情報とともに同梱しています。
活動表示とデータの帰属は [`docs/connectome-activity-view.md`](docs/connectome-activity-view.md)、
元データの照合用 checksum は [`docs/malecns-v1.0-SHA256SUMS.txt`](docs/malecns-v1.0-SHA256SUMS.txt) を参照してください。

`connectome-helper` は CooSenpAI の判断用プラグインです。学習済み artifact と
書込み可能な状態ファイルのパスを `--artifact` と `--state`、または
`HABITUA_RATE_ARTIFACT` と `HABITUA_RATE_STATE` で指定します。配布用 artifact は
`dist/connectome/malecns-frozen-response-learning-artifact-r14.json` にあります。
pack は同梱しません。別途導入した `rate-full` の絶対パスを `--pack` で渡せます
（次点は `HABITUA_RATE_PACK`、最後は artifact の `pack_path`）。

配布物の作成には `tools/prepare-connectome-distribution.py` を使います。
`python3 tools/prepare-connectome-distribution.py artifact <元の r14 JSON> dist/connectome`
で学習内容を保った配布用コピーと SHA-256 ファイルを作り、
`python3 tools/prepare-connectome-distribution.py pack <rate-full ディレクトリ> <配布用 artifact> <出力ディレクトリ>` で
別配布用 tar と、SHA-256・出典・ライセンス・変換内容を記した JSON を作ります。
tar は `rate-full/` を含むので展開先を選んで設置します。公開・アップロードは別途行います。
helper は `connectome-helper --artifact <配布用 artifact> --state <書込み可能な状態ファイル> --pack <rate-full の絶対パス>`
で起動できます。
通信契約は
[`docs/connectome-plugin-protocol-v1.md`](docs/connectome-plugin-protocol-v1.md) に記載しています。

## Fly model

`FlyHashEncoder` は seed から決定的な疎なランダム射影を作り、上位 `k` 個の
unit だけを残します。ハエの嗅覚系にある疎な符号化に着想を得た工学モデルであり、
生物学的なシミュレーションではありません。`PatternId` は安定した fingerprint
であり、衝突しない ID ではありません。

`DecayingMemory` は発火 unit ごとに短期と長期の重みを持ちます。`t0` で `w(0)`
だった重みの時刻 `t` の値は次です。

```text
w(t) = w(0) exp(-(t - t0) / tau)
```

現在の pattern を強化する前に、次を計算します。

```text
f_short = mean(short_unit_weight(t))
f_long  = mean(long_unit_weight(t))
familiarity = (alpha * f_short + beta * f_long) / (alpha + beta)
novelty = 1 - familiarity
```

学習 weight を `q` とした短期の更新は次です。

```text
short_weight = min(1, short_weight(t) + short_reinforcement * q)
```

設定した長期方針が長期強化を許可する場合、長期の更新は次です。

```text
long_weight = min(1, long_weight(t) + long_reinforcement * q)
```

許可しない場合は、減衰後の長期重みをそのまま保ちます。短期・長期の familiarity は
どちらも更新前の unit 重みから計算し、その後に unit 重みと完全一致 pattern の
recency 記録を更新します。

既定の長期方針は `LongLearningPolicy::Revisit` です。完全一致 pattern を設定した
間隔の後に再訪したときだけ長期を強化します。連続した正常経験が確認済みの場合は
`LongLearningPolicy::RepeatedNormal` を選べます。新しい設定で互換性専用の、全 commit
で長期強化する方針は使えません。そのため、1秒周期の障害が反復しただけで長期の
正常状態になることはありません。

familiarity は疎な unit の重なりに基づきます。完全一致 pattern の `recency` は
ソート済みの unit 列全体で判定します。完全一致の記録には `Config::max_patterns`
の LRU 上限があり、比較キーは `(last_seen, observation_sequence, pattern)` です。
同じ timestamp の観測順も保存します。追い出されるのは recency 用の完全一致記録
だけで、unit の重みは残ります。unit 番号は `Config::encoder_units` の範囲内か検証
します。

Fly の診断情報には正規化した入力強度と、running feature profile からの偏差が
あります。profile 偏差は特徴量の基準からの距離であり、観測間隔の検出ではありません。

## 数値偏差 model

`NumericDeviationModel` は `latency_ms` や `retry_count` のような名前付き数値を
受け取ります。stream ごとに、時間窓内の重み付き標本を上限付きで保持します。
現在値を学習する前に、次を計算します。

```text
z+ = max(0, (x - median) / max(1.4826 * MAD, mad_floor))       # Up
z- = max(0, (median - x) / max(1.4826 * MAD, mad_floor))       # Down
z  = max(z+, z-)                                               # Both
```

`mad_floor`、方向、標本の時間窓、基準に必要な最小標本数は設定値です。最小標本数
に達するまでは readiness が `BaselineInsufficient` であり、familiarity 1 の正常
とは扱いません。診断情報には現在値、中央値、MAD、標本数、時間窓、方向が含まれます。
raw の z は未校正なので、共通反応の `strength` は `None` のままです。標本窓、時刻、
完全一致の recency、model 版、schema 版は保存・復元されます。

## 有限近傍の参照 model

`NearestNeighborModel` は stream ごとに最大 `max_representatives` 件の重み付き代表
ベクトルを保持します。`merge_distance` 以下のコサイン距離なら重複としてまとめ、
それ以外は重みが小さく古い代表を追い出します。最近傍コサイン距離、保持代表への
距離の平均・中央値・最大値、完全一致の recency を報告します。既定の merge 距離は
`0.03` です。実際の schema では距離の診断値を見て設定を校正してください。Fly の
疎な unit 一般化と、有限個の例を使う方式を比較するための単純な参照 model です。

## 構造化された失敗 model

`FailureModel` は、adapter がすでに持っている構造化処理結果を評価する学習なしの
model です。特徴ベクトルではなく `ModelInput::Failure` を受け、結果コード、エラー
種別、状態を根拠として返します。失敗結果には `Failure` 反応を返し、成功を正常な
pattern として学習しません。障害の反復で novelty が下がる Fly や数値 model とは
独立した失敗指標を保てます。

## 永続化と互換性

`Habituation::save`／`load`、`NumericDeviationModel::save`／`load`、
`NearestNeighborModel::save`／`load`、`BrainRunner::save`／`load` はバージョン付き
binary format を使います。model payload には model 版、feature schema 版、設定
fingerprint を含めます。runner の状態は脳 ID と設定識別値で分離されます。

脳を無効化しても model 状態は保持します。runner から削除した脳はメモリ上から削除
されます。model や encoder を変更する場合は新しい状態識別値を使います。
`promote_shadow` は互換性を確認した shadow から active へ明示的に状態を引き継ぐ
操作です。

model 版と adapter の feature schema 版は呼び出し側が指定します。schema 版は特徴の
意味、正規化、並び、単位を含む契約です。意味を変更する場合は互換性のある新しい
状態識別値を使います。構造化された失敗 schema も、ベクトル次元を持たなくても同じ
版の識別を使います。

保存前と読み込み中は同じ累積 `LoadBudget` を検証します。脳 ID、stream、profile・
標本、代表例、unit 状態、pattern key を対象にし、保存側と復元側の上限を一致させます。
保存に成功した状態が割り当て上限だけを理由に読み込みで失敗することはありません。
破損または切断された状態は拒否し、失敗時に部分復元状態を返しません。

## replay による比較

`examples/replay.rs` は複数の脳を同じ入力で比較します。

```text
cargo run -p habitua --example replay -- records.jsonl experiment.json labels.jsonl
```

record は ID、派生 record を元 record に結び付ける optional な `source_id`、元の
timestamp、入力が利用可能になった timestamp（省略可）、stream、target、feature
schema ID／版、features または構造化された失敗結果を持ちます。experiment の設定には脳の一覧、
seed、warmup と校正の期間、反応閾値、学習方針、資源上限を指定します。ラベルは別
ファイルの時間区間または record 注釈で、`failure`、`important_change`、
`confirmed_normal`、`undetermined` を指定します。

正解ラベルは採点専用です。ラベルで学習可否を決めません。確認結果を学習へ使う
場合は、その確認が得られた時刻以降の別の時刻付き入力として表します。replay は
入力が利用可能になった時刻で評価し、warmup、校正、評価を時系列で分割します。
各脳について、学習前の評価、根拠、readiness、学習内容、処理時間を出力し、事象別の
検出率、遅延、見逃し、正常な稼働1時間当たりおよび正常 record 1,000件当たりの誤通知、
脳を追加した効果を閾値ごとの表で集計します。ラベルには stream と target を指定でき、
省略時だけ実験全体を対象にします。`record_id` または `source_id` で一致させられます。
障害の検出は同じ stream／target で、ラベル開始時刻から設定した有限の検出窓内だけを
数え、`end_ms` があればそれでも制限します。遅延には最初の反応を使います。
`confirmed_normal` 区間は stream／target ごとに重複を除いて正常時間の分母にします。
正常区間がない場合は1時間などを仮定せず、正常時間と率を不明として出力します。

## core に単一の salience を置かない理由

以前の salience は novelty、入力強度、profile 偏差の積でした。これらは意味と
校正が異なり、pattern に慣れた後の数値異常を隠すことがあります。特徴表現や次元
数でも値が変わります。そのため `habitua` は model 固有の指標と反応を独立して返し、
普遍的な単一重要度を作りません。必要な用途では CooSenpAI またはアプリケーション
側が、検証データに基づく校正と起動方針を担当します。

## 例とベンチ

`structured_log` は標準入力から構造化特徴を読み、無効入力で panic せず Result の
エラーを返します。`random_vectors` は決定的な乱数ベクトルを流します。Criterion
ベンチは通常・ドリフト入力、多数 pattern で上限に達した状態、複数 stream、Numeric、
4,096次元・2,048 unit の Fly を測定します。50マイクロ秒は通常の128次元に対する
開発目安であり、全 model と全状態量への保証ではありません。

## CooSenpAI からの利用予定

CooSenpAI の adapter は画像認識、OCR、音声認識、テキスト embedding をこの crate
の外で行います。既存の変化 gate の前に入力を分岐し、安定した文脈キーと schema 版を
付け、画面配置、文字、音、イベント、内部ログ、名前付き数値を別の脳へ渡します。
CooSenpAI は反応を受けて観察、会話、運用通知を始めるかを決めます。`habitua` は
独立した報告、根拠、状態互換性、evaluate／commit の順序を提供します。

## ライセンス

次のいずれかのライセンスで利用できます。

* Apache License, Version 2.0（[LICENSE-APACHE](LICENSE-APACHE)）
* MIT License（[LICENSE-MIT](LICENSE-MIT)）
