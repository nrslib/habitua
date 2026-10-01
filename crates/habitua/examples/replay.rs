use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::time::Duration;

use habitua::{
    BrainConfig, BrainInput, BrainInputSpec, BrainLearningPolicy, BrainMode, BrainRunner, Config,
    DeletionPolicy, Evidence, FailureObservation, FrequencyLimit, LearningMode, Metric,
    MetricDirection, ModelSettings, ModelSpec, NearestNeighborConfig, NumericDeviationConfig,
    NumericDirection, Reaction, ReactionKind, ReactionPolicy, Readiness, ResourcePolicy,
    StatePolicy, Timestamp,
};
use serde::{Deserialize, Serialize};

const DEFAULT_DETECTION_WINDOW_MINUTES: u64 = 5;
const NORMAL_LABEL: &str = "confirmed_normal";
const UNDETERMINED_LABEL: &str = "undetermined";
const EVENT_LABELS: [&str; 2] = ["failure", "important_change"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    #[serde(alias = "record_id")]
    id: String,
    #[serde(alias = "t_ms")]
    timestamp_ms: u64,
    #[serde(default)]
    available_at_ms: Option<u64>,
    stream: String,
    #[serde(default = "default_target", alias = "brain_input")]
    target: String,
    #[serde(default = "default_context")]
    context: String,
    #[serde(default)]
    schema_id: Option<String>,
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    feature_schema_version: Option<u32>,
    #[serde(default)]
    source_id: Option<String>,
    #[serde(default)]
    features: Vec<f32>,
    #[serde(default)]
    failure: Option<FileFailure>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileFailure {
    success: bool,
    #[serde(default)]
    result_code: Option<String>,
    #[serde(default)]
    error_type: Option<String>,
    #[serde(default)]
    state: Option<String>,
}

impl From<&FileFailure> for FailureObservation {
    fn from(value: &FileFailure) -> Self {
        Self {
            success: value.success,
            result_code: value.result_code.clone(),
            error_type: value.error_type.clone(),
            state: value.state.clone(),
        }
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Experiment {
    #[serde(default)]
    feature_schema_id: Option<String>,
    #[serde(default)]
    feature_schema_version: Option<u32>,
    #[serde(default)]
    warmup_until_ms: u64,
    #[serde(default)]
    calibration_until_ms: u64,
    #[serde(default)]
    learn_during_evaluation: bool,
    #[serde(default)]
    normal_operation_hours: Option<f64>,
    #[serde(default = "default_detection_window_minutes")]
    detection_window_minutes: u64,
    #[serde(default)]
    thresholds: Vec<f32>,
    #[serde(default)]
    brains: Vec<FileBrain>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileBrain {
    id: String,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    mode: Option<String>,
    target: String,
    #[serde(default = "default_context")]
    context: String,
    #[serde(default)]
    schema_id: Option<String>,
    #[serde(default)]
    schema_version: Option<u32>,
    #[serde(default = "default_model_version")]
    model_version: u32,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    threshold: Option<f32>,
    #[serde(default)]
    reaction_kinds: Option<Vec<String>>,
    #[serde(default)]
    score_metric: Option<String>,
    #[serde(default)]
    score_unit: Option<String>,
    #[serde(default)]
    score_direction: Option<String>,
    #[serde(default)]
    score_reaction_kind: Option<String>,
    #[serde(default)]
    score_label_kinds: Option<Vec<String>>,
    #[serde(default)]
    learning: FileLearning,
    #[serde(default)]
    resources: FileResources,
    #[serde(default)]
    state_identity: Option<String>,
    model: FileModel,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct FileLearning {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    weight: Option<f32>,
    #[serde(default)]
    interval_secs: Option<u64>,
    #[serde(default)]
    consecutive_observations: Option<u32>,
    #[serde(default)]
    require_evaluated: bool,
    #[serde(default)]
    bootstrap: Option<bool>,
    #[serde(default)]
    learn_on_reaction: Option<bool>,
    #[serde(default)]
    max_commits: Option<u32>,
    #[serde(default)]
    frequency_window_secs: Option<u64>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct FileResources {
    #[serde(default)]
    max_streams: Option<usize>,
    #[serde(default)]
    max_samples: Option<usize>,
    #[serde(default)]
    max_memory_bytes: Option<usize>,
    #[serde(default)]
    evaluation_deadline_ms: Option<u64>,
}

#[derive(Clone, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum FileLongLearning {
    Revisit { interval_secs: u64 },
    RepeatedNormal { consecutive_observations: u32 },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum FileModel {
    Fly {
        #[serde(default)]
        dimensions: Option<usize>,
        #[serde(default)]
        encoder_units: Option<usize>,
        #[serde(default)]
        k: Option<usize>,
        #[serde(default)]
        projection_density: Option<f32>,
        #[serde(default)]
        short_time_constant_secs: Option<u64>,
        #[serde(default)]
        long_time_constant_secs: Option<u64>,
        #[serde(default)]
        max_patterns: Option<usize>,
        #[serde(default)]
        short_reinforcement: Option<f32>,
        #[serde(default)]
        long_reinforcement: Option<f32>,
        #[serde(default)]
        short_familiarity_weight: Option<f32>,
        #[serde(default)]
        long_familiarity_weight: Option<f32>,
        #[serde(default)]
        input_strength_scale: Option<f32>,
        #[serde(default)]
        temporal_shift_scale: Option<f32>,
        #[serde(default)]
        variance_floor: Option<f32>,
        #[serde(default)]
        max_memory_elements: Option<usize>,
        #[serde(default)]
        long_learning: Option<FileLongLearning>,
    },
    Numeric {
        names: Vec<String>,
        #[serde(default)]
        mad_floor: Option<f32>,
        #[serde(default)]
        learning_rate: Option<f32>,
        #[serde(default)]
        max_samples: Option<usize>,
        #[serde(default)]
        sample_window_secs: Option<u64>,
        #[serde(default)]
        minimum_samples: Option<usize>,
        #[serde(default)]
        direction: Option<String>,
        #[serde(default)]
        max_patterns: Option<usize>,
    },
    NearestNeighbor {
        dimensions: usize,
        #[serde(default)]
        max_representatives: Option<usize>,
        #[serde(default)]
        merge_distance: Option<f32>,
        #[serde(default)]
        max_patterns: Option<usize>,
    },
    Failure,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Label {
    #[serde(default)]
    record_id: Option<String>,
    #[serde(default)]
    source_id: Option<String>,
    #[serde(default)]
    event_id: Option<String>,
    kind: String,
    start_ms: u64,
    #[serde(default)]
    end_ms: Option<u64>,
    #[serde(default)]
    stream: Option<String>,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    schema_id: Option<String>,
    #[serde(default)]
    schema_version: Option<u32>,
}

#[derive(Clone)]
struct RecordMeta {
    id: String,
    source_id: Option<String>,
    stream: String,
    target: String,
    context: String,
    schema_id: String,
    schema_version: u32,
    at_ms: u64,
}

#[derive(Serialize)]
struct RecordOutput {
    id: String,
    source_id: Option<String>,
    timestamp_ms: u64,
    available_at_ms: u64,
    stream: String,
    target: String,
    context: String,
    schema_id: String,
    schema_version: u32,
    phase: &'static str,
    brains: Vec<BrainOutput>,
}

#[derive(Serialize)]
struct BrainOutput {
    brain: String,
    role: String,
    mode: String,
    model_kind: String,
    model_version: u32,
    schema_id: String,
    schema_version: u32,
    readiness: String,
    learning_signal: String,
    novelty_short: Option<f32>,
    novelty_long: Option<f32>,
    familiarity_short: Option<f32>,
    familiarity_long: Option<f32>,
    recency_ms: Option<u128>,
    pattern_id: Option<u64>,
    score_metric: String,
    score_unit: String,
    score_direction: String,
    score_reaction_kind: String,
    score_label_kinds: Vec<String>,
    live_threshold: f32,
    configuration_fingerprint: u64,
    reactions: Vec<ReactionOutput>,
    metrics: Vec<MetricOutput>,
    evidence: Vec<EvidenceOutput>,
    learning_weight: f32,
    learned: bool,
    processing_time_us: u128,
}

#[derive(Serialize)]
struct ReactionOutput {
    brain: String,
    target: String,
    kind: String,
    strength: Option<f32>,
    readiness: String,
    metrics: Vec<MetricOutput>,
    evidence: Vec<EvidenceOutput>,
}

#[derive(Serialize)]
struct MetricOutput {
    name: String,
    value: f32,
    unit: String,
    direction: String,
}

#[derive(Serialize)]
struct EvidenceOutput {
    name: String,
    current: Option<f32>,
    baseline: Option<f32>,
    sample_count: Option<usize>,
    window_ms: Option<u128>,
    reference_ids: Vec<String>,
    detail: String,
}

#[derive(Serialize)]
struct ThresholdSummary {
    threshold: f32,
    brains: Vec<BrainSummary>,
}

#[derive(Serialize)]
struct BrainSummary {
    brain: String,
    model_kind: String,
    model_version: u32,
    configuration_fingerprint: u64,
    schema_id: String,
    schema_version: u32,
    score_metric: String,
    score_unit: String,
    score_direction: String,
    score_reaction_kind: String,
    score_label_kinds: Vec<String>,
    replay_threshold: f32,
    live_threshold: f32,
    normal_duration_hours: Option<f64>,
    normal_record_count: usize,
    selected_event_count: usize,
    selected_evaluable_event_count: usize,
    selected_recall: Option<f64>,
    selected_evaluable_recall: Option<f64>,
    selected_undetermined: usize,
    selected_missed: usize,
    failure_recall: Option<f64>,
    important_change_recall: Option<f64>,
    failure_mean_detection_delay_ms: Option<f64>,
    important_change_mean_detection_delay_ms: Option<f64>,
    failure_undetermined: usize,
    important_change_undetermined: usize,
    failure_missed: usize,
    important_change_missed: usize,
    false_notifications_per_normal_hour: Option<f64>,
    false_notifications_per_1000_normal_records: Option<f64>,
    false_notification_count: usize,
    confirmed_normal_duration_hours: Option<f64>,
    confirmed_normal_record_count: usize,
    false_notifications_per_confirmed_normal_hour: Option<f64>,
    false_notifications_per_1000_confirmed_normal_records: Option<f64>,
    false_notification_count_after_warmup: usize,
    additional_detections_over_previous: usize,
}

#[derive(Serialize)]
struct Summary {
    detection_window_minutes: u64,
    normal_duration_ms: Option<u128>,
    normal_record_count: usize,
    confirmed_normal_duration_ms: Option<u128>,
    confirmed_normal_record_count: usize,
    normal_operation_hours_hint: Option<f64>,
    thresholds: Vec<ThresholdSummary>,
}

#[derive(Default, Clone)]
struct EventResult {
    detected_at_ms: Option<u64>,
    candidate_count: usize,
    evaluable_count: usize,
}

#[derive(Clone, Debug)]
struct EventInstance {
    key: String,
    kind: String,
    start_ms: u64,
}

struct ThresholdState {
    threshold: f32,
    events: BTreeMap<String, BTreeMap<String, EventResult>>,
    false_notifications: BTreeMap<String, usize>,
    false_notifications_after_warmup: BTreeMap<String, usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScoreDirection {
    Higher,
    Lower,
    TwoSided,
}

impl ScoreDirection {
    fn as_str(self) -> &'static str {
        match self {
            Self::Higher => "higher_is_more",
            Self::Lower => "lower_is_more",
            Self::TwoSided => "two_sided",
        }
    }
}

#[derive(Clone, Debug)]
struct ScoreSelector {
    metric: String,
    unit: String,
    direction: ScoreDirection,
    reaction_kind: ReactionKind,
    label_kinds: Vec<String>,
}

impl ThresholdState {
    fn new(threshold: f32) -> Self {
        Self {
            threshold,
            events: BTreeMap::new(),
            false_notifications: BTreeMap::new(),
            false_notifications_after_warmup: BTreeMap::new(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().collect::<Vec<_>>();
    if arguments.len() != 4 {
        return Err(
            "usage: replay <records.jsonl> <experiment.json> <labels.jsonl>"
                .to_owned()
                .into(),
        );
    }

    let experiment: Experiment = serde_json::from_reader(File::open(&arguments[2])?)?;
    if experiment.brains.is_empty() {
        return Err("experiment must define at least one brain"
            .to_owned()
            .into());
    }
    if experiment.detection_window_minutes == 0 {
        return Err("detection_window_minutes must be greater than zero"
            .to_owned()
            .into());
    }
    if experiment
        .normal_operation_hours
        .is_some_and(|hours| !hours.is_finite() || hours < 0.0)
    {
        return Err("normal_operation_hours must be finite and non-negative"
            .to_owned()
            .into());
    }
    let mut records = read_records(&arguments[1])?;
    for record in &mut records {
        normalize_record(record, &experiment)?;
    }
    records.sort_by_key(|record| record.available_at_ms.unwrap_or(record.timestamp_ms));
    let labels = read_labels(&arguments[3])?;
    let configs = experiment
        .brains
        .iter()
        .map(|brain| make_brain_config(brain, &experiment))
        .collect::<Result<Vec<_>, _>>()?;
    let selectors = configs
        .iter()
        .zip(&experiment.brains)
        .map(|(config, brain)| {
            Ok::<_, Box<dyn std::error::Error>>((
                config.id.clone(),
                make_score_selector(brain, config)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let thresholds = choose_thresholds(&experiment, &configs)?;
    let mut threshold_states = thresholds
        .iter()
        .copied()
        .map(ThresholdState::new)
        .collect::<Vec<_>>();
    let registry = habitua::ModelRegistry::with_builtins();
    let mut runner = BrainRunner::new(configs.clone(), &registry)?;
    let window_ms = experiment
        .detection_window_minutes
        .checked_mul(60_000)
        .ok_or("detection window is too large")?;
    let mut all_records = Vec::with_capacity(records.len());
    let mut evaluation_records = Vec::new();
    let evaluation_bounds = records
        .iter()
        .filter(|record| {
            phase_for(
                record.available_at_ms.unwrap_or(record.timestamp_ms),
                experiment.warmup_until_ms,
                experiment.calibration_until_ms,
            ) == "evaluation"
        })
        .map(|record| record.available_at_ms.unwrap_or(record.timestamp_ms))
        .min()
        .zip(
            records
                .iter()
                .filter(|record| {
                    phase_for(
                        record.available_at_ms.unwrap_or(record.timestamp_ms),
                        experiment.warmup_until_ms,
                        experiment.calibration_until_ms,
                    ) == "evaluation"
                })
                .map(|record| record.available_at_ms.unwrap_or(record.timestamp_ms))
                .max(),
        );

    for (position, record) in records.into_iter().enumerate() {
        let Record {
            id,
            timestamp_ms,
            available_at_ms,
            stream,
            target,
            context,
            schema_id,
            feature_schema_version,
            source_id,
            features,
            failure,
            ..
        } = record;
        let schema_id = schema_id.ok_or("record schema_id was not normalized")?;
        let feature_schema_version =
            feature_schema_version.ok_or("record schema version was not normalized")?;
        let available_at_ms = available_at_ms.unwrap_or(timestamp_ms);
        let phase = phase_for(
            available_at_ms,
            experiment.warmup_until_ms,
            experiment.calibration_until_ms,
        );
        let meta = RecordMeta {
            id: id.clone(),
            source_id: source_id.clone(),
            stream: stream.clone(),
            target: target.clone(),
            context: context.clone(),
            schema_id: schema_id.clone(),
            schema_version: feature_schema_version,
            at_ms: available_at_ms,
        };
        all_records.push(meta.clone());
        if phase == "evaluation" {
            evaluation_records.push(meta.clone());
        }
        let input = BrainInput {
            id: id.clone(),
            stream_id: stream,
            target: target.clone(),
            context: context.clone(),
            schema_id,
            schema_version: feature_schema_version,
            features,
            failure: failure.as_ref().map(FailureObservation::from),
            available_at: Timestamp::from_millis(available_at_ms),
            processed_at: Some(Timestamp::from_millis(available_at_ms)),
            position: position as u64,
        };
        let evaluation = runner.evaluate_all(&input)?;
        let mut output_brains = Vec::with_capacity(evaluation.brains.len());
        for brain in evaluation.brains {
            let brain_id = brain.brain.clone();
            let config = runner
                .brain_config(&brain_id)
                .cloned()
                .ok_or_else(|| format!("brain {brain_id} disappeared from runner"))?;
            let selector = selectors
                .get(&brain_id)
                .ok_or_else(|| format!("score selector for brain {brain_id} is missing"))?;
            let positive_by_threshold = threshold_states
                .iter()
                .map(|state| {
                    (
                        state.threshold,
                        report_is_positive(&brain.report, state.threshold, selector),
                    )
                })
                .collect::<Vec<_>>();
            if phase == "evaluation" {
                for (state, (_, positive)) in threshold_states
                    .iter_mut()
                    .zip(positive_by_threshold.iter())
                {
                    let is_normal = brain_record_matches(&meta, &config)
                        && labels
                            .iter()
                            .any(|label| normal_label_matches(label, &meta));
                    record_threshold_result(ThresholdObservation {
                        state,
                        brain_id: &brain_id,
                        positive: *positive,
                        evaluable: positive.is_some(),
                        selector,
                        labels: &labels,
                        record: &meta,
                        window_ms,
                        is_normal,
                    });
                }
            }
            if phase != "warmup" {
                let is_normal = brain_record_matches(&meta, &config)
                    && labels
                        .iter()
                        .any(|label| normal_label_matches(label, &meta));
                for (state, (_, positive)) in threshold_states
                    .iter_mut()
                    .zip(positive_by_threshold.iter())
                {
                    record_false_notification(state, &brain_id, *positive, is_normal);
                }
            }
            let weight = if phase == "evaluation" && !experiment.learn_during_evaluation {
                0.0
            } else {
                brain.recommended_learning_weight
            };
            let report = brain.report;
            let learned = if let Some(update) = brain.update {
                let learned = weight > 0.0;
                runner.commit(update, weight)?;
                learned
            } else {
                false
            };
            output_brains.push(BrainOutput {
                brain: brain_id,
                role: config.role.clone(),
                mode: format!("{:?}", config.mode),
                model_kind: config.model.kind.clone(),
                model_version: config.model.version,
                schema_id: config.input.schema_id.clone(),
                schema_version: config.input.schema_version,
                readiness: format!("{:?}", report.readiness),
                learning_signal: format!("{:?}", report.learning_signal),
                novelty_short: report.novelty_short,
                novelty_long: report.novelty_long,
                familiarity_short: report.familiarity_short,
                familiarity_long: report.familiarity_long,
                recency_ms: report.recency.map(|value| value.as_millis()),
                pattern_id: report.pattern_id,
                score_metric: selector.metric.clone(),
                score_unit: selector.unit.clone(),
                score_direction: selector.direction.as_str().to_owned(),
                score_reaction_kind: format!("{:?}", selector.reaction_kind),
                score_label_kinds: selector.label_kinds.clone(),
                live_threshold: config.reactions.threshold,
                configuration_fingerprint: config.configuration_fingerprint(),
                reactions: report.reactions.into_iter().map(reaction_output).collect(),
                metrics: report.metrics.into_iter().map(metric_output).collect(),
                evidence: report.evidence.into_iter().map(evidence_output).collect(),
                learning_weight: weight,
                learned,
                processing_time_us: brain.processing_time.as_micros(),
            });
        }
        println!(
            "{}",
            serde_json::to_string(&RecordOutput {
                id,
                source_id,
                timestamp_ms,
                available_at_ms,
                stream: meta.stream,
                target,
                context,
                schema_id: input.schema_id,
                schema_version: input.schema_version,
                phase,
                brains: output_brains,
            })?
        );
    }

    let evaluation_normal_duration_ms = normal_duration_ms(&labels, &evaluation_records);
    let confirmed_normal_duration_ms = normal_duration_ms(&labels, &all_records);
    let summary = build_summary(SummaryInput {
        states: threshold_states,
        configs: &configs,
        selectors: &selectors,
        labels: &labels,
        evaluation_bounds,
        normal_duration_ms: evaluation_normal_duration_ms,
        evaluation_records: &evaluation_records,
        all_records: &all_records,
        detection_window_minutes: experiment.detection_window_minutes,
        detection_window_ms: window_ms,
        confirmed_normal_duration_ms,
        normal_operation_hours_hint: experiment.normal_operation_hours,
    });
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

fn make_brain_config(
    brain: &FileBrain,
    experiment: &Experiment,
) -> Result<BrainConfig, Box<dyn std::error::Error>> {
    let default_schema_id = experiment
        .feature_schema_id
        .clone()
        .unwrap_or_else(default_schema_id);
    let schema_id = brain.schema_id.clone().unwrap_or(default_schema_id);
    let schema_version = brain
        .schema_version
        .or(experiment.feature_schema_version)
        .unwrap_or_else(default_schema_version);
    let resources = ResourcePolicy {
        max_streams: brain.resources.max_streams.unwrap_or(4_096),
        max_samples: brain.resources.max_samples.unwrap_or(4_096),
        max_memory_bytes: brain.resources.max_memory_bytes.unwrap_or(4_000_000),
        evaluation_deadline: brain
            .resources
            .evaluation_deadline_ms
            .map(Duration::from_millis),
    };
    let (kind, settings, is_failure) = match &brain.model {
        FileModel::Fly {
            dimensions,
            encoder_units,
            k,
            projection_density,
            short_time_constant_secs,
            long_time_constant_secs,
            max_patterns,
            short_reinforcement,
            long_reinforcement,
            short_familiarity_weight,
            long_familiarity_weight,
            input_strength_scale,
            temporal_shift_scale,
            variance_floor,
            max_memory_elements,
            long_learning,
        } => {
            let defaults = Config::default();
            let long_learning_policy = long_learning
                .as_ref()
                .map(|policy| match policy {
                    FileLongLearning::Revisit { interval_secs } => {
                        habitua::LongLearningPolicy::Revisit {
                            interval: Duration::from_secs(*interval_secs),
                        }
                    }
                    FileLongLearning::RepeatedNormal {
                        consecutive_observations,
                    } => habitua::LongLearningPolicy::RepeatedNormal {
                        consecutive_observations: *consecutive_observations,
                    },
                })
                .unwrap_or_else(|| {
                    if matches!(brain.learning.mode.as_deref(), Some("repeated_normal")) {
                        habitua::LongLearningPolicy::RepeatedNormal {
                            consecutive_observations: brain
                                .learning
                                .consecutive_observations
                                .unwrap_or(2),
                        }
                    } else {
                        defaults.long_learning_policy.clone()
                    }
                });
            let config = Config {
                dimensions: dimensions.unwrap_or(defaults.dimensions),
                encoder_units: encoder_units.unwrap_or(defaults.encoder_units),
                k: k.unwrap_or(defaults.k),
                seed: brain.seed.unwrap_or(defaults.seed),
                projection_density: projection_density.unwrap_or(defaults.projection_density),
                short_time_constant: Duration::from_secs(
                    short_time_constant_secs.unwrap_or(defaults.short_time_constant.as_secs()),
                ),
                long_time_constant: Duration::from_secs(
                    long_time_constant_secs.unwrap_or(defaults.long_time_constant.as_secs()),
                ),
                max_patterns: max_patterns.unwrap_or(defaults.max_patterns),
                short_reinforcement: short_reinforcement.unwrap_or(defaults.short_reinforcement),
                long_reinforcement: long_reinforcement.unwrap_or(defaults.long_reinforcement),
                short_familiarity_weight: short_familiarity_weight
                    .unwrap_or(defaults.short_familiarity_weight),
                long_familiarity_weight: long_familiarity_weight
                    .unwrap_or(defaults.long_familiarity_weight),
                input_strength_scale: input_strength_scale.unwrap_or(defaults.input_strength_scale),
                temporal_shift_scale: temporal_shift_scale.unwrap_or(defaults.temporal_shift_scale),
                variance_floor: variance_floor.unwrap_or(defaults.variance_floor),
                long_learning_policy,
                max_streams: resources.max_streams,
                max_memory_elements: max_memory_elements.unwrap_or(defaults.max_memory_elements),
                model_version: brain.model_version,
                feature_schema_version: schema_version,
            };
            config.validate()?;
            ("fly", ModelSettings::Fly(config), false)
        }
        FileModel::Numeric {
            names,
            mad_floor,
            learning_rate,
            max_samples,
            sample_window_secs,
            minimum_samples,
            direction,
            max_patterns,
        } => {
            let defaults = NumericDeviationConfig::default();
            let direction = match direction.as_deref().unwrap_or("both") {
                "up" | "upper" => NumericDirection::Up,
                "down" | "lower" => NumericDirection::Down,
                "both" => NumericDirection::Both,
                other => return Err(format!("unknown numeric direction {other}").into()),
            };
            let config = NumericDeviationConfig {
                mad_floor: mad_floor.unwrap_or(defaults.mad_floor),
                learning_rate: learning_rate.unwrap_or(defaults.learning_rate),
                max_samples: max_samples.unwrap_or(defaults.max_samples),
                max_streams: resources.max_streams,
                sample_window: Duration::from_secs(
                    sample_window_secs.unwrap_or(defaults.sample_window.as_secs()),
                ),
                minimum_samples: minimum_samples.unwrap_or(defaults.minimum_samples),
                direction,
                max_patterns: max_patterns.unwrap_or(defaults.max_patterns),
                model_version: brain.model_version,
                feature_schema_version: schema_version,
            };
            (
                "numeric",
                ModelSettings::Numeric {
                    names: names.clone(),
                    config,
                },
                false,
            )
        }
        FileModel::NearestNeighbor {
            dimensions,
            max_representatives,
            merge_distance,
            max_patterns,
        } => {
            let defaults = NearestNeighborConfig::default();
            let config = NearestNeighborConfig {
                dimensions: *dimensions,
                max_representatives: max_representatives.unwrap_or(defaults.max_representatives),
                merge_distance: merge_distance.unwrap_or(defaults.merge_distance),
                max_patterns: max_patterns.unwrap_or(defaults.max_patterns),
                max_streams: resources.max_streams,
                model_version: brain.model_version,
                feature_schema_version: schema_version,
            };
            (
                "nearest_neighbor",
                ModelSettings::NearestNeighbor(config),
                false,
            )
        }
        FileModel::Failure => (
            "failure",
            ModelSettings::Failure(habitua::FailureModelConfig {
                model_version: brain.model_version,
                feature_schema_version: schema_version,
            }),
            true,
        ),
    };
    let mode = match brain.mode.as_deref().unwrap_or("active") {
        "active" => BrainMode::Active,
        "shadow" => BrainMode::Shadow,
        other => return Err(format!("unknown brain mode {other}").into()),
    };
    let learning_mode = if is_failure {
        LearningMode::Manual
    } else {
        match brain.learning.mode.as_deref().unwrap_or("every_evaluation") {
            "every_evaluation" => LearningMode::EveryEvaluation,
            "repeated_normal" => LearningMode::RepeatedNormal {
                consecutive_observations: brain.learning.consecutive_observations.unwrap_or(2),
            },
            "revisit" => LearningMode::Revisit {
                interval: Duration::from_secs(brain.learning.interval_secs.unwrap_or(3_600)),
            },
            "manual" => LearningMode::Manual,
            other => return Err(format!("unknown learning mode {other}").into()),
        }
    };
    let frequency_limit = brain
        .learning
        .max_commits
        .map(|maximum_commits| FrequencyLimit {
            maximum_commits,
            window: Duration::from_secs(brain.learning.frequency_window_secs.unwrap_or(3_600)),
        });
    let reaction_kinds = brain
        .reaction_kinds
        .as_ref()
        .map(|kinds| {
            kinds
                .iter()
                .map(|kind| parse_reaction_kind(kind))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let default_reactions = if is_failure {
        vec![ReactionKind::Failure]
    } else if kind == "numeric" {
        vec![ReactionKind::Deviation]
    } else {
        vec![ReactionKind::Novel]
    };
    let bootstrap = brain
        .learning
        .bootstrap
        .unwrap_or(matches!(learning_mode, LearningMode::RepeatedNormal { .. }));
    Ok(BrainConfig {
        id: brain.id.clone(),
        role: brain.role.clone().unwrap_or_else(|| brain.id.clone()),
        enabled: brain.enabled.unwrap_or(true),
        mode,
        input: BrainInputSpec {
            schema_id,
            schema_version,
            target: brain.target.clone(),
            context: brain.context.clone(),
        },
        model: ModelSpec {
            kind: kind.to_owned(),
            version: brain.model_version,
            seed: brain.seed,
            settings,
        },
        learning: BrainLearningPolicy {
            mode: learning_mode,
            weight: if is_failure {
                0.0
            } else {
                brain.learning.weight.unwrap_or(1.0)
            },
            require_evaluated: brain.learning.require_evaluated,
            bootstrap,
            learn_on_reaction: brain.learning.learn_on_reaction.unwrap_or(false),
            frequency_limit,
        },
        reactions: ReactionPolicy {
            threshold: brain.threshold.unwrap_or(0.8),
            kinds: reaction_kinds.unwrap_or(default_reactions),
            calibration: None,
        },
        resources,
        state: StatePolicy {
            format_version: 0,
            configuration_identity: brain
                .state_identity
                .clone()
                .unwrap_or_else(|| format!("{}:{}", brain.id, brain.model_version)),
            deletion: DeletionPolicy::Delete,
        },
    })
}

fn choose_thresholds(
    experiment: &Experiment,
    configs: &[BrainConfig],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let values = if experiment.thresholds.is_empty() {
        configs
            .iter()
            .flat_map(|config| {
                if config.model.kind == "numeric" {
                    vec![3.0, 10.0, 30.0]
                } else {
                    vec![0.5, 0.6, 0.7]
                }
            })
            .collect::<Vec<_>>()
    } else {
        experiment.thresholds.clone()
    };
    let mut thresholds = Vec::new();
    for value in values {
        if !value.is_finite() || value < 0.0 {
            return Err("thresholds must be finite and non-negative"
                .to_owned()
                .into());
        }
        if !thresholds
            .iter()
            .any(|existing: &f32| existing.to_bits() == value.to_bits())
        {
            thresholds.push(value);
        }
    }
    thresholds.sort_by(f32::total_cmp);
    if thresholds.is_empty() {
        thresholds.push(0.8);
    }
    Ok(thresholds)
}

fn make_score_selector(
    brain: &FileBrain,
    config: &BrainConfig,
) -> Result<ScoreSelector, Box<dyn std::error::Error>> {
    let (default_metric, default_unit, default_direction, default_kind) =
        match config.model.kind.as_str() {
            "fly" => (
                "novelty_short",
                "ratio",
                ScoreDirection::Higher,
                ReactionKind::Novel,
            ),
            "numeric" => (
                "z_plus",
                "z",
                ScoreDirection::Higher,
                ReactionKind::Deviation,
            ),
            "nearest_neighbor" => (
                "cosine_distance",
                "distance",
                ScoreDirection::Higher,
                ReactionKind::Novel,
            ),
            "failure" => (
                "failed",
                "boolean",
                ScoreDirection::Higher,
                ReactionKind::Failure,
            ),
            other => return Err(format!("no replay score selector for model {other}").into()),
        };
    let direction = match brain.score_direction.as_deref() {
        None | Some("higher") | Some("higher_is_more") => ScoreDirection::Higher,
        Some("lower") | Some("lower_is_more") => ScoreDirection::Lower,
        Some("two_sided") => ScoreDirection::TwoSided,
        Some(other) => return Err(format!("unknown score direction {other}").into()),
    };
    let reaction_kind = brain
        .score_reaction_kind
        .as_deref()
        .map(parse_reaction_kind)
        .transpose()?
        .unwrap_or(default_kind);
    if !config
        .reactions
        .kinds
        .iter()
        .any(|kind| kind == &reaction_kind)
    {
        return Err(format!(
            "score reaction kind {:?} is not enabled for brain {}",
            reaction_kind, config.id
        )
        .into());
    }
    let label_kinds = brain
        .score_label_kinds
        .clone()
        .unwrap_or_else(|| default_label_kinds(&reaction_kind));
    let selector = ScoreSelector {
        metric: brain
            .score_metric
            .clone()
            .unwrap_or_else(|| default_metric.to_owned()),
        unit: brain
            .score_unit
            .clone()
            .unwrap_or_else(|| default_unit.to_owned()),
        direction: if brain.score_direction.is_some() {
            direction
        } else {
            default_direction
        },
        reaction_kind,
        label_kinds,
    };
    if selector.metric.is_empty() || selector.unit.is_empty() {
        return Err(format!("score selector for brain {} is empty", config.id).into());
    }
    let mut label_kinds_seen = BTreeSet::new();
    if selector.label_kinds.is_empty()
        || selector
            .label_kinds
            .iter()
            .any(|kind| !is_event_label(kind) || !label_kinds_seen.insert(kind))
    {
        return Err(format!(
            "score label kinds for brain {} must contain failure or important_change",
            config.id
        )
        .into());
    }
    validate_builtin_score_selector(config, &selector)?;
    Ok(selector)
}

fn validate_builtin_score_selector(
    config: &BrainConfig,
    selector: &ScoreSelector,
) -> Result<(), Box<dyn std::error::Error>> {
    let (expected_metric, expected_unit, expected_kind) = match &config.model.settings {
        ModelSettings::Fly(_) => ("novelty_short", "ratio", ReactionKind::Novel),
        ModelSettings::Numeric { names, .. } => {
            let metric_is_valid = selector.metric == "z_plus"
                || names
                    .iter()
                    .any(|name| selector.metric == format!("{name}.z_plus"));
            if !metric_is_valid {
                return Err(format!(
                    "numeric replay score metric {} is not a live z_plus metric for brain {}",
                    selector.metric, config.id
                )
                .into());
            }
            ("z_plus", "z", ReactionKind::Deviation)
        }
        ModelSettings::NearestNeighbor(_) => ("cosine_distance", "distance", ReactionKind::Novel),
        ModelSettings::Failure(_) => ("failed", "boolean", ReactionKind::Failure),
        ModelSettings::Custom(_) => {
            return Err(format!(
                "custom model {} must provide an explicit replay selector contract",
                config.id
            )
            .into());
        }
    };
    if !matches!(&config.model.settings, ModelSettings::Numeric { .. })
        && selector.metric != expected_metric
    {
        return Err(format!(
            "replay score metric {} does not match live metric {} for brain {}",
            selector.metric, expected_metric, config.id
        )
        .into());
    }
    if selector.unit != expected_unit
        || selector.direction != ScoreDirection::Higher
        || selector.reaction_kind != expected_kind
    {
        return Err(format!(
            "replay selector for brain {} does not match the live model report",
            config.id
        )
        .into());
    }
    let allowed_label_kinds = default_label_kinds(&expected_kind);
    if selector
        .label_kinds
        .iter()
        .any(|kind| !allowed_label_kinds.iter().any(|allowed| allowed == kind))
    {
        return Err(format!(
            "replay label kinds do not match reaction {:?} for brain {}",
            expected_kind, config.id
        )
        .into());
    }
    Ok(())
}

fn default_label_kinds(reaction_kind: &ReactionKind) -> Vec<String> {
    match reaction_kind {
        ReactionKind::Failure => vec!["failure".to_owned()],
        ReactionKind::Deviation
        | ReactionKind::Novel
        | ReactionKind::Absence
        | ReactionKind::Repeat => EVENT_LABELS.iter().map(|kind| (*kind).to_owned()).collect(),
    }
}

fn metric_name_matches(actual: &str, selected: &str) -> bool {
    actual == selected
        || actual
            .strip_suffix(selected)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

fn metric_direction_matches(actual: MetricDirection, selected: ScoreDirection) -> bool {
    matches!(
        (actual, selected),
        (MetricDirection::HigherIsMore, ScoreDirection::Higher)
            | (MetricDirection::LowerIsMore, ScoreDirection::Lower)
            | (MetricDirection::TwoSided, ScoreDirection::TwoSided)
    )
}

struct ThresholdObservation<'a> {
    state: &'a mut ThresholdState,
    brain_id: &'a str,
    positive: Option<bool>,
    evaluable: bool,
    selector: &'a ScoreSelector,
    labels: &'a [Label],
    record: &'a RecordMeta,
    window_ms: u64,
    is_normal: bool,
}

fn record_threshold_result(observation: ThresholdObservation<'_>) {
    let ThresholdObservation {
        state,
        brain_id,
        positive,
        evaluable,
        selector,
        labels,
        record,
        window_ms,
        is_normal,
    } = observation;
    if positive == Some(true) && is_normal {
        let count = state
            .false_notifications
            .entry(brain_id.to_owned())
            .or_default();
        *count = count.saturating_add(1);
    }
    let events = state.events.entry(brain_id.to_owned()).or_default();
    for label in labels
        .iter()
        .filter(|label| selector.label_kinds.iter().any(|kind| kind == &label.kind))
    {
        if event_label_matches(label, record, window_ms) {
            let result = events.entry(event_key(label)).or_default();
            result.candidate_count = result.candidate_count.saturating_add(1);
            if evaluable {
                result.evaluable_count = result.evaluable_count.saturating_add(1);
            }
            if positive == Some(true) {
                result.detected_at_ms.get_or_insert(record.at_ms);
            }
        }
    }
}

fn record_false_notification(
    state: &mut ThresholdState,
    brain_id: &str,
    positive: Option<bool>,
    is_normal: bool,
) {
    if positive == Some(true) && is_normal {
        let count = state
            .false_notifications_after_warmup
            .entry(brain_id.to_owned())
            .or_default();
        *count = count.saturating_add(1);
    }
}

struct SummaryInput<'a> {
    states: Vec<ThresholdState>,
    configs: &'a [BrainConfig],
    selectors: &'a BTreeMap<String, ScoreSelector>,
    labels: &'a [Label],
    evaluation_bounds: Option<(u64, u64)>,
    normal_duration_ms: Option<u128>,
    evaluation_records: &'a [RecordMeta],
    all_records: &'a [RecordMeta],
    detection_window_minutes: u64,
    detection_window_ms: u64,
    confirmed_normal_duration_ms: Option<u128>,
    normal_operation_hours_hint: Option<f64>,
}

fn build_summary(input: SummaryInput<'_>) -> Summary {
    let SummaryInput {
        states,
        configs,
        selectors,
        labels,
        evaluation_bounds,
        normal_duration_ms,
        evaluation_records,
        all_records,
        detection_window_minutes,
        detection_window_ms,
        confirmed_normal_duration_ms,
        normal_operation_hours_hint,
    } = input;
    let mut threshold_summaries = Vec::with_capacity(states.len());
    for state in states {
        let mut previous_detected = BTreeSet::new();
        let mut brain_summaries = Vec::with_capacity(configs.len());
        for config in configs {
            let brain_id = &config.id;
            let Some(selector) = selectors.get(brain_id) else {
                continue;
            };
            let events = state.events.get(brain_id).cloned().unwrap_or_default();
            let event_instances = event_instances_for_brain(
                labels,
                evaluation_bounds,
                evaluation_records,
                all_records,
                config,
                selector,
                detection_window_ms,
            );
            let failure_instances = event_instances
                .iter()
                .filter(|instance| instance.kind == "failure")
                .collect::<Vec<_>>();
            let important_instances = event_instances
                .iter()
                .filter(|instance| instance.kind == "important_change")
                .collect::<Vec<_>>();
            let selected_instances = event_instances.iter().collect::<Vec<_>>();
            let failure_detected = detected_keys(&events, &failure_instances);
            let important_detected = detected_keys(&events, &important_instances);
            let mut detected = failure_detected.clone();
            detected.extend(important_detected.iter().cloned());
            let additional = detected.difference(&previous_detected).count();
            previous_detected.extend(detected);
            let failure_delays = detection_delays(&events, &failure_instances);
            let important_delays = detection_delays(&events, &important_instances);
            let false_count = state
                .false_notifications
                .get(brain_id)
                .copied()
                .unwrap_or(0);
            let false_count_after_warmup = state
                .false_notifications_after_warmup
                .get(brain_id)
                .copied()
                .unwrap_or(0);
            let normal_duration =
                normal_duration_for_brain(labels, evaluation_bounds, evaluation_records, config);
            let normal_record_count =
                normal_record_count_for_brain(config, labels, evaluation_records);
            let all_record_bounds = record_bounds(all_records);
            let confirmed_normal_duration =
                normal_duration_for_brain(labels, all_record_bounds, all_records, config);
            let confirmed_normal_record_count =
                normal_record_count_for_brain(config, labels, all_records);
            let failure_stats = event_stats(&events, &failure_instances);
            let important_stats = event_stats(&events, &important_instances);
            let selected_stats = event_stats(&events, &selected_instances);
            let evaluable_selected = selected_stats.detected + selected_stats.missed;
            brain_summaries.push(BrainSummary {
                brain: brain_id.clone(),
                model_kind: config.model.kind.clone(),
                model_version: config.model.version,
                configuration_fingerprint: config.configuration_fingerprint(),
                schema_id: config.input.schema_id.clone(),
                schema_version: config.input.schema_version,
                score_metric: selector.metric.clone(),
                score_unit: selector.unit.clone(),
                score_direction: selector.direction.as_str().to_owned(),
                score_reaction_kind: format!("{:?}", selector.reaction_kind),
                score_label_kinds: selector.label_kinds.clone(),
                replay_threshold: state.threshold,
                live_threshold: config.reactions.threshold,
                normal_duration_hours: normal_duration.map(|value| value as f64 / 3_600_000.0),
                normal_record_count,
                selected_event_count: event_instances.len(),
                selected_evaluable_event_count: evaluable_selected,
                selected_recall: recall(&selected_stats, event_instances.len()),
                selected_evaluable_recall: recall(&selected_stats, evaluable_selected),
                selected_undetermined: selected_stats.undetermined,
                selected_missed: selected_stats.missed,
                failure_recall: recall(&failure_stats, failure_instances.len()),
                important_change_recall: recall(&important_stats, important_instances.len()),
                failure_mean_detection_delay_ms: mean(&failure_delays),
                important_change_mean_detection_delay_ms: mean(&important_delays),
                failure_undetermined: failure_stats.undetermined,
                important_change_undetermined: important_stats.undetermined,
                failure_missed: failure_stats.missed,
                important_change_missed: important_stats.missed,
                false_notifications_per_normal_hour: normal_duration
                    .filter(|duration| *duration > 0)
                    .map(|duration| false_count as f64 * 3_600_000.0 / duration as f64),
                false_notifications_per_1000_normal_records: (normal_record_count > 0)
                    .then(|| false_count as f64 * 1_000.0 / normal_record_count as f64),
                false_notification_count: false_count,
                confirmed_normal_duration_hours: confirmed_normal_duration
                    .map(|value| value as f64 / 3_600_000.0),
                confirmed_normal_record_count,
                false_notifications_per_confirmed_normal_hour: confirmed_normal_duration
                    .filter(|duration| *duration > 0)
                    .map(|duration| {
                        false_count_after_warmup as f64 * 3_600_000.0 / duration as f64
                    }),
                false_notifications_per_1000_confirmed_normal_records:
                    (confirmed_normal_record_count > 0).then(|| {
                        false_count_after_warmup as f64 * 1_000.0
                            / confirmed_normal_record_count as f64
                    }),
                false_notification_count_after_warmup: false_count_after_warmup,
                additional_detections_over_previous: additional,
            });
        }
        threshold_summaries.push(ThresholdSummary {
            threshold: state.threshold,
            brains: brain_summaries,
        });
    }
    Summary {
        detection_window_minutes,
        normal_duration_ms,
        normal_record_count: evaluation_records
            .iter()
            .filter(|record| {
                labels
                    .iter()
                    .any(|label| normal_label_matches(label, record))
            })
            .count(),
        confirmed_normal_duration_ms,
        confirmed_normal_record_count: all_records
            .iter()
            .filter(|record| {
                labels
                    .iter()
                    .any(|label| normal_label_matches(label, record))
            })
            .count(),
        normal_operation_hours_hint,
        thresholds: threshold_summaries,
    }
}

#[derive(Default, Clone, Copy)]
struct EventStats {
    detected: usize,
    undetermined: usize,
    missed: usize,
}

fn event_instances_for_brain(
    labels: &[Label],
    evaluation_bounds: Option<(u64, u64)>,
    evaluation_records: &[RecordMeta],
    all_records: &[RecordMeta],
    config: &BrainConfig,
    selector: &ScoreSelector,
    window_ms: u64,
) -> Vec<EventInstance> {
    labels
        .iter()
        .filter(|label| selector.label_kinds.iter().any(|kind| kind == &label.kind))
        .filter_map(|label| {
            let matching_record = evaluation_records.iter().any(|record| {
                brain_record_matches(record, config)
                    && event_label_matches(label, record, window_ms)
            });
            let in_evaluation = event_is_in_evaluation(label, evaluation_bounds);
            if !in_evaluation && !matching_record {
                return None;
            }
            if !label_scope_matches_config(label, config)
                || !label_belongs_to_brain(label, all_records, config)
            {
                None
            } else {
                // Keep every event in the evaluation interval. A missing
                // candidate is scored as undetermined or missed; it must not
                // disappear from the denominator.
                Some(EventInstance {
                    key: event_key(label),
                    kind: label.kind.clone(),
                    start_ms: label.start_ms,
                })
            }
        })
        .collect()
}

fn event_stats(events: &BTreeMap<String, EventResult>, labels: &[&EventInstance]) -> EventStats {
    let mut stats = EventStats::default();
    for instance in labels {
        match events.get(&instance.key) {
            Some(result) if result.detected_at_ms.is_some() => stats.detected += 1,
            Some(result) if result.evaluable_count > 0 => stats.missed += 1,
            _ => stats.undetermined += 1,
        }
    }
    stats
}

fn recall(stats: &EventStats, total: usize) -> Option<f64> {
    (total > 0).then(|| stats.detected as f64 / total as f64)
}

fn detected_keys(
    events: &BTreeMap<String, EventResult>,
    labels: &[&EventInstance],
) -> BTreeSet<String> {
    labels
        .iter()
        .filter_map(|instance| {
            events
                .get(&instance.key)
                .and_then(|result| result.detected_at_ms)
                .map(|_| instance.key.clone())
        })
        .collect()
}

fn detection_delays(events: &BTreeMap<String, EventResult>, labels: &[&EventInstance]) -> Vec<f64> {
    labels
        .iter()
        .filter_map(|instance| {
            events
                .get(&instance.key)
                .and_then(|result| result.detected_at_ms)
                .map(|detected| detected.saturating_sub(instance.start_ms) as f64)
        })
        .collect()
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn report_is_positive(
    report: &habitua::ModelReport,
    threshold: f32,
    selector: &ScoreSelector,
) -> Option<bool> {
    if report.readiness != Readiness::Evaluated {
        return None;
    }
    let metrics = report
        .metrics
        .iter()
        .filter(|metric| {
            metric_name_matches(&metric.name, &selector.metric)
                && metric.unit == selector.unit
                && metric_direction_matches(metric.direction, selector.direction)
        })
        .map(|metric| metric.value)
        .collect::<Vec<_>>();
    (!metrics.is_empty()).then(|| match selector.direction {
        ScoreDirection::Higher => metrics
            .iter()
            .any(|value| *value >= threshold && (threshold > 0.0 || *value > 0.0)),
        ScoreDirection::Lower => metrics
            .iter()
            .any(|value| *value <= threshold && (threshold > 0.0 || *value < 0.0)),
        ScoreDirection::TwoSided => metrics
            .iter()
            .any(|value| value.abs() >= threshold && (threshold > 0.0 || *value != 0.0)),
    })
}

fn reaction_output(reaction: Reaction) -> ReactionOutput {
    ReactionOutput {
        brain: reaction.brain,
        target: reaction.target,
        kind: format!("{:?}", reaction.kind),
        strength: reaction.strength,
        readiness: format!("{:?}", reaction.readiness),
        metrics: reaction.metrics.into_iter().map(metric_output).collect(),
        evidence: reaction.evidence.into_iter().map(evidence_output).collect(),
    }
}

fn metric_output(metric: Metric) -> MetricOutput {
    MetricOutput {
        name: metric.name,
        value: metric.value,
        unit: metric.unit,
        direction: format!("{:?}", metric.direction),
    }
}

fn evidence_output(evidence: Evidence) -> EvidenceOutput {
    EvidenceOutput {
        name: evidence.name,
        current: evidence.current,
        baseline: evidence.baseline,
        sample_count: evidence.sample_count,
        window_ms: evidence.window.map(|window| window.as_millis()),
        reference_ids: evidence.reference_ids,
        detail: evidence.detail,
    }
}

fn normalize_record(
    record: &mut Record,
    experiment: &Experiment,
) -> Result<(), Box<dyn std::error::Error>> {
    if record.id.is_empty() || record.stream.is_empty() || record.target.is_empty() {
        return Err("record id, stream, and target must be non-empty"
            .to_owned()
            .into());
    }
    let fallback_id = experiment
        .feature_schema_id
        .clone()
        .unwrap_or_else(default_schema_id);
    let fallback_version = experiment
        .feature_schema_version
        .unwrap_or_else(default_schema_version);
    let (schema_id, schema_version_from_text) = record
        .schema_id
        .as_deref()
        .or(record.schema.as_deref())
        .map(parse_schema)
        .transpose()?
        .unwrap_or((fallback_id, Some(fallback_version)));
    record.schema_id = Some(schema_id);
    record.feature_schema_version = Some(
        record
            .feature_schema_version
            .or(schema_version_from_text)
            .unwrap_or(fallback_version),
    );
    if record.context.is_empty() {
        return Err("record context must be non-empty".to_owned().into());
    }
    Ok(())
}

fn parse_schema(value: &str) -> Result<(String, Option<u32>), Box<dyn std::error::Error>> {
    if let Some((schema_id, version)) = value.rsplit_once('/')
        && let Ok(version) = version.parse::<u32>()
    {
        if schema_id.is_empty() || version == 0 {
            return Err("schema identifier or version is invalid".to_owned().into());
        }
        return Ok((schema_id.to_owned(), Some(version)));
    }
    if value.is_empty() {
        return Err("schema identifier must be non-empty".to_owned().into());
    }
    Ok((value.to_owned(), None))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct NormalScope {
    stream: Option<String>,
    target: Option<String>,
    context: Option<String>,
    schema_id: Option<String>,
    schema_version: Option<u32>,
}

fn normal_duration_ms(labels: &[Label], records: &[RecordMeta]) -> Option<u128> {
    normal_duration_for_scopes(labels, record_bounds(records), |_| true)
}

fn record_bounds(records: &[RecordMeta]) -> Option<(u64, u64)> {
    records
        .iter()
        .map(|record| record.at_ms)
        .min()
        .zip(records.iter().map(|record| record.at_ms).max())
}

fn normal_duration_for_brain(
    labels: &[Label],
    bounds: Option<(u64, u64)>,
    records: &[RecordMeta],
    config: &BrainConfig,
) -> Option<u128> {
    normal_duration_for_scopes(labels, bounds, |label| {
        label_scope_matches_config(label, config) && label_belongs_to_brain(label, records, config)
    })
}

fn normal_duration_for_scopes<F>(
    labels: &[Label],
    bounds: Option<(u64, u64)>,
    mut include: F,
) -> Option<u128>
where
    F: FnMut(&Label) -> bool,
{
    let (evaluation_start, evaluation_end) = bounds?;
    let mut intervals = BTreeMap::<NormalScope, Vec<(u64, u64)>>::new();
    for label in labels
        .iter()
        .filter(|label| label.kind == NORMAL_LABEL && include(label))
    {
        // An interval is required for a duration denominator. A point label
        // is used for per-record normality only and contributes no wall time.
        let Some(label_end) = label.end_ms else {
            continue;
        };
        let start = label.start_ms.max(evaluation_start);
        let end = label_end.min(evaluation_end);
        if end <= start {
            continue;
        }
        let scope = NormalScope {
            stream: label.stream.clone(),
            target: label.target.clone(),
            context: label.context.clone(),
            schema_id: label.schema_id.clone(),
            schema_version: label.schema_version,
        };
        intervals.entry(scope).or_default().push((start, end));
    }
    let mut total = 0_u128;
    for values in intervals.values_mut() {
        values.sort_unstable();
        let mut merged = Vec::<(u64, u64)>::new();
        for (start, end) in values.drain(..) {
            if let Some((_, previous_end)) = merged.last_mut()
                && start <= *previous_end
            {
                *previous_end = (*previous_end).max(end);
            } else {
                merged.push((start, end));
            }
        }
        total = total.saturating_add(
            merged
                .into_iter()
                .map(|(start, end)| u128::from(end) - u128::from(start))
                .sum::<u128>(),
        );
    }
    (total > 0).then_some(total)
}

fn normal_record_count_for_brain(
    config: &BrainConfig,
    labels: &[Label],
    records: &[RecordMeta],
) -> usize {
    records
        .iter()
        .filter(|record| {
            brain_record_matches(record, config)
                && labels
                    .iter()
                    .any(|label| normal_label_matches(label, record))
        })
        .count()
}

fn normal_label_matches(label: &Label, record: &RecordMeta) -> bool {
    if label.kind != NORMAL_LABEL
        || !scope_matches(label, record)
        || !identity_matches(label, record)
    {
        return false;
    }
    match label.end_ms {
        Some(end) => record.at_ms >= label.start_ms && record.at_ms <= end,
        None => record.at_ms == label.start_ms,
    }
}

fn event_label_matches(label: &Label, record: &RecordMeta, window_ms: u64) -> bool {
    if !is_event_label(&label.kind)
        || !scope_matches(label, record)
        || !identity_matches(label, record)
    {
        return false;
    }
    let deadline = label
        .end_ms
        .unwrap_or(u64::MAX)
        .min(label.start_ms.saturating_add(window_ms));
    record.at_ms >= label.start_ms && record.at_ms <= deadline
}

fn scope_matches(label: &Label, record: &RecordMeta) -> bool {
    label
        .stream
        .as_deref()
        .is_none_or(|stream| stream == record.stream)
        && label
            .target
            .as_deref()
            .is_none_or(|target| target == record.target)
        && label
            .context
            .as_deref()
            .is_none_or(|context| context == record.context)
        && label
            .schema_id
            .as_deref()
            .is_none_or(|schema_id| schema_id == record.schema_id)
        && label
            .schema_version
            .is_none_or(|version| version == record.schema_version)
}

fn identity_matches(label: &Label, record: &RecordMeta) -> bool {
    label
        .record_id
        .as_deref()
        .is_none_or(|id| record.id == id || record.source_id.as_deref() == Some(id))
        && label.source_id.as_deref().is_none_or(|source_id| {
            record.source_id.as_deref() == Some(source_id) || record.id == source_id
        })
}

fn label_belongs_to_brain(label: &Label, records: &[RecordMeta], config: &BrainConfig) -> bool {
    let has_identity = label.record_id.is_some() || label.source_id.is_some();
    if !has_identity {
        return true;
    }

    let related_records = records
        .iter()
        .filter(|record| identity_matches(label, record));
    let mut found_related_record = false;
    let mut belongs_to_brain = false;
    for record in related_records {
        found_related_record = true;
        belongs_to_brain |= brain_record_matches(record, config);
    }
    if found_related_record {
        belongs_to_brain
    } else {
        // An explicit scope is sufficient to assign an event whose source
        // record is not present in this replay input. An unscoped unknown
        // identity cannot be assigned safely to a brain.
        label.target.is_some()
            || label.context.is_some()
            || label.schema_id.is_some()
            || label.schema_version.is_some()
    }
}

fn brain_record_matches(record: &RecordMeta, config: &BrainConfig) -> bool {
    record.target == config.input.target
        && record.context == config.input.context
        && record.schema_id == config.input.schema_id
        && record.schema_version == config.input.schema_version
}

fn label_scope_matches_config(label: &Label, config: &BrainConfig) -> bool {
    label
        .target
        .as_deref()
        .is_none_or(|target| target == config.input.target)
        && label
            .context
            .as_deref()
            .is_none_or(|context| context == config.input.context)
        && label
            .schema_id
            .as_deref()
            .is_none_or(|schema_id| schema_id == config.input.schema_id)
        && label
            .schema_version
            .is_none_or(|version| version == config.input.schema_version)
}

fn event_is_in_evaluation(label: &Label, bounds: Option<(u64, u64)>) -> bool {
    let Some((evaluation_start, evaluation_end)) = bounds else {
        return false;
    };
    let label_end = label.end_ms.unwrap_or(label.start_ms);
    label.start_ms <= evaluation_end && label_end >= evaluation_start
}

fn event_key(label: &Label) -> String {
    format!(
        "kind={:?};event={:?};record={:?};source={:?};start={};end={:?};stream={:?};target={:?};context={:?};schema={:?};schema_version={:?}",
        label.kind,
        label.event_id,
        label.record_id,
        label.source_id,
        label.start_ms,
        label.end_ms,
        label.stream,
        label.target,
        label.context,
        label.schema_id,
        label.schema_version,
    )
}

fn is_event_label(kind: &str) -> bool {
    EVENT_LABELS.contains(&kind)
}

fn validate_label(label: &Label) -> Result<(), io::Error> {
    if !is_event_label(&label.kind)
        && label.kind != NORMAL_LABEL
        && label.kind != UNDETERMINED_LABEL
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown label kind {}", label.kind),
        ));
    }
    if label.end_ms.is_some_and(|end| end < label.start_ms) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "label end_ms must not precede start_ms",
        ));
    }
    if label.stream.as_deref().is_some_and(str::is_empty)
        || label.target.as_deref().is_some_and(str::is_empty)
        || label.context.as_deref().is_some_and(str::is_empty)
        || label.schema_id.as_deref().is_some_and(str::is_empty)
        || label.event_id.as_deref().is_some_and(str::is_empty)
        || label.record_id.as_deref().is_some_and(str::is_empty)
        || label.source_id.as_deref().is_some_and(str::is_empty)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "label scope and identity values must be non-empty",
        ));
    }
    if label.schema_id.is_some() != label.schema_version.is_some()
        || label.schema_version.is_some_and(|version| version == 0)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "label schema_id and schema_version must be supplied together",
        ));
    }
    Ok(())
}

fn parse_reaction_kind(value: &str) -> Result<ReactionKind, Box<dyn std::error::Error>> {
    match value {
        "novel" => Ok(ReactionKind::Novel),
        "deviation" => Ok(ReactionKind::Deviation),
        "absence" => Ok(ReactionKind::Absence),
        "repeat" => Ok(ReactionKind::Repeat),
        "failure" => Ok(ReactionKind::Failure),
        other => Err(format!("unknown reaction kind {other}").into()),
    }
}

fn read_records(path: &str) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(line_number, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(serde_json::from_str::<Record>(&line).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("record line {}: {error}", line_number + 1),
                )
            })),
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn read_labels(path: &str) -> Result<Vec<Label>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let labels = BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(line_number, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(serde_json::from_str::<Label>(&line).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("label line {}: {error}", line_number + 1),
                )
            })),
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    for label in &labels {
        validate_label(label)?;
    }
    Ok(labels)
}

fn phase_for(at_ms: u64, warmup_until_ms: u64, calibration_until_ms: u64) -> &'static str {
    if at_ms <= warmup_until_ms {
        "warmup"
    } else if at_ms <= calibration_until_ms {
        "calibration"
    } else {
        "evaluation"
    }
}

fn default_target() -> String {
    "features".to_owned()
}

fn default_context() -> String {
    "default".to_owned()
}

fn default_schema_id() -> String {
    "features".to_owned()
}

fn default_schema_version() -> u32 {
    1
}

fn default_model_version() -> u32 {
    1
}

fn default_detection_window_minutes() -> u64 {
    DEFAULT_DETECTION_WINDOW_MINUTES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event_label() -> Label {
        Label {
            record_id: None,
            source_id: Some("source-1".to_owned()),
            event_id: Some("event-1".to_owned()),
            kind: "failure".to_owned(),
            start_ms: 1_000,
            end_ms: Some(1_500),
            stream: Some("stream-a".to_owned()),
            target: Some("template".to_owned()),
            context: None,
            schema_id: None,
            schema_version: None,
        }
    }

    fn record_meta(at_ms: u64) -> RecordMeta {
        RecordMeta {
            id: "derived-1".to_owned(),
            source_id: Some("source-1".to_owned()),
            stream: "stream-a".to_owned(),
            target: "template".to_owned(),
            context: "default".to_owned(),
            schema_id: "schema".to_owned(),
            schema_version: 1,
            at_ms,
        }
    }

    #[test]
    fn event_labels_match_source_scope_and_finite_window() {
        let label = event_label();
        let matching = record_meta(1_100);
        assert!(event_label_matches(&label, &matching, 200));

        let mut parent_label = event_label();
        parent_label.record_id = Some("source-1".to_owned());
        parent_label.source_id = None;
        assert!(event_label_matches(&parent_label, &matching, 200));

        let after_window = RecordMeta {
            at_ms: 1_201,
            ..matching.clone()
        };
        assert!(!event_label_matches(&label, &after_window, 200));

        let other_stream = RecordMeta {
            stream: "stream-b".to_owned(),
            ..matching
        };
        assert!(!event_label_matches(&label, &other_stream, 200));
        assert_ne!(event_key(&label), event_key(&parent_label));
    }

    #[test]
    fn threshold_scoring_keeps_the_first_detection() {
        let label = event_label();
        let labels = vec![label];
        let mut state = ThresholdState::new(0.5);
        let selector = ScoreSelector {
            metric: "failed".to_owned(),
            unit: "boolean".to_owned(),
            direction: ScoreDirection::Higher,
            reaction_kind: ReactionKind::Failure,
            label_kinds: vec!["failure".to_owned()],
        };
        let first = record_meta(1_100);
        let second = RecordMeta {
            at_ms: 1_200,
            ..first.clone()
        };
        record_threshold_result(ThresholdObservation {
            state: &mut state,
            brain_id: "brain-a",
            positive: Some(true),
            evaluable: true,
            selector: &selector,
            labels: &labels,
            record: &first,
            window_ms: 1_000,
            is_normal: false,
        });
        record_threshold_result(ThresholdObservation {
            state: &mut state,
            brain_id: "brain-a",
            positive: Some(true),
            evaluable: true,
            selector: &selector,
            labels: &labels,
            record: &second,
            window_ms: 1_000,
            is_normal: false,
        });
        let key = event_key(&labels[0]);
        assert_eq!(state.events["brain-a"][&key].detected_at_ms, Some(1_100));
    }

    #[test]
    fn normal_intervals_are_unioned_per_stream_and_target() {
        let records = vec![
            RecordMeta {
                id: "a0".to_owned(),
                source_id: None,
                stream: "stream-a".to_owned(),
                target: "template".to_owned(),
                context: "default".to_owned(),
                schema_id: "schema".to_owned(),
                schema_version: 1,
                at_ms: 0,
            },
            RecordMeta {
                id: "a20".to_owned(),
                source_id: None,
                stream: "stream-a".to_owned(),
                target: "template".to_owned(),
                context: "default".to_owned(),
                schema_id: "schema".to_owned(),
                schema_version: 1,
                at_ms: 20,
            },
            RecordMeta {
                id: "b0".to_owned(),
                source_id: None,
                stream: "stream-b".to_owned(),
                target: "template".to_owned(),
                context: "default".to_owned(),
                schema_id: "schema".to_owned(),
                schema_version: 1,
                at_ms: 0,
            },
            RecordMeta {
                id: "b20".to_owned(),
                source_id: None,
                stream: "stream-b".to_owned(),
                target: "template".to_owned(),
                context: "default".to_owned(),
                schema_id: "schema".to_owned(),
                schema_version: 1,
                at_ms: 20,
            },
        ];
        let labels = vec![
            Label {
                record_id: None,
                source_id: None,
                event_id: None,
                kind: NORMAL_LABEL.to_owned(),
                start_ms: 0,
                end_ms: Some(20),
                stream: Some("stream-a".to_owned()),
                target: Some("template".to_owned()),
                context: None,
                schema_id: None,
                schema_version: None,
            },
            Label {
                record_id: None,
                source_id: None,
                event_id: None,
                kind: NORMAL_LABEL.to_owned(),
                start_ms: 10,
                end_ms: Some(20),
                stream: Some("stream-a".to_owned()),
                target: Some("template".to_owned()),
                context: None,
                schema_id: None,
                schema_version: None,
            },
            Label {
                record_id: None,
                source_id: None,
                event_id: None,
                kind: NORMAL_LABEL.to_owned(),
                start_ms: 0,
                end_ms: Some(10),
                stream: Some("stream-b".to_owned()),
                target: Some("template".to_owned()),
                context: None,
                schema_id: None,
                schema_version: None,
            },
        ];
        assert_eq!(normal_duration_ms(&labels, &records), Some(30));
    }
}
