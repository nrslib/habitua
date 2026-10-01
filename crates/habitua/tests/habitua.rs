use std::collections::BTreeMap;
use std::io::Cursor;
use std::time::Duration;

use habitua::{
    BrainConfig, BrainInput, BrainInputSpec, BrainLearningPolicy, BrainMode, BrainRunner, Config,
    DecayingMemory, EncodeError, Encoder, FailureModel, FailureModelConfig, FailureObservation,
    FlyHashEncoder, FlyModel, Habituation, LearningMode, LearningSignal, Memory, MemoryError,
    MemoryObservation, MemoryObservationError, MemoryStats, MemoryTransaction, Model, ModelError,
    ModelInput, ModelRegistry, ModelReport, ModelSettings, ModelSpec, NearestNeighborConfig,
    NearestNeighborModel, NumericDeviationConfig, NumericDeviationError, NumericDeviationModel,
    NumericDirection, Pattern, ReactionKind, ReactionPolicy, Readiness, ResourcePolicy,
    StatePolicy, Timestamp,
};
use proptest::prelude::*;

fn test_config() -> Config {
    Config {
        dimensions: 4,
        encoder_units: 64,
        k: 8,
        seed: 41,
        short_time_constant: Duration::from_secs(10),
        long_time_constant: Duration::from_secs(1_000),
        long_learning_policy: habitua::LongLearningPolicy::Revisit {
            interval: Duration::from_secs(10),
        },
        ..Config::default()
    }
}

fn learn<E, M>(
    engine: &mut Habituation<E, M>,
    stream: &str,
    features: &[f32],
    at: Timestamp,
    weight: f32,
) where
    E: Encoder,
    M: Memory,
{
    let evaluation = engine
        .evaluate(stream, features, at)
        .expect("valid evaluation");
    engine
        .commit(evaluation.update, weight)
        .expect("valid commit");
}

fn observe_memory(
    memory: &mut DecayingMemory,
    pattern: &Pattern,
    at: Timestamp,
    weight: f32,
) -> MemoryObservation {
    let transaction = memory
        .observe(pattern, at)
        .expect("valid memory observation");
    let (observation, update) = transaction.into_parts();
    memory.commit_with_weight(update, weight);
    observation
}

fn assert_interval(value: Option<f32>) {
    let value = value.expect("Fly exposes this metric");
    assert!((0.0..=1.0).contains(&value), "value={value}");
}

#[test]
fn repeated_vector_becomes_familiar_after_commit() {
    let mut engine = FlyModel::with_config(test_config()).unwrap();
    let features = [0.2, -0.4, 0.6, 0.8];
    let first = engine
        .evaluate("screen", &features, Timestamp::ZERO)
        .unwrap();
    assert_eq!(first.novelty_short, Some(1.0));
    assert_eq!(engine.stats("screen"), None);
    engine.commit(first.update, 1.0).unwrap();
    let repeated = engine
        .evaluate("screen", &features, Timestamp::from_secs(1))
        .unwrap();
    assert!(repeated.novelty_short.unwrap() < 1.0);
    assert!(repeated.familiarity_short.unwrap() > 0.0);
    assert_eq!(repeated.recency, Some(Duration::from_secs(1)));
}

#[test]
fn zero_weight_records_arrival_without_learning() {
    let mut engine = FlyModel::with_config(test_config()).unwrap();
    let features = [1.0, 0.0, 0.0, 0.0];
    let first = engine
        .evaluate("screen", &features, Timestamp::ZERO)
        .unwrap();
    engine.commit(first.update, 0.0).unwrap();
    assert_eq!(
        engine.stats("screen").unwrap().last_observed,
        Some(Timestamp::ZERO)
    );
    let second = engine
        .evaluate("screen", &features, Timestamp::ZERO)
        .unwrap();
    assert_eq!(second.familiarity_short, Some(0.0));
    engine.commit(second.update, 0.5).unwrap();
    assert!(engine.stats("screen").unwrap().remembered_unit_count > 0);
}

#[test]
fn zero_weight_records_arrival_metadata_for_numeric_and_neighbor_models() {
    let numeric_config = NumericDeviationConfig {
        minimum_samples: 1,
        ..NumericDeviationConfig::default()
    };
    let mut numeric = NumericDeviationModel::with_config(["value"], numeric_config).unwrap();
    let numeric_first = numeric
        .evaluate("stream", &[10.0], Timestamp::ZERO)
        .unwrap();
    numeric.commit(numeric_first.update, 0.0).unwrap();
    let numeric_second = numeric
        .evaluate("stream", &[10.0], Timestamp::from_secs(1))
        .unwrap();
    assert_eq!(numeric_second.readiness, Readiness::BaselineInsufficient);
    assert_eq!(numeric_second.recency, Some(Duration::from_secs(1)));

    let mut nearest = NearestNeighborModel::new(NearestNeighborConfig {
        dimensions: 2,
        ..NearestNeighborConfig::default()
    })
    .unwrap();
    let nearest_first = nearest
        .evaluate("stream", &[1.0, 0.0], Timestamp::ZERO)
        .unwrap();
    nearest.commit(nearest_first.update, 0.0).unwrap();
    let nearest_second = nearest
        .evaluate("stream", &[1.0, 0.0], Timestamp::from_secs(1))
        .unwrap();
    assert_eq!(nearest_second.readiness, Readiness::BaselineInsufficient);
    assert_eq!(nearest_second.recency, Some(Duration::from_secs(1)));
}

#[test]
fn time_decay_and_revisit_separate_short_and_long_memory() {
    let mut engine = FlyModel::with_config(test_config()).unwrap();
    let features = [1.0, 0.0, 0.0, 0.0];
    let first = engine
        .evaluate("audio", &features, Timestamp::ZERO)
        .unwrap();
    engine.commit(first.update, 1.0).unwrap();
    let immediate = engine
        .evaluate("audio", &features, Timestamp::from_secs(1))
        .unwrap();
    assert!(immediate.novelty_short.unwrap() < 1.0);
    assert_eq!(immediate.novelty_long, Some(1.0));
    engine.commit(immediate.update, 1.0).unwrap();
    let revisit = engine
        .evaluate("audio", &features, Timestamp::from_secs(11))
        .unwrap();
    engine.commit(revisit.update, 1.0).unwrap();
    let after_revisit = engine
        .evaluate("audio", &features, Timestamp::from_secs(12))
        .unwrap();
    assert!(after_revisit.novelty_long.unwrap() < immediate.novelty_long.unwrap());
    let recovered = engine
        .evaluate("audio", &features, Timestamp::from_secs(601))
        .unwrap();
    assert!(recovered.novelty_short.unwrap() > after_revisit.novelty_short.unwrap());
    assert!(recovered.novelty_long.unwrap() > after_revisit.novelty_long.unwrap());
}

#[test]
fn repeated_normal_requires_contiguous_same_pattern_experiences() {
    let config = Config {
        encoder_units: 4,
        long_learning_policy: habitua::LongLearningPolicy::RepeatedNormal {
            consecutive_observations: 2,
        },
        ..test_config()
    };
    let mut memory = DecayingMemory::new(&config).unwrap();
    let first = Pattern::new(vec![0]).unwrap();
    let other = Pattern::new(vec![1]).unwrap();

    observe_memory(&mut memory, &first, Timestamp::ZERO, 1.0);
    observe_memory(&mut memory, &other, Timestamp::from_secs(1), 1.0);
    observe_memory(&mut memory, &first, Timestamp::from_secs(2), 1.0);
    let after_non_contiguous = memory.observe(&first, Timestamp::from_secs(3)).unwrap();
    assert_eq!(after_non_contiguous.observation().long_familiarity, 0.0);

    observe_memory(&mut memory, &first, Timestamp::from_secs(3), 1.0);
    let after_contiguous = memory.observe(&first, Timestamp::from_secs(4)).unwrap();
    assert!(after_contiguous.observation().long_familiarity > 0.0);
}

#[test]
fn streams_are_independent() {
    let mut engine = FlyModel::with_config(test_config()).unwrap();
    let features = [0.3, 0.1, -0.2, 0.4];
    learn(&mut engine, "screen", &features, Timestamp::ZERO, 1.0);
    let audio = engine
        .evaluate("audio", &features, Timestamp::from_secs(1))
        .unwrap();
    assert_eq!(audio.familiarity_short, Some(0.0));
    assert_eq!(engine.stats("audio"), None);
}

#[test]
fn factory_does_not_allow_preloaded_memory_or_state_leak() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let factory_calls = std::sync::Arc::clone(&calls);
    let result = Habituation::new(
        FixedEncoder,
        move || {
            if factory_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                Ok(CountingMemory::default())
            } else {
                Ok(CountingMemory {
                    observations: 1,
                    last_observed: Some(Timestamp::ZERO),
                })
            }
        },
        test_config(),
    )
    .unwrap();
    let evaluation = result.evaluate("new", &[0.0; 4], Timestamp::ZERO);
    assert!(matches!(
        evaluation,
        Err(habitua::ObserveError::Memory(
            MemoryError::FactoryReturnedNonEmpty
        ))
    ));
    assert_eq!(result.stats("new"), None);
}

#[test]
fn failed_first_observation_does_not_leave_an_empty_stream() {
    let config = test_config();
    let factory_config = config.clone();
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let factory_calls = std::sync::Arc::clone(&calls);
    let encoder = FlyHashEncoder::from_config(&config).unwrap();
    let engine = Habituation::new(
        encoder,
        move || {
            if factory_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                DecayingMemory::new(&factory_config).map_err(|_| MemoryError::FactoryCreationFailed)
            } else {
                Err(MemoryError::FactoryCreationFailed)
            }
        },
        config.clone(),
    )
    .unwrap();
    assert!(matches!(
        engine.evaluate("new", &[0.0; 4], Timestamp::ZERO),
        Err(habitua::ObserveError::Memory(
            MemoryError::FactoryCreationFailed
        ))
    ));
    assert_eq!(engine.stats("new"), None);

    let mut bytes = Vec::new();
    engine.save(&mut bytes).unwrap();
    let restore_config = config;
    let restored = Habituation::<FlyHashEncoder, DecayingMemory>::load_with(
        &mut Cursor::new(bytes),
        FlyHashEncoder::from_config(&restore_config).unwrap(),
        move || {
            DecayingMemory::new(&restore_config).map_err(|_| MemoryError::FactoryCreationFailed)
        },
    )
    .unwrap();
    assert_eq!(restored.stats("new"), None);
}

#[test]
fn custom_memory_postcondition_failure_does_not_change_stream() {
    let mut engine = Habituation::new(
        FixedEncoder,
        || Ok(CountingMemory::default()),
        test_config(),
    )
    .unwrap();
    for timestamp in 0..3 {
        let evaluation = engine
            .evaluate("custom", &[0.0; 4], Timestamp::from_secs(timestamp))
            .unwrap();
        engine.commit(evaluation.update, 1.0).unwrap();
    }
    let invalid = engine.evaluate("custom", &[0.0; 4], Timestamp::from_secs(3));
    assert!(matches!(
        invalid,
        Err(habitua::ObserveError::InvalidMemoryObservation(
            MemoryObservationError::ValueOutOfRange {
                field: "short_familiarity"
            }
        ))
    ));
    assert_eq!(
        engine.stats("custom").unwrap().last_observed,
        Some(Timestamp::from_secs(2))
    );
}

#[test]
fn invalid_external_features_return_errors() {
    let engine = FlyModel::with_config(test_config()).unwrap();
    assert!(matches!(
        engine.evaluate("screen", &[], Timestamp::ZERO),
        Err(habitua::ObserveError::Encode(EncodeError::EmptyFeatures))
    ));
    assert!(matches!(
        engine.evaluate("screen", &[f32::NAN; 4], Timestamp::ZERO),
        Err(habitua::ObserveError::Encode(
            EncodeError::NonFiniteFeature { .. }
        ))
    ));
    assert!(
        engine
            .evaluate("screen", &[0.0, 1.0], Timestamp::ZERO)
            .is_err()
    );
}

#[test]
fn decaying_memory_equation_and_equal_timestamp_lru_are_deterministic() {
    let mut config = test_config();
    config.long_learning_policy = habitua::LongLearningPolicy::RepeatedNormal {
        consecutive_observations: 2,
    };
    let pattern = Pattern::new(vec![1]).unwrap();
    let mut memory = DecayingMemory::new(&config).unwrap();
    let first = observe_memory(&mut memory, &pattern, Timestamp::ZERO, 1.0);
    assert_eq!(first.familiarity, 0.0);
    let repeated = observe_memory(&mut memory, &pattern, Timestamp::ZERO, 1.0);
    assert!((repeated.short_familiarity - 0.35).abs() < 1.0e-6);
    assert_eq!(repeated.long_familiarity, 0.0);
    assert!((repeated.familiarity - 0.21).abs() < 1.0e-6);
    let long_repeated = observe_memory(&mut memory, &pattern, Timestamp::ZERO, 1.0);
    assert!((long_repeated.long_familiarity - 0.12).abs() < 1.0e-6);

    let mut bounded = Config {
        max_patterns: 2,
        ..test_config()
    };
    bounded.long_learning_policy = habitua::LongLearningPolicy::Revisit {
        interval: Duration::from_secs(1),
    };
    let patterns = [
        Pattern::new(vec![0]).unwrap(),
        Pattern::new(vec![1]).unwrap(),
        Pattern::new(vec![2]).unwrap(),
    ];
    let mut memory = DecayingMemory::new(&bounded).unwrap();
    observe_memory(&mut memory, &patterns[0], Timestamp::ZERO, 1.0);
    observe_memory(&mut memory, &patterns[1], Timestamp::ZERO, 1.0);
    observe_memory(&mut memory, &patterns[0], Timestamp::ZERO, 1.0);
    observe_memory(&mut memory, &patterns[2], Timestamp::ZERO, 1.0);
    assert_eq!(memory.last_seen(&patterns[0]), Some(Timestamp::ZERO));
    assert_eq!(memory.last_seen(&patterns[1]), None);
    assert_eq!(memory.last_seen(&patterns[2]), Some(Timestamp::ZERO));
}

#[test]
fn new_memory_rejects_legacy_every_commit_policy() {
    let config = Config {
        long_learning_policy: habitua::LongLearningPolicy::LegacyEveryCommit,
        ..test_config()
    };
    assert!(matches!(
        DecayingMemory::new(&config),
        Err(habitua::MemoryConfigError::LegacyLearningPolicy)
    ));
}

#[test]
fn numeric_reports_baseline_readiness_direction_and_a_b_a_recency() {
    let config = NumericDeviationConfig {
        mad_floor: 0.1,
        minimum_samples: 2,
        direction: NumericDirection::Up,
        ..NumericDeviationConfig::default()
    };
    let mut model = NumericDeviationModel::with_config(["latency_ms"], config).unwrap();
    let a = model.evaluate("runtime", &[10.0], Timestamp::ZERO).unwrap();
    assert_eq!(a.readiness, Readiness::BaselineInsufficient);
    model.commit(a.update, 1.0).unwrap();
    let b = model
        .evaluate("runtime", &[11.0], Timestamp::from_secs(1))
        .unwrap();
    model.commit(b.update, 1.0).unwrap();
    let high = model
        .evaluate("runtime", &[16.0], Timestamp::from_secs(2))
        .unwrap();
    assert_eq!(high.readiness, Readiness::Evaluated);
    assert!(high.diagnostics.deviations[0].z_plus > 10.0);
    model.commit(high.update, 0.0).unwrap();
    let returned = model
        .evaluate("runtime", &[10.0], Timestamp::from_secs(3))
        .unwrap();
    assert_eq!(returned.recency, Some(Duration::from_secs(3)));
    assert_eq!(
        returned.diagnostics.deviations[0].direction,
        NumericDirection::Up
    );
}

#[test]
fn numeric_save_load_preserves_window_and_recency() {
    let config = NumericDeviationConfig {
        minimum_samples: 1,
        sample_window: Duration::from_secs(30),
        ..NumericDeviationConfig::default()
    };
    let mut original = NumericDeviationModel::with_config(["latency_ms"], config).unwrap();
    let first = original
        .evaluate("runtime", &[10.0], Timestamp::ZERO)
        .unwrap();
    original.commit(first.update, 1.0).unwrap();
    let mut bytes = Vec::new();
    original.save(&mut bytes).unwrap();
    let restored = NumericDeviationModel::load(&mut Cursor::new(bytes)).unwrap();
    let left = original
        .evaluate("runtime", &[20.0], Timestamp::from_secs(5))
        .unwrap();
    let right = restored
        .evaluate("runtime", &[20.0], Timestamp::from_secs(5))
        .unwrap();
    assert_eq!(left.diagnostics, right.diagnostics);
    assert_eq!(left.recency, right.recency);
}

#[test]
fn numeric_even_sample_window_uses_documented_lower_median() {
    let config = NumericDeviationConfig {
        minimum_samples: 1,
        ..NumericDeviationConfig::default()
    };
    let mut model = NumericDeviationModel::with_config(["value"], config).unwrap();
    for (timestamp, value) in [(0, 50.0), (1, 55.0)] {
        let evaluation = model
            .evaluate("stream", &[value], Timestamp::from_secs(timestamp))
            .unwrap();
        model.commit(evaluation.update, 1.0).unwrap();
    }
    let evaluation = model
        .evaluate("stream", &[55.0], Timestamp::from_secs(2))
        .unwrap();
    assert_eq!(evaluation.diagnostics.deviations[0].baseline, Some(50.0));
}

#[test]
fn nearest_neighbor_merges_duplicates_and_is_bounded() {
    let config = NearestNeighborConfig {
        dimensions: 2,
        max_representatives: 2,
        merge_distance: 0.1,
        ..NearestNeighborConfig::default()
    };
    let mut model = NearestNeighborModel::new(config).unwrap();
    for (index, features) in [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        .into_iter()
        .enumerate()
    {
        let evaluation = model
            .evaluate("screen", &features, Timestamp::from_secs(index as u64))
            .unwrap();
        model.commit(evaluation.update, 1.0).unwrap();
    }
    assert_eq!(model.stats("screen").unwrap().representative_count, 2);
    let mut bytes = Vec::new();
    model.save(&mut bytes).unwrap();
    let restored = NearestNeighborModel::load(&mut Cursor::new(bytes)).unwrap();
    assert_eq!(restored.stats("screen"), model.stats("screen"));
}

#[test]
fn different_model_instances_reject_each_others_updates() {
    let config = NumericDeviationConfig::default();
    let one = NumericDeviationModel::with_config(["one"], config.clone()).unwrap();
    let mut two = NumericDeviationModel::with_config(["one"], config).unwrap();
    let update = one
        .evaluate("stream", &[1.0], Timestamp::ZERO)
        .unwrap()
        .update;
    assert!(matches!(
        two.commit(update, 1.0),
        Err(habitua::ModelError::ForeignUpdate)
    ));
}

#[test]
fn runner_evaluates_heterogeneous_brains_before_commit_and_keeps_shadow_out_of_active() {
    let fly_config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    };
    let numeric_config = NumericDeviationConfig {
        minimum_samples: 1,
        ..NumericDeviationConfig::default()
    };
    let fly = brain_config(
        "fly-shadow",
        BrainMode::Shadow,
        "fly",
        ModelSettings::Fly(fly_config.clone()),
        1,
    );
    let numeric = brain_config(
        "numeric-active",
        BrainMode::Active,
        "numeric",
        ModelSettings::Numeric {
            names: vec!["latency_ms".to_owned()],
            config: numeric_config,
        },
        1,
    );
    let mut runner = BrainRunner::with_builtins([fly, numeric]).unwrap();
    let evaluation = runner
        .evaluate_all(&runner_input("r1", "target", &[10.0], 0))
        .unwrap();
    assert_eq!(evaluation.brains.len(), 2);
    assert!(
        evaluation
            .brains
            .iter()
            .find(|brain| brain.brain == "fly-shadow")
            .unwrap()
            .report
            .reactions
            .iter()
            .all(|reaction| reaction.kind == ReactionKind::Novel)
    );
    assert!(
        evaluation
            .active_reactions
            .iter()
            .all(|reaction| reaction.brain == "numeric-active")
    );
    let commits = runner.commit_all(evaluation).unwrap();
    assert_eq!(commits.len(), 2);
    assert_eq!(
        runner.state("fly-shadow").unwrap().last_learning_weight,
        Some(1.0)
    );
}

#[test]
fn generic_habituation_can_be_registered_in_the_brain_runner() {
    let model_config = test_config();
    let factory_config = model_config.clone();
    let mut registry = ModelRegistry::empty();
    registry
        .register::<Habituation<FixedEncoder, DecayingMemory>, _, _>(
            "custom_habituation",
            move |brain| {
                let config = factory_config.clone();
                let memory_config = config.clone();
                Habituation::new(
                    FixedEncoder,
                    move || {
                        DecayingMemory::new(&memory_config)
                            .map_err(|_| MemoryError::FactoryCreationFailed)
                    },
                    config,
                )
                .map_err(|error| habitua::RegistryError::Model {
                    brain: brain.id.clone(),
                    reason: error.to_string(),
                })
            },
            |evaluation, _config| ModelReport {
                novelty_short: evaluation.novelty_short,
                novelty_long: evaluation.novelty_long,
                familiarity_short: evaluation.familiarity_short,
                familiarity_long: evaluation.familiarity_long,
                recency: evaluation.recency,
                pattern_id: evaluation.pattern_id,
                readiness: evaluation.readiness.clone(),
                learning_signal: LearningSignal::Unknown,
                metrics: Vec::new(),
                evidence: Vec::new(),
                reactions: Vec::new(),
            },
        )
        .unwrap();
    let brain = brain_config(
        "custom",
        BrainMode::Active,
        "custom_habituation",
        ModelSettings::Custom(BTreeMap::new()),
        1,
    );
    let configs = vec![brain];
    let mut runner = BrainRunner::new(configs.clone(), &registry).unwrap();
    let evaluation = runner
        .evaluate_all(&runner_input("stream", "target", &[1.0], 0))
        .unwrap();
    runner.commit_all(evaluation).unwrap();
    let mut bytes = Vec::new();
    runner.save(&mut bytes).unwrap();
    let restored = BrainRunner::load(&mut Cursor::new(bytes), &configs, &registry).unwrap();
    assert_eq!(restored.states(), runner.states());
}

#[test]
fn runner_can_build_a_numeric_baseline_before_reactions_are_ready() {
    let numeric = brain_config(
        "numeric",
        BrainMode::Active,
        "numeric",
        ModelSettings::Numeric {
            names: vec!["latency_ms".to_owned()],
            config: NumericDeviationConfig {
                minimum_samples: 2,
                ..NumericDeviationConfig::default()
            },
        },
        1,
    );
    let mut numeric = numeric;
    numeric.learning.require_evaluated = false;
    let mut runner = BrainRunner::with_builtins([numeric]).unwrap();
    for (position, value) in [10.0, 10.5].into_iter().enumerate() {
        let evaluation = runner
            .evaluate_all(&runner_input(
                "runtime",
                "target",
                &[value],
                position as u64,
            ))
            .unwrap();
        runner.commit_all(evaluation).unwrap();
    }
    let evaluation = runner
        .evaluate_all(&runner_input("runtime", "target", &[10.0], 2))
        .unwrap();
    assert_eq!(evaluation.brains[0].report.readiness, Readiness::Evaluated);
}

#[test]
fn runner_rejects_foreign_updates_and_schema_mismatch() {
    let config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    };
    let brain = brain_config(
        "one",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(config),
        1,
    );
    let mut first = BrainRunner::with_builtins([brain.clone()]).unwrap();
    let mut second = BrainRunner::with_builtins([brain.clone()]).unwrap();
    let evaluation = first
        .evaluate_all(&runner_input("r", "target", &[1.0], 0))
        .unwrap();
    let update = evaluation
        .brains
        .into_iter()
        .next()
        .unwrap()
        .update
        .unwrap();
    assert!(matches!(
        second.commit(update, 1.0),
        Err(habitua::RunnerError::ForeignUpdate)
    ));
    let wrong = BrainInput {
        schema_id: "wrong".to_owned(),
        ..runner_input("r", "target", &[1.0], 0)
    };
    assert!(matches!(
        first.evaluate_all(&wrong),
        Err(habitua::RunnerError::SchemaMismatch { .. })
    ));
}

#[test]
fn runner_rejects_non_finite_and_out_of_range_learning_weights() {
    let brain = brain_config(
        "one",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(Config {
            dimensions: 1,
            encoder_units: 32,
            k: 4,
            ..Config::default()
        }),
        1,
    );
    let mut runner = BrainRunner::with_builtins([brain]).unwrap();
    let mut evaluation = runner
        .evaluate_all(&runner_input("stream", "target", &[1.0], 0))
        .unwrap();
    let update = evaluation.brains[0].update.take().unwrap();
    assert!(matches!(
        runner.commit(update, f32::NAN),
        Err(habitua::RunnerError::Model {
            source: ModelError::InvalidLearningWeight,
            ..
        })
    ));

    let mut evaluation = runner
        .evaluate_all(&runner_input("stream", "target", &[1.0], 0))
        .unwrap();
    let update = evaluation.brains[0].update.take().unwrap();
    assert!(matches!(
        runner.commit(update, 1.1),
        Err(habitua::RunnerError::Model {
            source: ModelError::InvalidLearningWeight,
            ..
        })
    ));
}

#[test]
fn runner_save_load_preserves_brain_state_and_disabled_state() {
    let config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    };
    let brain = brain_config(
        "one",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(config),
        1,
    );
    let configs = vec![brain];
    let registry = ModelRegistry::with_builtins();
    let mut original = BrainRunner::new(configs.clone(), &registry).unwrap();
    let evaluation = original
        .evaluate_all(&runner_input("r", "target", &[1.0], 0))
        .unwrap();
    original.commit_all(evaluation).unwrap();
    original.set_enabled("one", false).unwrap();
    let mut bytes = Vec::new();
    original.save(&mut bytes).unwrap();
    let mut disabled_configs = configs.clone();
    disabled_configs[0].enabled = false;
    let restored =
        BrainRunner::load(&mut Cursor::new(bytes), &disabled_configs, &registry).unwrap();
    assert_eq!(restored.states(), original.states());
}

#[test]
fn runner_limits_unique_streams() {
    let config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    };
    let mut brain = brain_config(
        "one",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(config),
        1,
    );
    brain.resources = ResourcePolicy {
        max_streams: 1,
        ..ResourcePolicy::default()
    };
    let mut runner = BrainRunner::with_builtins([brain]).unwrap();
    let first = runner
        .evaluate_all(&runner_input("a", "target", &[1.0], 0))
        .unwrap();
    runner.commit_all(first).unwrap();
    assert!(matches!(
        runner.evaluate_all(&runner_input("b", "target", &[1.0], 1)),
        Err(habitua::RunnerError::ResourceLimit { .. })
    ));
}

#[test]
fn runner_commits_pending_updates_from_different_streams() {
    let config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    };
    let brain = brain_config(
        "fly",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(config),
        1,
    );
    let mut runner = BrainRunner::with_builtins([brain]).unwrap();
    let first = runner
        .evaluate_all(&runner_input("stream-a", "target", &[1.0], 0))
        .unwrap()
        .brains
        .into_iter()
        .next()
        .unwrap()
        .update
        .unwrap();
    let second = runner
        .evaluate_all(&runner_input("stream-b", "target", &[1.0], 1))
        .unwrap()
        .brains
        .into_iter()
        .next()
        .unwrap()
        .update
        .unwrap();
    runner.commit(first, 1.0).unwrap();
    runner.commit(second, 1.0).unwrap();
}

#[test]
fn repeated_normal_learning_streak_is_stream_local() {
    let config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        long_learning_policy: habitua::LongLearningPolicy::RepeatedNormal {
            consecutive_observations: 3,
        },
        ..Config::default()
    };
    let mut brain = brain_config(
        "repeated",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(config),
        1,
    );
    brain.learning.mode = LearningMode::RepeatedNormal {
        consecutive_observations: 3,
    };
    brain.learning.bootstrap = false;
    brain.reactions.threshold = 2.0;
    brain.reactions.kinds.clear();
    let mut runner = BrainRunner::with_builtins([brain]).unwrap();

    for (position, stream) in ["stream-a", "stream-b", "stream-a", "stream-b"]
        .into_iter()
        .enumerate()
    {
        let evaluation = runner
            .evaluate_all(&runner_input(stream, "target", &[1.0], position as u64))
            .unwrap();
        assert_eq!(evaluation.brains[0].recommended_learning_weight, 0.0);
        runner.commit_all(evaluation).unwrap();
    }

    let evaluation = runner
        .evaluate_all(&runner_input("stream-a", "target", &[1.0], 4))
        .unwrap();
    assert_eq!(evaluation.brains[0].recommended_learning_weight, 1.0);
}

#[test]
fn repeated_normal_learning_is_not_blocked_by_fly_novelty_reaction() {
    let config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        long_learning_policy: habitua::LongLearningPolicy::RepeatedNormal {
            consecutive_observations: 2,
        },
        ..Config::default()
    };
    let mut brain = brain_config(
        "repeated-fly",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(config),
        1,
    );
    brain.learning.mode = LearningMode::RepeatedNormal {
        consecutive_observations: 2,
    };
    brain.learning.bootstrap = false;
    brain.reactions.threshold = 0.5;
    let mut runner = BrainRunner::with_builtins([brain]).unwrap();

    let first = runner
        .evaluate_all(&runner_input("stream", "target", &[1.0], 0))
        .unwrap();
    assert_eq!(
        first.brains[0].report.learning_signal,
        LearningSignal::Normal
    );
    assert!(!first.brains[0].report.reactions.is_empty());
    assert_eq!(first.brains[0].recommended_learning_weight, 0.0);
    runner.commit_all(first).unwrap();

    let second = runner
        .evaluate_all(&runner_input("stream", "target", &[1.0], 1))
        .unwrap();
    assert_eq!(
        second.brains[0].report.learning_signal,
        LearningSignal::Normal
    );
    assert_eq!(second.brains[0].recommended_learning_weight, 1.0);
}

#[test]
fn repeated_normal_streak_is_committed_with_the_update() {
    let config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        long_learning_policy: habitua::LongLearningPolicy::RepeatedNormal {
            consecutive_observations: 2,
        },
        ..Config::default()
    };
    let mut brain = brain_config(
        "repeated-transaction",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(config),
        1,
    );
    brain.learning.mode = LearningMode::RepeatedNormal {
        consecutive_observations: 2,
    };
    brain.learning.bootstrap = false;
    brain.reactions.threshold = 2.0;
    let mut runner = BrainRunner::with_builtins([brain]).unwrap();

    let first = runner
        .evaluate_all(&runner_input("stream", "target", &[1.0], 0))
        .unwrap();
    assert_eq!(first.brains[0].recommended_learning_weight, 0.0);
    drop(first);

    let second = runner
        .evaluate_all(&runner_input("stream", "target", &[1.0], 1))
        .unwrap();
    assert_eq!(second.brains[0].recommended_learning_weight, 0.0);
    runner.commit_all(second).unwrap();

    let third = runner
        .evaluate_all(&runner_input("stream", "target", &[1.0], 2))
        .unwrap();
    assert_eq!(third.brains[0].recommended_learning_weight, 1.0);
}

#[test]
fn runner_filters_by_target_before_validating_schema() {
    let config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    };
    let mut screen = brain_config(
        "z-screen",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(config),
        1,
    );
    screen.input.schema_id = "screen-v1".to_owned();
    screen.input.target = "screen".to_owned();
    let numeric = brain_config(
        "a-latency",
        BrainMode::Active,
        "numeric",
        ModelSettings::Numeric {
            names: vec!["latency_ms".to_owned()],
            config: NumericDeviationConfig::default(),
        },
        1,
    );
    let mut numeric = numeric;
    numeric.input.schema_id = "latency-v1".to_owned();
    numeric.input.target = "latency".to_owned();
    let mut runner = BrainRunner::with_builtins([screen, numeric]).unwrap();
    let input = BrainInput {
        target: "screen".to_owned(),
        schema_id: "screen-v1".to_owned(),
        features: vec![1.0],
        ..runner_input("screen-record", "target", &[1.0], 0)
    };
    let evaluation = runner.evaluate_all(&input).unwrap();
    assert_eq!(evaluation.brains.len(), 2);
    assert_eq!(
        evaluation
            .brains
            .iter()
            .find(|brain| brain.brain == "z-screen")
            .unwrap()
            .report
            .readiness,
        Readiness::Evaluated
    );
    assert_eq!(
        evaluation
            .brains
            .iter()
            .find(|brain| brain.brain == "a-latency")
            .unwrap()
            .report
            .readiness,
        Readiness::InputMissing
    );
}

#[test]
fn numeric_rejects_impossible_baseline_requirement() {
    let config = NumericDeviationConfig {
        max_samples: 1,
        minimum_samples: 2,
        ..NumericDeviationConfig::default()
    };
    assert!(matches!(
        NumericDeviationModel::with_config(["value"], config),
        Err(NumericDeviationError::InvalidParameter {
            name: "minimum_samples"
        })
    ));
}

#[test]
fn generic_memory_receives_fractional_learning_weight() {
    let mut engine = Habituation::new(
        FixedEncoder,
        || Ok(WeightedMemory::default()),
        test_config(),
    )
    .unwrap();
    let first = engine
        .evaluate("weighted", &[0.0; 4], Timestamp::ZERO)
        .unwrap();
    engine.commit(first.update, 0.25).unwrap();
    let second = engine
        .evaluate("weighted", &[0.0; 4], Timestamp::from_secs(1))
        .unwrap();
    assert!((second.familiarity_short.unwrap() - 0.25).abs() < 1.0e-6);
}

#[test]
fn failure_brain_uses_structured_input_and_does_not_learn() {
    let mut brain = brain_config(
        "failure",
        BrainMode::Active,
        "failure",
        ModelSettings::Failure(FailureModelConfig::default()),
        1,
    );
    brain.reactions.kinds = vec![ReactionKind::Failure];
    brain.learning.mode = LearningMode::Manual;
    brain.learning.weight = 0.0;
    let mut runner = BrainRunner::with_builtins([brain]).unwrap();
    let input = BrainInput {
        failure: Some(FailureObservation {
            success: false,
            result_code: Some("timeout".to_owned()),
            error_type: Some("io".to_owned()),
            state: Some("running".to_owned()),
        }),
        features: Vec::new(),
        ..runner_input("request", "target", &[], 0)
    };
    let evaluation = runner.evaluate_all(&input).unwrap();
    assert_eq!(evaluation.brains[0].report.readiness, Readiness::Evaluated);
    assert_eq!(
        evaluation.brains[0].report.reactions[0].kind,
        ReactionKind::Failure
    );
    let commits = runner.commit_all(evaluation).unwrap();
    assert_eq!(commits[0].weight, 0.0);
    assert!(!commits[0].learned);
}

#[test]
fn nearest_neighbor_reports_distance_distribution() {
    let mut model = NearestNeighborModel::new(NearestNeighborConfig {
        dimensions: 2,
        ..NearestNeighborConfig::default()
    })
    .unwrap();
    let first = model
        .evaluate("screen", &[1.0, 0.0], Timestamp::ZERO)
        .unwrap();
    model.commit(first.update, 1.0).unwrap();
    let second = model
        .evaluate("screen", &[0.0, 1.0], Timestamp::from_secs(1))
        .unwrap();
    assert!(second.diagnostics.distance_mean.is_some());
    assert!(second.diagnostics.distance_median.is_some());
    assert!(second.diagnostics.distance_max.is_some());
}

#[test]
fn promotion_rejects_different_input_context() {
    let config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    };
    let shadow = brain_config(
        "shadow",
        BrainMode::Shadow,
        "fly",
        ModelSettings::Fly(config.clone()),
        1,
    );
    let mut active = brain_config(
        "active",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(config),
        1,
    );
    active.input.context = "other-context".to_owned();
    let mut runner = BrainRunner::with_builtins([shadow, active]).unwrap();
    assert!(matches!(
        runner.promote_shadow("shadow", "active"),
        Err(habitua::RunnerError::ForeignUpdate)
    ));
}

#[test]
fn runner_fingerprint_includes_fly_long_learning_policy() {
    let revisit = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        long_learning_policy: habitua::LongLearningPolicy::Revisit {
            interval: Duration::from_secs(10),
        },
        ..Config::default()
    };
    let repeated = Config {
        long_learning_policy: habitua::LongLearningPolicy::RepeatedNormal {
            consecutive_observations: 2,
        },
        ..revisit.clone()
    };
    let first = brain_config(
        "fly",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(revisit),
        1,
    );
    let second = brain_config(
        "fly",
        BrainMode::Active,
        "fly",
        ModelSettings::Fly(repeated),
        1,
    );
    assert_ne!(
        first.configuration_fingerprint(),
        second.configuration_fingerprint()
    );
}

#[test]
fn multiplier_is_same_fly_pattern_but_numeric_deviation() {
    let fly_config = Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    };
    let encoder = FlyHashEncoder::from_config(&fly_config).unwrap();
    assert_eq!(
        encoder.encode(&[1.0]).unwrap(),
        encoder.encode(&[16.0]).unwrap()
    );

    let numeric_config = NumericDeviationConfig {
        mad_floor: 0.1,
        minimum_samples: 2,
        direction: NumericDirection::Up,
        ..NumericDeviationConfig::default()
    };
    let mut numeric = NumericDeviationModel::with_config(["latency_ms"], numeric_config).unwrap();
    for timestamp in 0..2 {
        let evaluation = numeric
            .evaluate("latency", &[1.0], Timestamp::from_secs(timestamp))
            .unwrap();
        numeric.commit(evaluation.update, 1.0).unwrap();
    }
    let evaluation = numeric
        .evaluate("latency", &[16.0], Timestamp::from_secs(2))
        .unwrap();
    assert!(evaluation.diagnostics.deviations[0].z_plus > 10.0);
}

#[test]
fn repeated_failure_does_not_change_fly_novelty_or_failure_readiness() {
    let mut fly = FlyModel::with_config(Config {
        dimensions: 1,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    })
    .unwrap();
    let mut failure = FailureModel::new(FailureModelConfig::default()).unwrap();
    let features = [1.0];
    let first = fly
        .evaluate("operation", &features, Timestamp::ZERO)
        .unwrap();
    fly.commit(first.update, 1.0).unwrap();
    let repeated = fly
        .evaluate("operation", &features, Timestamp::from_secs(1))
        .unwrap();
    assert!(repeated.novelty_short.unwrap() < 1.0);

    for timestamp in 0..2 {
        let result = FailureObservation {
            success: false,
            result_code: Some("timeout".to_owned()),
            error_type: None,
            state: None,
        };
        let evaluation = failure
            .evaluate_input(
                "operation",
                ModelInput::Failure(&result),
                Timestamp::from_secs(timestamp),
            )
            .unwrap();
        assert_eq!(evaluation.readiness, Readiness::Evaluated);
        failure.commit(evaluation.update, 0.0).unwrap();
    }
}

fn brain_config(
    id: &str,
    mode: BrainMode,
    kind: &str,
    settings: ModelSettings,
    version: u32,
) -> BrainConfig {
    let seed = match &settings {
        ModelSettings::Fly(config) => Some(config.seed),
        ModelSettings::Numeric { .. }
        | ModelSettings::NearestNeighbor(_)
        | ModelSettings::Failure(_) => None,
        ModelSettings::Custom(_) => Some(7),
    };
    BrainConfig {
        id: id.to_owned(),
        role: id.to_owned(),
        enabled: true,
        mode,
        input: BrainInputSpec {
            schema_id: "features".to_owned(),
            schema_version: version,
            target: "target".to_owned(),
            context: "test".to_owned(),
        },
        model: ModelSpec {
            kind: kind.to_owned(),
            version,
            seed,
            settings,
        },
        learning: BrainLearningPolicy {
            mode: LearningMode::EveryEvaluation,
            weight: 1.0,
            require_evaluated: true,
            bootstrap: false,
            learn_on_reaction: false,
            frequency_limit: None,
        },
        reactions: ReactionPolicy {
            threshold: 0.8,
            kinds: vec![ReactionKind::Novel, ReactionKind::Deviation],
            calibration: None,
        },
        resources: ResourcePolicy::default(),
        state: StatePolicy::default(),
    }
}

fn runner_input(id: &str, target: &str, features: &[f32], seconds: u64) -> BrainInput {
    BrainInput {
        id: id.to_owned(),
        stream_id: id.to_owned(),
        target: target.to_owned(),
        context: "test".to_owned(),
        schema_id: "features".to_owned(),
        schema_version: 1,
        features: features.to_vec(),
        failure: None,
        available_at: Timestamp::from_secs(seconds),
        processed_at: None,
        position: seconds,
    }
}

proptest! {
    #[test]
    fn fly_values_stay_in_unit_interval(
        features in prop::collection::vec(-10.0_f32..10.0_f32, 4),
    ) {
        let mut engine = FlyModel::with_config(test_config()).unwrap();
        for step in 0..5 {
            let evaluation = engine.evaluate("property", &features, Timestamp::from_secs(step)).unwrap();
            assert_interval(evaluation.novelty_short);
            assert_interval(evaluation.novelty_long);
            assert_interval(evaluation.familiarity_short);
            assert_interval(evaluation.familiarity_long);
            engine.commit(evaluation.update, 1.0).unwrap();
        }
    }

    #[test]
    fn repeated_familiarity_is_monotonic(
        features in prop::collection::vec(-1.0_f32..1.0_f32, 4),
        repetitions in 2_usize..20,
    ) {
        let mut engine = FlyModel::with_config(test_config()).unwrap();
        let mut previous = 0.0;
        for _ in 0..repetitions {
            let evaluation = engine.evaluate("property", &features, Timestamp::ZERO).unwrap();
            prop_assert!(evaluation.familiarity_short.unwrap() >= previous);
            previous = evaluation.familiarity_short.unwrap();
            engine.commit(evaluation.update, 1.0).unwrap();
        }
    }
}

#[derive(Clone)]
struct FixedEncoder;

impl Encoder for FixedEncoder {
    fn encode(&self, features: &[f32]) -> Result<Pattern, EncodeError> {
        if features.len() != 4 {
            return Err(EncodeError::DimensionMismatch {
                expected: 4,
                actual: features.len(),
            });
        }
        Pattern::new(vec![0]).map_err(EncodeError::Pattern)
    }

    fn input_dimensions(&self) -> usize {
        4
    }

    fn fingerprint(&self) -> u64 {
        0x4649_5845_445f_454e
    }
}

#[derive(Default)]
struct CountingMemory {
    observations: u32,
    last_observed: Option<Timestamp>,
}

impl Memory for CountingMemory {
    type Update = Box<dyn FnOnce(&mut Self, f32)>;

    fn observe(
        &self,
        pattern: &Pattern,
        at: Timestamp,
    ) -> Result<MemoryTransaction<Self::Update>, MemoryError> {
        let familiarity = self.observations as f32 / 2.0;
        let observation = MemoryObservation {
            short_familiarity: familiarity,
            long_familiarity: familiarity,
            familiarity,
            novelty: 1.0 - familiarity,
            recency: self
                .last_observed
                .and_then(|last| at.checked_duration_since(last)),
            pattern_id: pattern.id(),
        };
        let next = self.observations.saturating_add(1);
        let update = Box::new(move |memory: &mut Self, _weight: f32| {
            memory.observations = next;
            memory.last_observed = Some(at);
        });
        Ok(MemoryTransaction::new(observation, update))
    }

    fn commit(&mut self, update: Self::Update) {
        update(self, 1.0);
    }

    fn commit_with_weight(&mut self, update: Self::Update, weight: f32) {
        update(self, weight);
    }

    fn last_seen(&self, _pattern: &Pattern) -> Option<Timestamp> {
        self.last_observed
    }

    fn fingerprint(&self) -> u64 {
        0x434f_554e_5449_4e47
    }

    fn forget(&mut self) {
        self.observations = 0;
        self.last_observed = None;
    }

    fn stats(&self) -> MemoryStats {
        MemoryStats {
            pattern_count: usize::from(self.observations > 0),
            remembered_unit_count: usize::from(self.observations > 0),
            short_familiarity: self.observations as f32 / 2.0,
            long_familiarity: self.observations as f32 / 2.0,
            last_observed: self.last_observed,
        }
    }
}

#[derive(Default)]
struct WeightedMemory {
    familiarity: f32,
    last_observed: Option<Timestamp>,
}

impl Memory for WeightedMemory {
    type Update = Box<dyn FnOnce(&mut Self, f32)>;

    fn observe(
        &self,
        pattern: &Pattern,
        at: Timestamp,
    ) -> Result<MemoryTransaction<Self::Update>, MemoryError> {
        let familiarity = self.familiarity;
        let recency = self
            .last_observed
            .and_then(|last| at.checked_duration_since(last));
        let update = Box::new(move |memory: &mut Self, weight: f32| {
            memory.familiarity = (memory.familiarity + weight).min(1.0);
            memory.last_observed = Some(at);
        });
        Ok(MemoryTransaction::new(
            MemoryObservation {
                short_familiarity: familiarity,
                long_familiarity: familiarity,
                familiarity,
                novelty: 1.0 - familiarity,
                recency,
                pattern_id: pattern.id(),
            },
            update,
        ))
    }

    fn commit(&mut self, update: Self::Update) {
        update(self, 1.0);
    }

    fn commit_with_weight(&mut self, update: Self::Update, weight: f32) {
        update(self, weight);
    }

    fn last_seen(&self, _pattern: &Pattern) -> Option<Timestamp> {
        self.last_observed
    }

    fn fingerprint(&self) -> u64 {
        0x5745_4947_4854_4544
    }

    fn forget(&mut self) {
        self.familiarity = 0.0;
        self.last_observed = None;
    }

    fn stats(&self) -> MemoryStats {
        MemoryStats {
            pattern_count: usize::from(self.last_observed.is_some()),
            remembered_unit_count: usize::from(self.last_observed.is_some()),
            short_familiarity: self.familiarity,
            long_familiarity: self.familiarity,
            last_observed: self.last_observed,
        }
    }
}
