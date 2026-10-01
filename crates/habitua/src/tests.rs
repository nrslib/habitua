use std::time::Duration;

use super::{Config, DecayingMemory, Memory, Pattern, Timestamp};

#[test]
fn pattern_indices_are_canonicalized_before_id_generation() {
    let first = Pattern::new(vec![4, 1, 7]).unwrap();
    let second = Pattern::new(vec![7, 4, 1]).unwrap();

    assert_eq!(first.units(), &[1, 4, 7]);
    assert_eq!(first, second);
    assert_eq!(first.id(), second.id());
}

#[test]
fn empty_decaying_memory_has_no_decay_state() {
    let config = Config {
        short_time_constant: std::time::Duration::from_secs(1),
        long_time_constant: std::time::Duration::from_secs(10),
        ..Config::default()
    };
    let memory = DecayingMemory::new(&config).unwrap();
    let stats = memory.stats();

    assert_eq!(stats.pattern_count, 0);
    assert_eq!(stats.remembered_unit_count, 0);
    assert_eq!(stats.short_familiarity, 0.0);
    assert_eq!(stats.long_familiarity, 0.0);
    assert_eq!(stats.last_observed, None);
    assert_eq!(Timestamp::ZERO.as_duration(), Duration::ZERO);
}
