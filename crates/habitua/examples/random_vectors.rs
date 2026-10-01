use habitua::{Config, FlyModel, Timestamp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        dimensions: 8,
        encoder_units: 64,
        k: 8,
        ..Config::default()
    };
    let mut engine = FlyModel::with_config(config)?;
    let mut random = SplitMix64::new(7);

    for index in 0..20_u64 {
        let features: Vec<f32> = (0..8).map(|_| random.next_unit() * 2.0 - 1.0).collect();
        let evaluation =
            engine.evaluate("random", &features, Timestamp::from_millis(index * 1_000))?;
        println!(
            "at={} novelty_short={:?} novelty_long={:?} profile_deviation={:.3} pattern_id={:?}",
            index,
            evaluation.novelty_short,
            evaluation.novelty_long,
            evaluation.diagnostics.profile_deviation,
            evaluation.pattern_id
        );
        engine.commit(evaluation.update, 1.0)?;
    }
    Ok(())
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let bits = value ^ (value >> 31);
        (bits >> 11) as f32 / (u64::MAX >> 11) as f32
    }
}
