use std::io::{self, BufRead};

use habitua::{Config, FlyModel, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct InputLog {
    stream_id: String,
    timestamp_ms: u64,
    features: Vec<f32>,
}

#[derive(Serialize)]
struct OutputLog {
    stream_id: String,
    novelty_short: Option<f32>,
    novelty_long: Option<f32>,
    familiarity_short: Option<f32>,
    familiarity_long: Option<f32>,
    recency_ms: Option<u128>,
    pattern_id: Option<u64>,
    input_strength: f32,
    profile_deviation: f32,
    readiness: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        dimensions: 4,
        encoder_units: 32,
        k: 4,
        ..Config::default()
    };
    let mut engine = FlyModel::with_config(config)?;

    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let input: InputLog = serde_json::from_str(&line)?;
        let evaluation = engine.evaluate(
            &input.stream_id,
            &input.features,
            Timestamp::from_millis(input.timestamp_ms),
        )?;
        let output = OutputLog {
            stream_id: input.stream_id,
            novelty_short: evaluation.novelty_short,
            novelty_long: evaluation.novelty_long,
            familiarity_short: evaluation.familiarity_short,
            familiarity_long: evaluation.familiarity_long,
            recency_ms: evaluation.recency.map(|duration| duration.as_millis()),
            pattern_id: evaluation.pattern_id,
            input_strength: evaluation.diagnostics.input_strength,
            profile_deviation: evaluation.diagnostics.profile_deviation,
            readiness: format!("{:?}", evaluation.readiness),
        };
        engine.commit(evaluation.update, 1.0)?;
        println!("{}", serde_json::to_string(&output)?);
    }
    Ok(())
}
