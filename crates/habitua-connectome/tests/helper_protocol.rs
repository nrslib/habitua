use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use habitua_connectome::{
    RATE_HELPER_OBSERVATION_SCHEMA, RateGraph, RateHelperArtifact, RateHelperBrain,
    RateHelperCaseMemory, RateHelperResponseGroup, RateNeuronMetadata, RatePackManifest,
    RateParameters, RetinaMap, RetinaMapConfig,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const COOSENPAI_RATE_HELPER_ARGUMENTS: &[&str] =
    &["rate", "--max-in-flight", "4", "--timeout-ms", "5000"];

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn test_graph() -> RateGraph {
    RateGraph::new(
        vec![1, 2],
        vec![0, 0, 0],
        Vec::new(),
        vec![0, 0, 0],
        Vec::new(),
        vec![-1, 0],
        vec![
            RateNeuronMetadata {
                body_id: 1,
                type_name: Some("R1-R6".to_owned()),
                class: None,
                superclass: None,
                subclass: None,
                soma_side: Some("L".to_owned()),
                root_side: None,
                assigned_ol_hex1: Some(0.0),
                assigned_ol_hex2: Some(0.0),
                selected_neurotransmitter: Some("histamine".to_owned()),
                neurotransmitter_source: "test".to_owned(),
                source_sign: -1,
            },
            RateNeuronMetadata {
                body_id: 2,
                type_name: Some("DNp20".to_owned()),
                class: None,
                superclass: None,
                subclass: None,
                soma_side: Some("L".to_owned()),
                root_side: None,
                assigned_ol_hex1: None,
                assigned_ol_hex2: None,
                selected_neurotransmitter: None,
                neurotransmitter_source: "test".to_owned(),
                source_sign: 0,
            },
        ],
        BTreeMap::from([("R1-R6".to_owned(), vec![0]), ("DNp20".to_owned(), vec![1])]),
    )
    .expect("test graph")
}

fn test_artifact(root: &std::path::Path, steps: usize) -> (std::path::PathBuf, std::path::PathBuf) {
    let pack_path = root.join("pack");
    let graph = test_graph();
    graph
        .write_pack(
            &pack_path,
            &RatePackManifest {
                format_version: 0,
                dataset_version: "test".to_owned(),
                source_commit: "test".to_owned(),
                generated_at: "test".to_owned(),
                license: "test".to_owned(),
                neuron_count: 0,
                edge_count: 0,
                contact_count: 0,
                neurons_sha256: String::new(),
                outgoing_sha256: String::new(),
                incoming_sha256: String::new(),
                populations_sha256: String::new(),
                normalization: "test".to_owned(),
                sign_rule: "test".to_owned(),
                exclusion_reasons: BTreeMap::new(),
                sign_edge_counts: BTreeMap::new(),
                raw_edge_row_count: 0,
                retained_raw_edge_row_count: 0,
                connectome_row_group_count: 0,
                row_group_limit: None,
                external_sort_run_count: 0,
                conversion_elapsed_ms: 0.0,
                peak_rss_bytes: None,
                source_files: Vec::new(),
                intermediate_files: Vec::new(),
            },
        )
        .expect("test pack");
    let retina_map = RetinaMap::from_graph(&graph, RetinaMapConfig::default()).expect("retina map");
    let response_dimension = 5;
    let full_dimension = response_dimension + 4;
    let artifact = RateHelperArtifact {
        observation_schema: RATE_HELPER_OBSERVATION_SCHEMA.to_owned(),
        artifact_version: 1,
        pack_path: pack_path.to_string_lossy().into_owned(),
        pack_manifest_sha256: sha256_hex(&fs::read(pack_path.join("rate_manifest.json")).unwrap()),
        graph_fingerprint: graph.fingerprint(),
        retina_map,
        rate_steps_per_observation: steps,
        hmax: RateParameters::initial(&graph).hmax,
        response_projection_dimension: 1,
        history_recent_observations: 1,
        history_distance_threshold: 1.0,
        history_previous_absolute_difference_scale: 1.0,
        history_elapsed_time_scale_seconds: 60.0,
        history_max_age_seconds: 300.0,
        zero_response_epsilon: 1.0e-6,
        response_groups: vec![RateHelperResponseGroup {
            name: "dn".to_owned(),
            neuron_indices: vec![1],
            scale: 1.0,
        }],
        brains: vec![
            RateHelperBrain {
                id: "brain-change".to_owned(),
                role: "change".to_owned(),
                feature_dimension: response_dimension,
                standardization_mean: vec![0.0; full_dimension],
                standardization_scale: vec![1.0; full_dimension],
                weights: vec![0.0; full_dimension],
                bias: 0.0,
                reaction_threshold: 0.5,
            },
            RateHelperBrain {
                id: "brain-no-change".to_owned(),
                role: "no_change".to_owned(),
                feature_dimension: response_dimension,
                standardization_mean: vec![0.0; full_dimension],
                standardization_scale: vec![1.0; full_dimension],
                weights: vec![0.0; full_dimension],
                bias: 0.0,
                reaction_threshold: 0.5,
            },
        ],
        consensus_margin: 0.1,
        case_memory: RateHelperCaseMemory {
            match_policy: "exact_image_sha256".to_owned(),
            time_constant_seconds: 600.0,
            logit_scale: 8.0,
            max_cases: 16,
        },
    };
    let artifact_path = root.join("artifact.json");
    artifact.write(&artifact_path).expect("test artifact");
    let image_path = root.join("frame.ppm");
    let mut image_bytes = b"P3\n256 144\n255\n".to_vec();
    for _ in 0..(256 * 144) {
        image_bytes.extend_from_slice(b"128 128 128\n");
    }
    fs::write(&image_path, image_bytes).expect("test image write");
    (artifact_path, image_path)
}

fn request(writer: &mut impl Write, reader: &mut impl BufRead, value: Value) -> Value {
    serde_json::to_writer(&mut *writer, &value).expect("request JSON");
    writer.write_all(b"\n").expect("request newline");
    writer.flush().expect("request flush");
    let mut line = String::new();
    reader.read_line(&mut line).expect("response line");
    serde_json::from_str(line.trim()).expect("response JSON")
}

fn evaluate_request(id: &str, input_id: &str, image_path: &std::path::Path, at_ms: u64) -> Value {
    json!({
        "v": 1,
        "id": id,
        "session": "test-session",
        "op": "evaluate",
        "params": {
            "input_id": input_id,
            "image_path": image_path,
            "stream_id": "screen/main",
            "at_ms": at_ms,
            "timestamp_s": at_ms as f64 / 1000.0,
            "feature_schema": RATE_HELPER_OBSERVATION_SCHEMA,
            "features": [],
            "position": 0,
            "shadow": true,
            "observation": {
                "frame_id": input_id,
                "at_ms": at_ms,
                "app": "Editor",
                "kind": "visual",
                "target": "fullscreen",
                "image_path": image_path,
            },
            "context": {"frames": [], "audio": []}
        }
    })
}

#[test]
fn resident_helper_uses_contract_envelope_feedback_and_shutdown() {
    let root = std::env::current_dir()
        .expect("current directory")
        .join("target")
        .join(format!(
            "connectome-helper-protocol-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
    fs::create_dir_all(&root).expect("test directory");
    let (artifact_path, image_path) = test_artifact(&root, 1);
    let state_path = root.join("state.json");
    let binary =
        std::env::var("CARGO_BIN_EXE_connectome-helper").expect("connectome-helper binary path");
    let mut child = Command::new(&binary)
        .args(COOSENPAI_RATE_HELPER_ARGUMENTS)
        .env("HABITUA_RATE_ARTIFACT", &artifact_path)
        .env("HABITUA_RATE_STATE", &state_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start helper with CooSenpAI arguments and environment");
    let mut writer = child.stdin.take().expect("helper stdin");
    let stdout = child.stdout.take().expect("helper stdout");
    let mut reader = BufReader::new(stdout);

    let health = request(
        &mut writer,
        &mut reader,
        json!({"v":1,"id":"health","session":"test-session","op":"health","params":{}}),
    );
    assert_eq!(health["ok"], true);
    assert_eq!(health["result"]["resident"], true);

    let first = request(
        &mut writer,
        &mut reader,
        evaluate_request("observation-1", "frame-1", &image_path, 1_000),
    );
    assert_eq!(first["ok"], true);
    assert_eq!(first["result"]["readiness"], "insufficient-history");
    assert_eq!(first["result"]["action"], "hold");
    assert_eq!(first["result"]["hold_reason"], "insufficient-history");

    let replay = request(
        &mut writer,
        &mut reader,
        evaluate_request("observation-1-replay", "frame-1", &image_path, 1_000),
    );
    assert_eq!(replay["ok"], true);
    assert_eq!(replay["result"]["replayed"], true);
    let persisted_after_replay: Value =
        serde_json::from_slice(&fs::read(&state_path).expect("state after replay"))
            .expect("persisted state JSON");
    assert_eq!(
        persisted_after_replay["histories"]["screen/main"]
            .as_array()
            .expect("screen history")
            .len(),
        1
    );
    assert_eq!(
        persisted_after_replay["observation_ledger"]["frame-1"]["input_id"],
        "frame-1"
    );
    assert!(
        persisted_after_replay["observation_ledger"]["frame-1"]["digest"]
            .as_str()
            .is_some_and(|digest| !digest.is_empty())
    );
    let mut changed_shadow =
        evaluate_request("observation-shadow-mismatch", "frame-1", &image_path, 1_000);
    changed_shadow["params"]["shadow"] = json!(false);
    let changed_shadow_response = request(&mut writer, &mut reader, changed_shadow);
    assert_eq!(changed_shadow_response["ok"], false);
    assert!(
        changed_shadow_response["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("canonical observation digest"))
    );

    let mut missing_visual_app = evaluate_request(
        "observation-missing-app",
        "frame-missing-app",
        &image_path,
        1_250,
    );
    missing_visual_app["params"]["observation"]["app"] = Value::Null;
    let missing_visual_app_response = request(&mut writer, &mut reader, missing_visual_app);
    assert_eq!(missing_visual_app_response["ok"], true);
    assert_eq!(
        missing_visual_app_response["result"]["readiness"],
        "input-insufficient"
    );
    assert_eq!(missing_visual_app_response["result"]["action"], "hold");
    let missing_visual_app_replay = request(&mut writer, &mut reader, {
        let mut value = evaluate_request(
            "observation-missing-app-replay",
            "frame-missing-app",
            &image_path,
            1_250,
        );
        value["params"]["observation"]["app"] = Value::Null;
        value
    });
    assert_eq!(missing_visual_app_replay["ok"], true);
    assert_eq!(missing_visual_app_replay["result"]["replayed"], true);
    let persisted_after_missing_app: Value =
        serde_json::from_slice(&fs::read(&state_path).expect("state after missing app"))
            .expect("persisted state JSON");
    assert_eq!(
        persisted_after_missing_app["histories"]["screen/main"]
            .as_array()
            .expect("screen history")
            .len(),
        1
    );

    let throttled = request(
        &mut writer,
        &mut reader,
        json!({
            "v": 1,
            "id": "observation-throttled",
            "session": "test-session",
            "op": "evaluate",
            "params": {
                "input_id": "frame-throttled",
                "image_path": image_path,
                "stream_id": "screen/main",
                "at_ms": 1_500,
                "timestamp_s": 1.5,
                "feature_schema": RATE_HELPER_OBSERVATION_SCHEMA,
                "features": [],
                "position": 1,
                "shadow": true,
                "repeat_index": 1,
                "throttle_every": 2,
                "observation": {
                    "frame_id": "frame-throttled",
                    "at_ms": 1_500,
                    "app": "Editor",
                    "kind": "visual",
                    "target": "fullscreen",
                    "image_path": image_path
                }
            }
        }),
    );
    assert_eq!(throttled["result"]["readiness"], "insufficient-history");
    assert_eq!(throttled["result"]["action"], "hold");
    assert_eq!(throttled["result"]["skipped"], true);

    let audio = request(
        &mut writer,
        &mut reader,
        json!({
            "v": 1,
            "id": "audio-1",
            "session": "test-session",
            "op": "evaluate",
            "params": {
                "input_id": "audio-1",
                "image_path": "",
                "stream_id": "audio/main",
                "at_ms": 1_750,
                "timestamp_s": 1.75,
                "feature_schema": RATE_HELPER_OBSERVATION_SCHEMA,
                "features": [],
                "observation": {
                    "frame_id": "audio-1",
                    "at_ms": 1_750,
                    "app": null,
                    "kind": "audio",
                    "source": "microphone",
                    "text": "test input"
                }
            }
        }),
    );
    assert_eq!(audio["ok"], true);
    assert_eq!(audio["result"]["readiness"], "unsupported-input");
    assert_eq!(audio["result"]["action"], "hold");
    let audio_replay = request(
        &mut writer,
        &mut reader,
        json!({
            "v": 1,
            "id": "audio-replay",
            "session": "test-session",
            "op": "evaluate",
            "params": {
                "input_id": "audio-1",
                "image_path": "",
                "stream_id": "audio/main",
                "at_ms": 1_750,
                "timestamp_s": 1.75,
                "feature_schema": RATE_HELPER_OBSERVATION_SCHEMA,
                "features": [],
                "observation": {
                    "frame_id": "audio-1",
                    "at_ms": 1_750,
                    "app": null,
                    "kind": "audio",
                    "source": "microphone",
                    "text": "test input"
                }
            }
        }),
    );
    assert_eq!(audio_replay["ok"], true);
    assert_eq!(audio_replay["result"]["replayed"], true);

    let malformed = request(
        &mut writer,
        &mut reader,
        json!({
            "v": 1,
            "id": "malformed",
            "session": "test-session",
            "op": "evaluate",
            "params": {
                "input_id": "bad",
                "image_path": image_path,
                "stream_id": "screen/main",
                "at_ms": 2_000,
                "timestamp_s": 2.0,
                "feature_schema": RATE_HELPER_OBSERVATION_SCHEMA,
                "features": [],
                "observation": {
                    "frame_id": "bad",
                    "at_ms": 2_000,
                    "app": "Editor",
                    "kind": "visual",
                    "target": "fullscreen",
                    "image_path": image_path
                },
                "unexpected": true
            }
        }),
    );
    assert_eq!(malformed["ok"], false);

    let parse_error = request(
        &mut writer,
        &mut reader,
        json!({
            "v": 1,
            "id": "parse-error",
            "session": "test-session",
            "op": "evaluate",
            "params": {},
            "unexpected": true
        }),
    );
    assert_eq!(parse_error["ok"], false);
    assert_eq!(parse_error["id"], "parse-error");
    assert_eq!(parse_error["session"], "test-session");

    let mut mismatched_identity = evaluate_request(
        "observation-mismatched-identity",
        "frame-mismatched-identity",
        &image_path,
        2_500,
    );
    mismatched_identity["params"]["observation"]["frame_id"] = json!("other-frame");
    let mismatched_identity_response = request(&mut writer, &mut reader, mismatched_identity);
    assert_eq!(mismatched_identity_response["ok"], false);

    let mut mismatched_timestamp = evaluate_request(
        "observation-mismatched-timestamp",
        "frame-mismatched-timestamp",
        &image_path,
        2_750,
    );
    mismatched_timestamp["params"]["timestamp_s"] = json!(2.0);
    let mismatched_timestamp_response = request(&mut writer, &mut reader, mismatched_timestamp);
    assert_eq!(mismatched_timestamp_response["ok"], false);

    let mut malformed_context = evaluate_request(
        "observation-malformed-context",
        "frame-malformed-context",
        &image_path,
        3_000,
    );
    malformed_context["params"]["context"] = json!({"frames": {"frame_id": "not-an-array"}});
    let malformed_context_response = request(&mut writer, &mut reader, malformed_context);
    assert_eq!(malformed_context_response["ok"], false);

    let invalid_cancel_strength = request(
        &mut writer,
        &mut reader,
        json!({
            "v": 1,
            "id": "feed-invalid-cancel-strength",
            "session": "test-session",
            "op": "feed",
            "params": {
                "event_id": "event-invalid-cancel-strength",
                "input_id": "frame-1",
                "sign": "reward",
                "strength": -1.0,
                "event_time_s": 1.0,
                "received_at_s": 2.0,
                "revision": 1,
                "cancelled": true,
                "source": "explicit"
            }
        }),
    );
    assert_eq!(invalid_cancel_strength["ok"], false);
    assert_eq!(invalid_cancel_strength["error"]["code"], "request_error");

    let feed = request(
        &mut writer,
        &mut reader,
        json!({
            "v": 1,
            "id": "feed-1",
            "session": "test-session",
            "op": "feed",
            "params": {
                "event_id": "event-1",
                "input_id": "frame-1",
                "sign": "reward",
                "strength": 1.0,
                "event_time_s": 1.0,
                "received_at_s": 2.0,
                "revision": 1,
                "cancelled": false,
                "source": "explicit"
            }
        }),
    );
    assert_eq!(feed["ok"], true);
    assert_eq!(feed["result"]["applied"], true);
    assert_eq!(feed["result"]["source"], "explicit");
    assert!(feed["result"]["applied_at_s"].as_f64().is_some());

    let shutdown = request(
        &mut writer,
        &mut reader,
        json!({"v":1,"id":"shutdown","session":"test-session","op":"shutdown","params":{}}),
    );
    assert_eq!(shutdown["ok"], true);
    drop(writer);
    assert!(child.wait().expect("helper wait").success());
    assert!(state_path.exists());

    let mut restarted = Command::new(&binary)
        .args(COOSENPAI_RATE_HELPER_ARGUMENTS)
        .env("HABITUA_RATE_ARTIFACT", &artifact_path)
        .env("HABITUA_RATE_STATE", &state_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("restart helper with CooSenpAI arguments and environment");
    let mut restarted_writer = restarted.stdin.take().expect("restarted helper stdin");
    let restarted_stdout = restarted.stdout.take().expect("restarted helper stdout");
    let mut restarted_reader = BufReader::new(restarted_stdout);
    let restarted_health = request(
        &mut restarted_writer,
        &mut restarted_reader,
        json!({"v":1,"id":"health-after-restart","session":"test-session","op":"health","params":{}}),
    );
    assert_eq!(restarted_health["ok"], true);
    assert_eq!(restarted_health["result"]["case_count"], 1);
    let correction = request(
        &mut restarted_writer,
        &mut restarted_reader,
        json!({
            "v": 1,
            "id": "feed-correction",
            "session": "test-session",
            "op": "feed",
            "params": {
                "event_id": "event-1",
                "input_id": "frame-1",
                "sign": "punish",
                "strength": 0.5,
                "event_time_s": 1.0,
                "received_at_s": 3.0,
                "revision": 2,
                "cancelled": false,
                "source": "automatic"
            }
        }),
    );
    assert_eq!(correction["ok"], true);
    assert_eq!(correction["result"]["applied"], true);
    assert_eq!(correction["result"]["source"], "automatic");
    let cancellation = request(
        &mut restarted_writer,
        &mut restarted_reader,
        json!({
            "v": 1,
            "id": "feed-cancel",
            "session": "test-session",
            "op": "feed",
            "params": {
                "event_id": "event-1",
                "input_id": "frame-1",
                "sign": "punish",
                "strength": 0.0,
                "event_time_s": 1.0,
                "received_at_s": 4.0,
                "revision": 3,
                "cancelled": true,
                "source": "explicit"
            }
        }),
    );
    assert_eq!(cancellation["ok"], true);
    assert_eq!(cancellation["result"]["applied"], true);
    assert_eq!(cancellation["result"]["case_event_count"], 0);
    let restarted_shutdown = request(
        &mut restarted_writer,
        &mut restarted_reader,
        json!({"v":1,"id":"shutdown-after-restart","session":"test-session","op":"shutdown","params":{}}),
    );
    assert_eq!(restarted_shutdown["ok"], true);
    drop(restarted_writer);
    assert!(restarted.wait().expect("restarted helper wait").success());
    fs::remove_dir_all(root).expect("remove test directory");
}

#[test]
fn resident_helper_reports_timeout_for_a_slow_request() {
    let root = std::env::current_dir()
        .expect("current directory")
        .join("target")
        .join(format!(
            "connectome-helper-timeout-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
    fs::create_dir_all(&root).expect("test directory");
    let (artifact_path, image_path) = test_artifact(&root, 1_000_000);
    let state_path = root.join("state.json");
    let binary = std::env::var("CARGO_BIN_EXE_connectome-helper")
        .or_else(|_| std::env::var("CARGO_BIN_EXE_connectome_helper"))
        .expect("connectome-helper binary path");
    let mut child = Command::new(&binary)
        .args(["rate", "--timeout-ms", "1"])
        .env("HABITUA_RATE_ARTIFACT", &artifact_path)
        .env("HABITUA_RATE_STATE", &state_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start helper for timeout test");
    let mut writer = child.stdin.take().expect("helper stdin");
    let stdout = child.stdout.take().expect("helper stdout");
    let mut reader = BufReader::new(stdout);
    let response = request(
        &mut writer,
        &mut reader,
        evaluate_request("slow-observation", "slow-frame", &image_path, 1_000),
    );
    assert_eq!(response["ok"], false);
    assert_eq!(response["error"]["code"], "timeout");
    let shutdown = request(
        &mut writer,
        &mut reader,
        json!({"v":1,"id":"shutdown","session":"test-session","op":"shutdown","params":{}}),
    );
    assert_eq!(shutdown["ok"], true);
    drop(writer);
    assert!(child.wait().expect("timeout helper wait").success());
    assert!(!state_path.exists());
    fs::remove_dir_all(root).expect("remove test directory");
}
