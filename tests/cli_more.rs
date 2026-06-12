use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn inspect_json_cli_smoke() {
    let mut cmd = Command::cargo_bin("apex-rust").expect("binary exists");
    cmd.args(["inspect", "--format", "json"])
        .assert()
        .success()
        .stdout(contains("\"model\""))
        .stdout(contains("\"parameters\""));
}

#[test]
fn tokenizer_train_cli_writes_json() -> apex_rust::Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("text.txt");
    let output = dir.path().join("tokenizer.json");
    std::fs::write(&input, "hello rust\nhello candle\n")?;
    let mut cmd = Command::cargo_bin("apex-rust").expect("binary exists");
    cmd.args([
        "tokenizer",
        "train",
        "--input",
        input.to_str().unwrap_or_default(),
        "--output",
        output.to_str().unwrap_or_default(),
        "--vocab-size",
        "128",
    ])
    .assert()
    .success()
    .stdout(contains("wrote tokenizer"));
    assert!(output.exists());
    Ok(())
}

#[test]
fn pretrain_dry_run_writes_checkpoint_artifacts() -> apex_rust::Result<()> {
    let dir = tempfile::tempdir()?;
    let output_dir = dir.path().join("run");
    let mut cmd = Command::cargo_bin("apex-rust").expect("binary exists");
    cmd.args([
        "train",
        "pretrain",
        "--dry-run",
        "--steps",
        "1",
        "--output-dir",
        output_dir.to_str().unwrap_or_default(),
    ])
    .assert()
    .success()
    .stdout(contains("wrote training artifacts"));
    assert!(output_dir.join("metadata.json").exists());
    assert!(output_dir.join("model.safetensors").exists());
    assert!(output_dir.join("config.yaml").exists());
    Ok(())
}
