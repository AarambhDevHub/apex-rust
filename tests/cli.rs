use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn inspect_cli_smoke() {
    let mut cmd = Command::cargo_bin("apex-rust").expect("binary exists");
    cmd.arg("inspect")
        .assert()
        .success()
        .stdout(contains("APEX-1 Rust Candle"));
}

#[test]
fn infer_random_cli_smoke() {
    let mut cmd = Command::cargo_bin("apex-rust").expect("binary exists");
    cmd.args([
        "infer",
        "--random",
        "--max-new-tokens",
        "1",
        "--temperature",
        "0",
    ])
    .assert()
    .success();
}

#[test]
fn benchmark_cli_smoke() {
    let mut cmd = Command::cargo_bin("apex-rust").expect("binary exists");
    cmd.args(["benchmark", "--seq-len", "4", "--repeats", "1"])
        .assert()
        .success()
        .stdout(contains("tokens_per_second"));
}
