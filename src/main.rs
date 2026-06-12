use std::fs;
use std::path::{Path, PathBuf};

use apex_rust::config::{
    get_tiny_adapter_dpo_config, get_tiny_config, get_tiny_lora_config, ApexConfig, PeftMethod,
};
use apex_rust::data::{read_preference_jsonl, read_pretrain_jsonl, read_sft_jsonl};
use apex_rust::eval;
use apex_rust::generation::{ApexGenerator, GenerationConfig};
use apex_rust::model::ApexModel;
use apex_rust::tokenizer::{train_bpe_tokenizer, ApexTokenizer, ChatMessage};
use apex_rust::train;
use apex_rust::utils::{architecture_text, estimate_flops, ModelInspection};
use apex_rust::Result;
use candle_core::Device;
use clap::{Args, Parser, Subcommand, ValueEnum};
use tracing::{info, Level};

#[derive(Parser, Debug)]
#[command(name = "apex-rust")]
#[command(about = "Rust/Candle language and vision model toolkit", version)]
struct Cli {
    #[arg(long, global = true, default_value = "info")]
    log_level: String,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Train {
        #[command(subcommand)]
        command: TrainCommand,
    },
    Infer(InferArgs),
    Inspect(InspectArgs),
    Benchmark(BenchmarkArgs),
    MergeAdapter(MergeAdapterArgs),
    Tokenizer {
        #[command(subcommand)]
        command: TokenizerCommand,
    },
}

#[derive(Subcommand, Debug)]
enum TrainCommand {
    Pretrain(TrainArgs),
    Sft(TrainArgs),
    PeftSft(TrainArgs),
    AdapterDpo(TrainArgs),
}

#[derive(Args, Debug, Clone)]
struct TrainArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    data: Option<PathBuf>,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "checkpoints/apex-rust")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    steps: usize,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[derive(Args, Debug)]
struct InferArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = false)]
    random: bool,
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = false)]
    thinking: bool,
}

#[derive(Args, Debug)]
struct InspectArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = InspectFormat::Text)]
    format: InspectFormat,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InspectFormat {
    Text,
    Json,
}

#[derive(Args, Debug)]
struct BenchmarkArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 16)]
    seq_len: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
}

#[derive(Args, Debug)]
struct MergeAdapterArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = "checkpoints/apex-rust/merged")]
    output_dir: PathBuf,
}

#[derive(Subcommand, Debug)]
enum TokenizerCommand {
    Train(TokenizerTrainArgs),
}

#[derive(Args, Debug)]
struct TokenizerTrainArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 32_000)]
    vocab_size: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);
    match cli.command {
        Commands::Train { command } => run_train(command),
        Commands::Infer(args) => run_infer(args),
        Commands::Inspect(args) => run_inspect(args),
        Commands::Benchmark(args) => run_benchmark(args),
        Commands::MergeAdapter(args) => run_merge_adapter(args),
        Commands::Tokenizer { command } => match command {
            TokenizerCommand::Train(args) => run_tokenizer_train(args),
        },
    }
}

fn init_tracing(level: &str) {
    let level = match level {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };
    let _ = tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .try_init();
}

fn run_train(command: TrainCommand) -> Result<()> {
    match command {
        TrainCommand::Pretrain(args) => run_pretrain(args),
        TrainCommand::Sft(args) => run_sft(args, false),
        TrainCommand::PeftSft(args) => run_sft(args, true),
        TrainCommand::AdapterDpo(args) => run_adapter_dpo(args),
    }
}

fn run_pretrain(args: TrainArgs) -> Result<()> {
    let cfg = load_config_or(args.config.as_deref(), get_tiny_config)?;
    let tokenizer = ApexTokenizer::new(args.tokenizer.as_deref())?;
    let mut model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let samples = if let Some(path) = args.data.as_deref() {
        read_pretrain_jsonl(path, &tokenizer, cfg.training.seq_len)?
            .into_iter()
            .map(|s| s.input_ids)
            .collect::<Vec<_>>()
    } else {
        vec![synthetic_tokens(cfg.training.seq_len, cfg.model.vocab_size)]
    };
    let mut last_loss = 0.0;
    for step in 0..args.steps.max(1) {
        let batch = samples
            .get(step % samples.len().max(1))
            .cloned()
            .unwrap_or_else(|| synthetic_tokens(cfg.training.seq_len, cfg.model.vocab_size));
        let metrics = train::dry_run_pretrain_step(&mut model, &[batch])?;
        last_loss = metrics.loss_total;
        info!(
            "pretrain step={} loss={:.4} valid_tokens={}",
            step, metrics.loss_total, metrics.valid_tokens
        );
        if args.dry_run {
            break;
        }
    }
    write_train_outputs(&args.output_dir, &cfg, last_loss, &model)
}

fn run_sft(args: TrainArgs, peft: bool) -> Result<()> {
    let default_cfg = if peft {
        get_tiny_lora_config
    } else {
        get_tiny_config
    };
    let cfg = load_config_or(args.config.as_deref(), default_cfg)?;
    let tokenizer = ApexTokenizer::new(args.tokenizer.as_deref())?;
    let mut model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let samples = if let Some(path) = args.data.as_deref() {
        read_sft_jsonl(path, &tokenizer, cfg.training.seq_len)?
    } else {
        let ids = synthetic_tokens(cfg.training.seq_len, cfg.model.vocab_size);
        let types = vec![2u8; ids.len()];
        vec![apex_rust::data::SftSample {
            input_ids: ids,
            token_types: types,
        }]
    };
    let mut last_loss = 0.0;
    for step in 0..args.steps.max(1) {
        let sample = &samples[step % samples.len()];
        let output = model.forward(
            std::slice::from_ref(&sample.input_ids),
            None,
            0,
            None,
            false,
        )?;
        let metrics = train::compute_sft_loss(
            &output.logits,
            std::slice::from_ref(&sample.input_ids),
            std::slice::from_ref(&sample.token_types),
        )?;
        last_loss = metrics.loss_total;
        info!(
            "sft step={} peft={} loss={:.4} valid_tokens={}",
            step, peft, metrics.loss_total, metrics.valid_tokens
        );
        if args.dry_run {
            break;
        }
    }
    write_train_outputs(&args.output_dir, &cfg, last_loss, &model)
}

fn run_adapter_dpo(args: TrainArgs) -> Result<()> {
    let cfg = load_config_or(args.config.as_deref(), || {
        get_tiny_adapter_dpo_config(PeftMethod::Dora)
    })?;
    let tokenizer = ApexTokenizer::new(args.tokenizer.as_deref())?;
    let mut policy = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let mut reference = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let samples = if let Some(path) = args.data.as_deref() {
        read_preference_jsonl(
            path,
            &tokenizer,
            cfg.adapter_dpo.max_prompt_len,
            cfg.adapter_dpo.max_response_len,
        )?
    } else {
        vec![apex_rust::data::format_preference_example(
            &tokenizer,
            "Explain APEX briefly.",
            "APEX is an educational transformer model.",
            "I cannot answer.",
            cfg.adapter_dpo.max_prompt_len,
            cfg.adapter_dpo.max_response_len,
            true,
        )?]
    };
    let mut last_loss = 0.0;
    for step in 0..args.steps.max(1) {
        let sample = &samples[step % samples.len()];
        let metrics = apex_rust::alignment::adapter_dpo_step(
            &mut policy,
            Some(&mut reference),
            sample,
            cfg.adapter_dpo.beta,
            cfg.adapter_dpo.label_smoothing,
            cfg.adapter_dpo.reference_free,
            cfg.adapter_dpo.length_normalize,
        )?;
        last_loss = metrics.loss;
        info!(
            "adapter-dpo step={} loss={:.4} margin={:.4} acc={:.2}",
            step, metrics.loss, metrics.reward_margin, metrics.accuracy
        );
        if args.dry_run {
            break;
        }
    }
    write_train_outputs(&args.output_dir, &cfg, last_loss, &policy)
}

fn run_infer(args: InferArgs) -> Result<()> {
    let cfg = load_config_or(args.config.as_deref(), get_tiny_config)?;
    let tokenizer = ApexTokenizer::new(args.tokenizer.as_deref())?;
    let mut model = ApexModel::new(cfg, Device::Cpu)?;
    let input = if args.random {
        synthetic_tokens(8, model.config.model.vocab_size)
    } else {
        let prompt = args.prompt.unwrap_or_else(|| "Hello".to_string());
        tokenizer.encode_chat(
            &[ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            true,
            args.thinking,
        )?
    };
    let gen_cfg = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        enable_thinking: args.thinking,
        eos_token_id: tokenizer.eos_token_id(),
        pad_token_id: tokenizer.pad_token_id(),
        thinking_start_id: tokenizer.thinking_start_id(),
        thinking_end_id: tokenizer.thinking_end_id(),
        ..GenerationConfig::default()
    };
    let mut generator = ApexGenerator::new(&mut model, gen_cfg);
    let out = generator.generate(input, 0)?;
    let text = tokenizer.decode(&out.token_ids, true)?;
    println!("{text}");
    eprintln!(
        "tokens={} thinking_tokens={} finished={}",
        out.total_tokens, out.thinking_tokens, out.finished
    );
    Ok(())
}

fn run_inspect(args: InspectArgs) -> Result<()> {
    let cfg = load_config_or(args.config.as_deref(), get_tiny_config)?;
    let model = ApexModel::new(cfg, Device::Cpu)?;
    match args.format {
        InspectFormat::Text => {
            println!("{}", architecture_text(&model));
            println!(
                "parameters: total={} active={} trainable={}",
                model.total_parameters(),
                model.active_parameters(),
                model.trainable_parameters()
            );
        }
        InspectFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&ModelInspection::from_model(&model))?
            );
        }
    }
    Ok(())
}

fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    let cfg = load_config_or(args.config.as_deref(), get_tiny_config)?;
    let mut model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let batch = (0..args.batch_size)
        .map(|_| synthetic_tokens(args.seq_len, cfg.model.vocab_size))
        .collect::<Vec<_>>();
    let result = eval::run_forward_benchmark(&mut model, &batch, args.repeats)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    println!("{}", result.to_markdown());
    let flops = estimate_flops(&cfg, args.batch_size, args.seq_len);
    eprintln!(
        "estimated_forward_flops={:.3e} estimated_train_flops={:.3e}",
        flops.forward_flops, flops.train_flops
    );
    Ok(())
}

fn run_merge_adapter(args: MergeAdapterArgs) -> Result<()> {
    let cfg = load_config_or(args.config.as_deref(), get_tiny_lora_config)?;
    let mut model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let before =
        model.count_lora_modules() + model.count_qlora_modules() + model.count_dora_modules();
    model.merge_and_unload_adapters()?;
    fs::create_dir_all(&args.output_dir)?;
    fs::write(
        args.output_dir.join("inspection.json"),
        serde_json::to_string_pretty(&ModelInspection::from_model(&model))?,
    )?;
    train::save_checkpoint_metadata(
        args.output_dir.join("metadata.json"),
        0,
        0,
        0.0,
        Some(&model.config),
    )?;
    train::save_model_safetensors(args.output_dir.join("model.safetensors"), &model)?;
    println!(
        "merged adapters: before={} after={}",
        before,
        model.count_lora_modules() + model.count_qlora_modules() + model.count_dora_modules()
    );
    Ok(())
}

fn run_tokenizer_train(args: TokenizerTrainArgs) -> Result<()> {
    train_bpe_tokenizer(&args.input, &args.output, args.vocab_size)?;
    println!("wrote tokenizer to {}", args.output.display());
    Ok(())
}

fn load_config_or(path: Option<&Path>, default: impl FnOnce() -> ApexConfig) -> Result<ApexConfig> {
    match path {
        Some(path) => ApexConfig::from_yaml(path),
        None => Ok(default()),
    }
}

fn synthetic_tokens(seq_len: usize, vocab_size: usize) -> Vec<u32> {
    let limit = vocab_size.max(16) as u32;
    (0..seq_len.max(2))
        .map(|i| 9 + (i as u32 % limit.saturating_sub(9).max(1)))
        .collect()
}

fn write_train_outputs(
    output_dir: &Path,
    cfg: &ApexConfig,
    loss: f64,
    model: &ApexModel,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    cfg.to_yaml(output_dir.join("config.yaml"))?;
    train::save_checkpoint_metadata(output_dir.join("metadata.json"), 0, 0, loss, Some(cfg))?;
    train::save_model_safetensors(output_dir.join("model.safetensors"), model)?;
    if model.config.peft.enabled {
        train::save_adapter_safetensors(output_dir.join("adapter.safetensors"), model)?;
    }
    fs::write(
        output_dir.join("README.md"),
        "APEX Rust checkpoint artifacts. Tensor payloads use safetensors with JSON metadata and YAML config sidecars.\n",
    )?;
    println!(
        "wrote training artifacts to {} (loss={:.4})",
        output_dir.display(),
        loss
    );
    Ok(())
}
