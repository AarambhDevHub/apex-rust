use apex_rust::config::get_tiny_config;
use apex_rust::generation::{ApexGenerator, GenerationConfig};
use apex_rust::model::ApexModel;
use apex_rust::tokenizer::{ApexTokenizer, ChatMessage};
use candle_core::Device;

fn main() -> apex_rust::Result<()> {
    let tokenizer = ApexTokenizer::new(None::<&str>)?;
    let mut model = ApexModel::new(get_tiny_config(), Device::Cpu)?;
    let input = tokenizer.encode_chat(
        &[ChatMessage {
            role: "user".to_string(),
            content: "Say hello.".to_string(),
        }],
        true,
        false,
    )?;
    let mut generator = ApexGenerator::new(
        &mut model,
        GenerationConfig {
            max_new_tokens: 8,
            temperature: 0.0,
            eos_token_id: tokenizer.eos_token_id(),
            ..GenerationConfig::default()
        },
    );
    let output = generator.generate(input, 0)?;
    println!("{}", tokenizer.decode(&output.token_ids, true)?);
    Ok(())
}
