use apex_rust::tokenizer::{ApexTokenizer, ChatMessage};

fn main() -> apex_rust::Result<()> {
    let tokenizer = ApexTokenizer::new(None::<&str>)?;
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are concise.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Explain RoPE in one sentence.".to_string(),
        },
    ];
    let prompt = tokenizer.format_chat(&messages, true, true);
    let ids = tokenizer.encode(&prompt, false)?;
    println!("{prompt}");
    println!("tokens={}", ids.len());
    println!("token_types={:?}", tokenizer.get_token_types(&ids));
    Ok(())
}
