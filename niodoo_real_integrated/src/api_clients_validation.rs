// This file validates the api_clients module structure at compile time
// It can be used to ensure the public interface is correct

#[allow(dead_code)]
mod validation {
    use crate::api_clients::{ClaudeClient, GptClient};
    
    // These functions validate that the API exists and is correct type
    // They are never called but prevent compilation if the interface changes
    
    fn _validate_claude_client_creation() {
        // Proves ClaudeClient::new exists and takes the right parameters
        let _: fn(String, &str, u64) -> _ = |key, model, timeout| {
            ClaudeClient::new(key, model, timeout)
        };
    }
    
    fn _validate_gpt_client_creation() {
        // Proves GptClient::new exists and takes the right parameters
        let _: fn(String, &str, u64) -> _ = |key, model, timeout| {
            GptClient::new(key, model, timeout)
        };
    }
}
