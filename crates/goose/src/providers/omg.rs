use crate::message::{Message, MessageContent};
use crate::model::ModelConfig;
use crate::providers::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use crate::providers::errors::ProviderError;
use mcp_core::tool::Tool;
use anyhow::Result;
use async_trait::async_trait;
use reqwest::{Client, header};
use serde::Deserialize;

const OMG_API_URL: &str = "https://api.ohmygpt.com/v1";
const OMG_DEFAULT_MODEL: &str = "gpt-4o";
const OMG_DOC_URL: &str = "https://docs.ohmygpt.com";
const OMG_KNOWN_MODELS: &[&str] = &["gpt-4o", "claude-3-5-sonnet"];

#[derive(Debug, Clone)]
pub struct OmgProvider {
    client: Client,
    api_key: String,
    model: ModelConfig,
}

impl OmgProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("OMG_API_KEY")?;

        Ok(Self {
            client: Client::new(),
            api_key,
            model,
        })
    }

    fn create_headers(&self) -> Result<header::HeaderMap, ProviderError> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|e| ProviderError::ExecutionError(e.to_string()))?,
        );
        Ok(headers)
    }
}

impl Default for OmgProvider {
    fn default() -> Self {
        let model = ModelConfig::new(OmgProvider::metadata().default_model);
        OmgProvider::from_env(model).expect("Failed to initialize Omg provider")
    }
}

#[async_trait]
impl Provider for OmgProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "omg",
            "Omg",
            "Access GPT models through Omg API",
            OMG_DEFAULT_MODEL,
            OMG_KNOWN_MODELS.iter().map(|&s| s.to_string()).collect(),
            OMG_DOC_URL,
            vec![ConfigKey::new("OMG_API_KEY", true, true, None)],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let headers = self.create_headers()?;

        // Convert messages to the format expected by Omg API
        let mut api_messages = Vec::new();
        
        // Add system message first
        if !system.is_empty() {
            api_messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }

        // Add conversation messages
        for message in messages {
            let role = match message.role {
                mcp_core::role::Role::User => "user",
                mcp_core::role::Role::Assistant => "assistant",
            };

            for content in &message.content {
                if let MessageContent::Text(text) = content {
                    api_messages.push(serde_json::json!({
                        "role": role,
                        "content": text.text
                    }));
                }
            }
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", OMG_API_URL))
            .headers(headers)
            .json(&serde_json::json!({
                "model": self.model.model_name,
                "messages": api_messages,
            }))
            .send()
            .await
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await
                .map_err(|e| ProviderError::ExecutionError(e.to_string()))?;
            return Err(ProviderError::RequestFailed(error_text));
        }

        let chat_response: ChatResponse = response.json().await
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))?;
        
        let usage = if let Some(api_usage) = chat_response.usage {
            Usage::new(
                Some(api_usage.prompt_tokens),
                Some(api_usage.completion_tokens),
                Some(api_usage.total_tokens),
            )
        } else {
            Usage::default()
        };

        let message = Message {
            role: mcp_core::role::Role::Assistant,
            created: chrono::Utc::now().timestamp(),
            content: vec![MessageContent::text(
                chat_response.choices[0].message.content.clone(),
            )],
        };

        Ok((message, ProviderUsage::new(self.model.model_name.clone(), usage)))
    }
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<ApiUsage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: String,
}

#[derive(Debug, Deserialize, Clone)]
struct ApiUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
} 