use crate::message::Message;
use crate::model::ModelConfig;
use crate::providers::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use crate::providers::errors::ProviderError;
use crate::providers::formats::openai::{create_request, get_usage, response_to_message};
use crate::providers::utils::{emit_debug_trace, get_model, handle_response_openai_compat};
use mcp_core::tool::Tool;
use anyhow::Result;
use async_trait::async_trait;
use reqwest::{Client, header};
use serde::Serialize;
use serde_json::Value;

const OMG_API_URL: &str = "https://api.ohmygpt.com/v1";
const OMG_DEFAULT_MODEL: &str = "gpt-4o";
const OMG_DOC_URL: &str = "https://docs.ohmygpt.com";
const OMG_KNOWN_MODELS: &[&str] = &["gpt-4o", "claude-3-5-sonnet"];

#[derive(Debug, Clone, Serialize)]
pub struct OmgProvider {
    #[serde(skip)]
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

    async fn post(&self, payload: Value) -> Result<Value, ProviderError> {
        let url = format!("{}/chat/completions", OMG_API_URL);

        let response = self
            .client
            .post(&url)
            .headers(self.create_headers()?)
            .json(&payload)
            .send()
            .await?;

        handle_response_openai_compat(response).await
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
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        // Create the request payload using OpenAI format
        let payload = create_request(
            &self.model,
            system,
            messages,
            tools,
            &super::utils::ImageFormat::OpenAi,
        )?;

        // Make request
        let response = self.post(payload.clone()).await?;

        // Parse response
        let message = response_to_message(response.clone())?;
        let usage = match get_usage(&response) {
            Ok(usage) => usage,
            Err(ProviderError::UsageError(e)) => {
                tracing::warn!("Failed to get usage data: {}", e);
                Usage::default()
            }
            Err(e) => return Err(e),
        };
        let model = get_model(&response);
        emit_debug_trace(self, &payload, &response, &usage);
        Ok((message, ProviderUsage::new(model, usage)))
    }
} 