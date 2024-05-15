use std::any;

use crate::prelude::*;
use mistralai_client::v1::{
    chat::{ChatMessage, ChatParams},
    client::Client,
    constants::{EmbedModel, Model},
    error::{ApiError, ClientError},
    model_list::ModelListResponse,
};
use serde::Deserialize;

pub struct MistralClient(Client);

impl MistralClient {
    pub fn new() -> Result<Self, ClientError> {
        let client = Client::new(Some(MISTRAL_API_KEY.to_string()), None, None, None)
            .map(|client| MistralClient(client));
        client
    }
}

impl MistralClient {
    pub async fn get_models(&self) -> Result<ModelListResponse> {
        self.0
            .list_models_async()
            .await
            .map_err(anyhow::Error::from)
    }

    pub async fn get_embeddings(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, ApiError> {
        let model = EmbedModel::MistralEmbed;
        let mut embeddings = vec![];
        for super_chunk in input.chunks(50) {
            let response = self
                .0
                .embeddings_async(model.clone(), super_chunk.to_vec(), None)
                .await?;
            embeddings.extend(response.data.into_iter());
        }
        Ok(embeddings.iter().map(|e| e.embedding.clone()).collect())
    }

    pub async fn get_embeddings_single(&self, input: impl ToString) -> Result<Vec<f32>, ApiError> {
        self.get_embeddings(vec![input.to_string()])
            .await
            .map(|e| e.into_iter().next().unwrap())
    }

    fn get_chunks(
        input: &str,
        chunk_size: usize,
        overlap_size: usize,
    ) -> Vec<(usize, usize, String)> {
        let mut chunks = vec![];
        let mut chunk_start = 0;
        while chunk_start < input.len() {
            let chunk_end = (chunk_start + chunk_size).min(input.len());
            chunks.push((
                chunk_start,
                chunk_end,
                input[chunk_start..chunk_end].to_string(),
            ));
            chunk_start += overlap_size;
        }
        chunks
    }

    pub async fn get_embeddings_chunked(
        &self,
        input: &String,
        chunk_size: usize,
        overlap_size: usize,
    ) -> Result<Vec<(usize, usize, Vec<f32>)>, ApiError> {
        let chunks = Self::get_chunks(input, chunk_size, overlap_size);
        self.get_embeddings(chunks.iter().map(|(_, _, c)| c.clone()).collect())
            .await
            .map(|embeddings| {
                chunks
                    .into_iter()
                    .zip(embeddings.into_iter())
                    .map(|((start, end, _), embedding)| (start, end, embedding))
                    .collect()
            })
    }

    pub async fn get_embeddings_summarized(
        &self,
        input: &String,
    ) -> Result<(String, Vec<f32>), ApiError> {
        let summary_prompt = format!(
            "Summarize the following news posting. Do not explain your answer. Do not include anything before or after the summary.\nNews posting:{}\nSummary:",
            input
        );
        let summary = self.chat(summary_prompt, Model::OpenMistral7b).await?;
        let embeddings = self.get_embeddings_single(&summary).await?;
        Ok((summary, embeddings))
    }

    pub async fn chat(&self, prompt: String, model: Model) -> Result<String, ApiError> {
        let chat = ChatMessage {
            content: prompt,
            role: mistralai_client::v1::chat::ChatMessageRole::User,
            tool_calls: None,
        };
        let response = self.0.chat_async(model, vec![chat], None).await?;
        Ok(response.choices[0].message.content.to_string())
    }

    pub async fn chat_with_context(&self, input: &str, context: &str) -> Result<String, ApiError> {
        let model = mistralai_client::v1::constants::Model::OpenMistral7b;
        let prompt = format!("Context information is below:\n{}\n\nGiven the context information and not prior knowledge, answer the query authoritatively. Some of the context may not be relevant to query. Do not explain your answer. Dates with MA come before dates with AC, and AC is more current and relevant. The current year is 5 AC.\nQuery:\n{}\nAnswer:\n", context, input);
        self.chat(prompt, model).await
    }

    pub async fn get_proper_nouns(&self, input: impl ToString) -> Result<Vec<String>, ApiError> {
        let model = Model::OpenMixtral8x7b;
        let prompt = "Given the following query, generate a single list of any and all of the proper nouns for people, places, or things in the query. Follow this example:\nQuery: who was the first king of Blastonia?\nResponse:".to_string();
        let prompt_chat = ChatMessage {
            content: prompt,
            role: mistralai_client::v1::chat::ChatMessageRole::User,
            tool_calls: None,
        };
        let example_chat = ChatMessage {
            content: "[\"Blastonia\"]".to_string(),
            role: mistralai_client::v1::chat::ChatMessageRole::Assistant,
            tool_calls: None,
        };
        let query_chat = ChatMessage {
            content: format!("Query: {}\nResponse:", input.to_string()),
            role: mistralai_client::v1::chat::ChatMessageRole::User,
            tool_calls: None,
        };
        let response = self
            .0
            .chat_async(
                model,
                vec![prompt_chat, example_chat, query_chat],
                Some(ChatParams::json_default()),
            )
            .await?;
        Ok(
            serde_json::de::from_str(&response.choices[0].message.content).map_err(|err| {
                ApiError {
                    message: err.to_string(),
                }
            })?,
        )
    }

    pub async fn hypothetical_document(&self, input: &str) -> Result<String, ApiError> {
        let model = Model::OpenMixtral8x7b;
        let prompt = format!("Given the following query, generate a list of all of the proper nouns in the query. The setting for the query is a dark fantasy world. Do not use names of people or places that are not included in the original question. \nQuery:{}\nList:", input);
        let chat = ChatMessage {
            content: prompt,
            role: mistralai_client::v1::chat::ChatMessageRole::User,
            tool_calls: None,
        };
        let response = self.0.chat_async(model, vec![chat], None).await?;
        Ok(format!(
            "{}\n{}",
            input, response.choices[0].message.content
        ))
    }
}
