use std::collections::HashMap;

use qdrant_client::{client::Payload, qdrant::Value};
use reqwest::Client;

use crate::prelude::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct JinaRequest {
    model: &'static str,
    query: String,
    documents: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JinaResult {
    index: usize,
    relevance_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JinaResults {
    pub results: Vec<JinaResult>,
}

pub struct JinaClient(Client);

impl JinaClient {
    pub fn new() -> Self {
        Self(Client::new())
    }

    pub async fn rerank_payloads_and_limit(
        &self,
        query: &str,
        payloads: Vec<HashMap<String, Value>>,
        limit: u64,
    ) -> Result<Vec<HashMap<String, Value>>, reqwest::Error> {
        let url = "https://api.jina.ai/v1/rerank";
        let documents = payloads
            .iter()
            .flat_map(|payload| {
                if let Some(summary) = payload.get("summary") {
                    Some(summary.to_string())
                } else if let Some(chunk_data) = payload.get("chunk_data") {
                    Some(chunk_data.to_string())
                } else if let (Some(chunk_start), Some(chunk_end)) =
                    (payload.get("chunk_start"), payload.get("chunk_end"))
                {
                    let start = chunk_start.as_integer().unwrap() as usize;
                    let end = chunk_end.as_integer().unwrap() as usize;
                    Some(payload["message"].as_str().unwrap()[start..end].to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let response = self
            .0
            .post(url)
            .json(&JinaRequest {
                model: "jina-reranker-v1-base-en",
                query: query.to_string(),
                documents,
            })
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", JINA_API_KEY))
            .send()
            .await?;
        let results = response.json::<JinaResults>().await?;
        Ok(results
            .results
            .iter()
            .map(|r| payloads[r.index].clone())
            .take(limit as usize)
            .collect())
    }
}
