use std::collections::HashMap;

use crate::{prelude::*, BookwormResponse};
use anyhow::Result;
pub use qdrant_client::prelude::*;
pub use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{with_payload_selector::SelectorOptions, ScoredPoint, SearchResponse};
pub use qdrant_client::qdrant::{
    Condition, CreateCollection, Filter, SearchPoints, VectorParams, VectorsConfig,
};
use serde_json::json;

pub async fn make_client() -> Result<QdrantClient> {
    let client = QdrantClient::from_url(
        "https://22ec7284-818c-4ea1-ba15-ed26f7afdd0e.us-east4-0.gcp.cloud.qdrant.io:6334",
    )
    // using an env variable for the API KEY for example
    .with_api_key(QDRANT_API_KEY)
    .build()?;
    Ok(client)
}

pub async fn initialize_collection(client: &QdrantClient, collection: &Collection) {
    client
        .create_collection(&CreateCollection {
            collection_name: collection.name(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: EMBEDDINGS_SIZE as u64,
                    distance: Distance::Euclid.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await;
}

pub async fn remember_query_and_results(
    client: &QdrantClient,
    bookworm_response: &BookwormResponse,
    query_embeddings: Vec<f32>,
    query: &str,
) -> Result<()> {
    client
        .create_collection(&CreateCollection {
            collection_name: "queries".to_string(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: EMBEDDINGS_SIZE as u64,
                    distance: Distance::Euclid.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await;
    let payload: Payload = json!({
        "query": query,
        "answer": bookworm_response.answer,
        "references": bookworm_response.references,
        "proper_nouns": bookworm_response.proper_nouns,
        "context": bookworm_response.context,
    })
    .try_into()
    .unwrap();
    let point = PointStruct::new(uuid::Uuid::new_v4().to_string(), query_embeddings, payload);
    client
        .upsert_points_blocking("queries", None, vec![point], None)
        .await?;
    Ok(())
}

pub async fn news_post_exists(client: &QdrantClient, collection: &Collection, id: u32) -> bool {
    let first_id: u64 = (id * 10000) as u64;
    client
        .get_points(
            collection.name(),
            None,
            &[first_id.into()],
            false.into(),
            false.into(),
            None,
        )
        .await
        .map(|r| !r.result.is_empty())
        .unwrap_or(true)
}

pub fn get_context_from_payload(payload: &HashMap<String, Value>) -> (i64, String) {
    let message = if payload.contains_key("chunk_start") {
        let full_message = payload["message"]
            .as_str()
            .cloned()
            .unwrap_or("".to_string());
        let chunk_start = payload["chunk_start"].as_integer().unwrap() as usize;
        let chunk_end = payload["chunk_end"].as_integer().unwrap() as usize;
        full_message[chunk_start..chunk_end].to_string()
    } else {
        payload["summary"]
            .as_str()
            .cloned()
            .unwrap_or("".to_string())
    };
    let id = payload["id"].as_integer().unwrap_or(0);
    (id, message)
}

pub fn get_context_from_scored_point(scored_points: &Vec<ScoredPoint>) -> String {
    let chunks = scored_points
        .into_iter()
        .map(|scored_point| get_context_from_payload(&scored_point.payload))
        .collect::<Vec<_>>();
    let mut best_chunks = HashMap::new();
    for (id, message) in chunks.iter() {
        if best_chunks.contains_key(id) {
            best_chunks.insert(*id, format!("{}\n\n{}", best_chunks[id], message));
        } else {
            best_chunks.insert(*id, format!("Post {}:\n{}", id, message.clone()));
        }
    }
    let mut chunks_in_order = best_chunks.iter().collect::<Vec<_>>();
    chunks_in_order.sort_by_key(|(id, _)| *id);
    chunks_in_order
        .iter()
        .map(|(_, m)| m.to_string())
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub async fn search_with_pronouns(
    qdrant: &QdrantClient,
    mistral: &MistralClient,
    collection: &Collection,
    query: &str,
    nouns: &Vec<String>,
    limit: u64,
) -> Result<(Vec<f32>, SearchResponse)> {
    let embeddings = mistral.get_embeddings_single(query).await?;
    let search_result = qdrant
        .search_points(&SearchPoints {
            collection_name: collection.name(),
            vector: embeddings.clone(),
            limit,
            with_payload: Some(true.into()),
            filter: Some(Filter::any(
                nouns
                    .iter()
                    .map(|noun| Condition::matches_text("message", noun))
                    .collect::<Vec<_>>(),
            )),
            ..Default::default()
        })
        .await?;
    Ok((embeddings, search_result))
}

pub async fn search_without_pronouns(
    qdrant: &QdrantClient,
    mistral: &MistralClient,
    collection: &Collection,
    query: &str,
    limit: u64,
) -> Result<(Vec<f32>, SearchResponse)> {
    let embeddings = mistral.get_embeddings_single(query).await?;
    let search_result = qdrant
        .search_points(&SearchPoints {
            collection_name: collection.name(),
            vector: embeddings.clone(),
            limit,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;
    Ok((embeddings, search_result))
}

pub fn get_post_id_from_payload(payload: &HashMap<String, Value>) -> i64 {
    payload["id"].as_integer().unwrap() as i64
}
