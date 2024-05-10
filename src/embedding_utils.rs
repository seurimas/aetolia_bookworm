use qdrant_client::{
    client::{Payload, QdrantClient},
    qdrant::PointStruct,
};
use serde_json::json;

use crate::prelude::*;

pub async fn add_news_post(
    client: &QdrantClient,
    mistral_client: &MistralClient,
    aetolia: &AetoliaClient,
    collection: &Collection,
    id: u32,
    verbose: bool,
) -> Result<bool> {
    if news_post_exists(client, collection, id).await {
        if verbose {
            println!("Post {} already exists", id);
        }
        return Ok(false);
    }
    let post = aetolia.get_news_post(collection.section(), id).await?;
    if collection.is_summary() {
        add_news_post_summarized(client, mistral_client, post, collection).await
    } else {
        add_news_post_chunked(
            client,
            mistral_client,
            post,
            collection,
            collection.get_chunk_size(),
            collection.get_overlap_size(),
        )
        .await
    }
}

pub async fn add_news_post_chunked(
    client: &QdrantClient,
    mistral_client: &MistralClient,
    post: NewsPost,
    collection: &Collection,
    chunk_size: usize,
    overlap_size: usize,
) -> Result<bool> {
    let collection_name = collection.name();
    let mut embeddings = mistral_client
        .get_embeddings_chunked(&post.message, chunk_size, overlap_size)
        .await?;
    let mut payloads: Vec<(u64, Vec<f32>, Payload)> = vec![];
    for (i, (chunk_start, chunk_end, embedding)) in embeddings.drain(..).enumerate() {
        let id = post.id * 10000 + i as u32;
        let payload = json!({
            "id": post.id,
            "section": post.section.clone(),
            "date": post.date,
            "date_ingame": post.date_ingame.clone(),
            "from": post.from.clone(),
            "to": post.to.clone(),
            "subject": post.subject.clone(),
            "message": post.message.clone(),
            "chunk": i,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        })
        .try_into()
        .unwrap();
        payloads.push((id as u64, embedding, payload));
    }
    let points = payloads
        .into_iter()
        .map(|(id, vector, payload)| PointStruct::new(id as u64, vector, payload))
        .collect();
    client
        .upsert_points_blocking(collection_name, None, points, None)
        .await?;
    Ok(true)
}

pub async fn add_news_post_summarized(
    client: &QdrantClient,
    mistral_client: &MistralClient,
    post: NewsPost,
    collection: &Collection,
) -> Result<bool> {
    let collection_name = collection.name();
    let (summary, embeddings) = mistral_client
        .get_embeddings_summarized(&post.message)
        .await?;
    let payload = json!({
        "id": post.id,
        "section": post.section.clone(),
        "date": post.date,
        "date_ingame": post.date_ingame.clone(),
        "from": post.from.clone(),
        "to": post.to.clone(),
        "subject": post.subject.clone(),
        "message": post.message.clone(),
        "summary": summary,
    })
    .try_into()
    .unwrap();
    let id = post.id * 10000;
    let point = PointStruct::new(id as u64, embeddings, payload);
    client
        .upsert_points_blocking(collection_name, None, vec![point], None)
        .await?;
    Ok(true)
}
