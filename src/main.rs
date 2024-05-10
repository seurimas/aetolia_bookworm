use anyhow::Result;

use clap::Parser;
mod aetolia_api;
mod collection;
mod embedding_utils;
mod mistral_api;
mod prelude;
mod qdrant_utils;
use aetolia_api::*;
use embedding_utils::*;
use prelude::*;
use qdrant_utils::*;

mod cli;
use cli::*;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Query::parse();
    // Example of top level client
    // You may also use tonic-generated client from `src/qdrant.rs`
    if args.verbose {
        println!("Opening qdrant connection");
    }
    let qdrant = make_client().await?;
    if args.verbose {
        println!("Opening mistral connection");
    }
    let mistral = MistralClient::new()?;
    let aetolia = AetoliaClient::new();

    let collection = args.collection.to_events();
    if args.verbose {
        println!("Collection: {:?}", collection);
    }
    initialize_collection(&qdrant, &collection).await;

    if args.catchup {
        aetolia
            .nstat_catchup(&qdrant, &mistral, args.verbose)
            .await?;
    }

    let query = args.query;

    let proper_noun = if !args.no_pronouns {
        Some(mistral.get_proper_nouns(&query).await?)
    } else {
        None
    };

    let (query_embeddings, search_result) = if let Some(proper_noun) = &proper_noun {
        search_with_pronouns(
            &qdrant,
            &mistral,
            &collection,
            &query,
            proper_noun,
            args.limit,
        )
        .await?
    } else {
        search_without_pronouns(&qdrant, &mistral, &collection, &query, args.limit).await?
    };

    let context = get_context_from_scored_point(&search_result.result);

    let answer = mistral.chat_with_context(&query, &context).await?;

    let mut bookworm_response =
        BookwormResponse::from_search_response_and_answer(&search_result, answer.clone())
            .with_proper_nouns(proper_noun)
            .with_collection(collection.clone());
    if args.no_context {
        bookworm_response = bookworm_response.without_context();
    }
    remember_query_and_results(&qdrant, &bookworm_response, query_embeddings, &query).await?;

    if !args.no_json {
        println!("{}", serde_json::to_string(&bookworm_response)?);
    } else {
        println!("{}", answer);
    }
    Ok(())
}
