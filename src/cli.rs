use std::collections::HashSet;

use crate::{get_context_from_scored_point, get_post_id_from_payload, prelude::*};
use clap::{Parser, ValueEnum};
use qdrant_client::qdrant::SearchResponse;

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
pub struct Query {
    #[arg(value_enum)]
    pub collection: CollectionType,

    #[arg(short, long, default_value = "false")]
    pub catchup: bool,

    #[arg(long, short = 'x', default_value = "false")]
    pub no_pronouns: bool,

    #[arg(long, short = 'a', default_value = "false")]
    pub no_json: bool,

    #[arg(long, default_value = "false")]
    pub no_context: bool,

    #[arg(short, long)]
    pub limit: Option<u64>,

    #[arg(short, long, default_value = "false")]
    pub verbose: bool,

    #[arg(value_enum)]
    pub query: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BookwormResponse {
    pub references: HashSet<i64>,
    pub proper_nouns: Option<Vec<String>>,
    pub context: Option<String>,
    pub collection: Option<Collection>,
    pub answer: String,
}

impl BookwormResponse {
    pub fn from_search_response_and_answer(response: &SearchResponse, answer: String) -> Self {
        let references = response
            .result
            .iter()
            .map(|point| get_post_id_from_payload(&point.payload))
            .collect();
        let context = get_context_from_scored_point(&response.result);
        Self {
            references,
            proper_nouns: None,
            context: Some(context),
            collection: None,
            answer,
        }
    }

    pub fn with_collection(mut self, collection: Collection) -> Self {
        self.collection = Some(collection);
        self
    }

    pub fn with_proper_nouns(mut self, proper_nouns: Option<Vec<String>>) -> Self {
        self.proper_nouns = proper_nouns;
        self
    }

    pub fn without_context(mut self) -> Self {
        self.context = None;
        self
    }
}
