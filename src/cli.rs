use std::collections::{HashMap, HashSet};

use crate::{get_context_from_payloads, get_post_id_from_payload, prelude::*};
use clap::{Parser, ValueEnum};
use mistralai_client::v1::constants::Model;
use qdrant_client::qdrant::{SearchResponse, Value};

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
pub struct Query {
    #[arg(value_enum)]
    pub collection: CollectionType,

    #[arg(long)]
    pub model: Option<String>,

    #[arg(short, long, default_value = "false")]
    pub catchup: bool,

    #[arg(long, short = 'x', default_value = "false")]
    pub no_pronouns: bool,

    #[arg(long, short = 'a', default_value = "false")]
    pub no_json: bool,

    #[arg(long, short = 'k', default_value = "false")]
    pub reranker: bool,

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
    pub used_references: HashSet<i64>,
    pub proper_nouns: Option<Vec<String>>,
    pub context: Option<String>,
    pub collection: Option<Collection>,
    pub answer: String,
    pub model: Option<Model>,
}

impl BookwormResponse {
    pub fn from_search_response_and_answer(
        response: &SearchResponse,
        payloads: &Vec<HashMap<String, Value>>,
        answer: String,
    ) -> Self {
        let references = response
            .result
            .iter()
            .map(|point| get_post_id_from_payload(&point.payload))
            .collect();
        let used_references = payloads
            .iter()
            .map(|point| get_post_id_from_payload(&point))
            .collect();
        let context = get_context_from_payloads(
            &response
                .result
                .iter()
                .map(|point| &point.payload)
                .cloned()
                .collect(),
        );
        Self {
            references,
            used_references,
            proper_nouns: None,
            context: Some(context),
            collection: None,
            answer,
            model: None,
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

    pub fn with_model(mut self, model: Model) -> Self {
        self.model = Some(model);
        self
    }
}
