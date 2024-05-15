use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum Collection {
    Short(String),
    Long(String),
    Dense(String),
    Summary(String),
}

impl Collection {
    pub fn name(&self) -> String {
        match self {
            Collection::Short(name) => name.to_string(),
            Collection::Long(name) => format!("{}_long", name),
            Collection::Dense(name) => format!("{}_dense", name),
            Collection::Summary(name) => format!("{}_summary", name),
        }
    }

    pub fn section(&self) -> String {
        match self {
            Collection::Short(name) => name.to_string(),
            Collection::Long(name) => name.to_string(),
            Collection::Dense(name) => name.to_string(),
            Collection::Summary(name) => name.to_string(),
        }
    }

    pub fn is_summary(&self) -> bool {
        match self {
            Collection::Summary(_) => true,
            _ => false,
        }
    }

    pub fn get_chunk_size(&self) -> usize {
        match self {
            Collection::Short(_) => 400,
            Collection::Dense(_) => 500,
            Collection::Long(_) => 1000,
            Collection::Summary(_) => unreachable!(),
        }
    }

    pub fn get_overlap_size(&self) -> usize {
        match self {
            Collection::Short(_) => 200,
            Collection::Dense(_) => 20,
            Collection::Long(_) => 400,
            Collection::Summary(_) => unreachable!(),
        }
    }

    pub fn default_limit(&self) -> u64 {
        match self {
            Collection::Short(_) => 10,
            Collection::Dense(_) => 10,
            Collection::Long(_) => 10,
            Collection::Summary(_) => 5,
        }
    }

    pub fn has_pronouns(&self) -> bool {
        match self {
            Collection::Dense(_) => false,
            _ => true,
        }
    }
}

#[derive(ValueEnum, Clone, Debug)]
pub enum CollectionType {
    Short,
    Long,
    Summary,
    Dense,
    PublicSummary,
}

impl CollectionType {
    pub fn to_collection(&self) -> Collection {
        match self {
            CollectionType::Short => Collection::Short("events".to_string()),
            CollectionType::Long => Collection::Long("events".to_string()),
            CollectionType::Summary => Collection::Summary("events".to_string()),
            CollectionType::Dense => Collection::Dense("events".to_string()),
            CollectionType::PublicSummary => Collection::Summary("public".to_string()),
        }
    }
}
