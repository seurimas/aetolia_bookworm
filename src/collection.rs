use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum Collection {
    Short(String),
    Long(String),
    Summary(String),
}

impl Collection {
    pub fn name(&self) -> String {
        match self {
            Collection::Short(name) => name.to_string(),
            Collection::Long(name) => format!("{}_long", name),
            Collection::Summary(name) => format!("{}_summary", name),
        }
    }

    pub fn section(&self) -> String {
        match self {
            Collection::Short(name) => name.to_string(),
            Collection::Long(name) => name.to_string(),
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
            Collection::Long(_) => 1000,
            Collection::Summary(_) => unreachable!(),
        }
    }

    pub fn get_overlap_size(&self) -> usize {
        match self {
            Collection::Short(_) => 200,
            Collection::Long(_) => 400,
            Collection::Summary(_) => unreachable!(),
        }
    }
}

#[derive(ValueEnum, Clone, Debug)]
pub enum CollectionType {
    Short,
    Long,
    Summary,
}

impl CollectionType {
    pub fn to_events(&self) -> Collection {
        match self {
            CollectionType::Short => Collection::Short("events".to_string()),
            CollectionType::Long => Collection::Long("events".to_string()),
            CollectionType::Summary => Collection::Summary("events".to_string()),
        }
    }
}
