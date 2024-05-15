pub use crate::aetolia_api::{AetoliaClient, NewsPost, NstatEntry};
pub use crate::collection::{Collection, CollectionType};
pub use crate::mistral_api::MistralClient;
pub use crate::qdrant_utils::{initialize_collection, make_client, news_post_exists};

pub use anyhow::Result;
pub use serde::{Deserialize, Serialize};

pub const QDRANT_API_KEY: &str = include_str!("C:\\Users\\mtgat\\OneDrive\\qdrant.txt");
pub const MISTRAL_API_KEY: &str = include_str!("C:\\Users\\mtgat\\OneDrive\\mistral.txt");
pub const JINA_API_KEY: &str = include_str!("C:\\Users\\mtgat\\OneDrive\\jina.txt");
pub const EMBEDDINGS_SIZE: usize = 1024;
