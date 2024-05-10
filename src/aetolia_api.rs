use crate::{add_news_post, prelude::*};
use qdrant_client::client::QdrantClient;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct NewsPost {
    pub id: u32,
    pub section: String,
    pub date: u64,
    pub date_ingame: String,
    pub from: String,
    pub to: String,
    pub subject: String,
    pub message: String,
}

fn false_is_none<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Bool(false) => Ok(None),
        serde_json::Value::String(s) => Ok(Some(s)),
        _ => Err(serde::de::Error::custom("expected string or false")),
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PostResult {
    #[serde(deserialize_with = "false_is_none")]
    pub previous: Option<String>,
    #[serde(deserialize_with = "false_is_none")]
    pub next: Option<String>,
    pub post: NewsPost,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NstatEntry {
    pub uri: String,
    pub total: u32,
    pub name: String,
}

impl NstatEntry {
    pub fn section(&self) -> String {
        self.name.to_ascii_lowercase()
    }

    pub async fn catchup(
        &self,
        qdrant: &QdrantClient,
        mistral: &MistralClient,
        aetolia: &AetoliaClient,
        collection: &Collection,
        verbose: bool,
    ) -> Result<()> {
        for i in (1..=self.total).rev() {
            if verbose {
                println!("Catching up to news {}", i);
            }
            if !add_news_post(&qdrant, &mistral, &aetolia, &collection, i, verbose).await? {
                break;
            }
        }
        Ok(())
    }
}

pub struct AetoliaClient(Client);

impl AetoliaClient {
    pub fn new() -> Self {
        Self(Client::new())
    }

    pub async fn get_news_post(
        &self,
        section: impl ToString,
        id: u32,
    ) -> Result<NewsPost, reqwest::Error> {
        let url = format!(
            "https://api.aetolia.com/news/{}/{}.json",
            section.to_string(),
            id
        );
        let response = match self.0.get(&url).send().await {
            Ok(response) => Ok(response),
            Err(e) => self.0.get(&url).send().await,
        }?;
        let post = response.json::<PostResult>().await?;
        Ok(post.post)
    }

    pub async fn get_news_stats(&self) -> Result<Vec<NstatEntry>, reqwest::Error> {
        let url = "https://api.aetolia.com/news.json";
        let response = self.0.get(url).send().await?;
        let stats = response.json::<Vec<NstatEntry>>().await?;
        Ok(stats)
    }

    pub async fn nstat_catchup(
        &self,
        qdrant: &QdrantClient,
        mistral: &MistralClient,
        verbose: bool,
    ) -> Result<()> {
        let stats = self.get_news_stats().await?;
        for stat in stats {
            if stat.section() != "events" {
                continue;
            }
            stat.catchup(
                qdrant,
                mistral,
                self,
                &Collection::Short(stat.section()),
                verbose,
            )
            .await?;
        }
        Ok(())
    }
}
