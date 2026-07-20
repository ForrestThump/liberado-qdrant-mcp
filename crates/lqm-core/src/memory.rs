//! Long-term agent memories: schema helpers and recency/importance ranking.
//!
//! Memories live in a dedicated collection by default (`DEFAULT_MEMORY_COLLECTION`)
//! with `source_type = "memory"` so they stay distinguishable from document chunks.
//! Generation stays in the host agent — this module only store/recall.

use crate::constants;
use crate::types::{DocumentChunk, SearchResult, payload_schema};
use serde::{Deserialize, Serialize};

/// Default Qdrant collection for agent memories.
pub const DEFAULT_MEMORY_COLLECTION: &str = "memories";

/// Payload / source_type marker for memory points.
pub const MEMORY_SOURCE_TYPE: &str = constants::SOURCE_TYPE_MEMORY;

/// Extra payload keys for memories (in addition to standard ingest keys).
pub mod memory_payload {
    pub const IMPORTANCE: &str = "importance";
    pub const LAST_ACCESSED: &str = "last_accessed";
    pub const MEMORY_ID: &str = "memory_id";
}

/// Parse a 0–1 importance from a Qdrant-restored payload value.
///
/// Production upsert maps non-string JSON to Qdrant `StringValue` via `v.to_string()`,
/// so numbers come back as JSON strings (`"0.85"`). Accept both number and string forms.
pub fn parse_importance_value(v: &serde_json::Value) -> Option<f32> {
    if let Some(f) = v.as_f64() {
        return Some((f as f32).clamp(0.0, 1.0));
    }
    if let Some(s) = v.as_str() {
        return s.trim().parse::<f32>().ok().map(|f| f.clamp(0.0, 1.0));
    }
    // Integers sometimes appear as i64 in synthetic JSON.
    if let Some(i) = v.as_i64() {
        return Some((i as f32).clamp(0.0, 1.0));
    }
    None
}

/// Input for storing one memory note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNote {
    pub text: String,
    /// 0.0–1.0 importance; default 0.5 when omitted at the API layer.
    pub importance: Option<f32>,
    pub tags: Option<Vec<String>>,
    pub project: Option<String>,
    /// Optional stable id; becomes `source` / `memory_id` for idempotent replace.
    pub memory_id: Option<String>,
}

/// One recalled memory with ranking metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryHit {
    pub text: String,
    /// Base semantic score from the vector search.
    pub score: f32,
    /// Score after optional recency/importance blend.
    pub blended_score: f32,
    pub importance: Option<f32>,
    pub last_accessed: Option<String>,
    pub memory_id: Option<String>,
    pub tags: Option<Vec<String>>,
    pub project: Option<String>,
    pub payload: serde_json::Value,
}

/// Build a `DocumentChunk` for the memory collection with stable schema fields.
///
/// Pure helper — does not touch Qdrant.
pub fn memory_note_to_chunk(
    note: &MemoryNote,
    collection: &str,
    now_unix_secs: u64,
) -> DocumentChunk {
    let importance = note
        .importance
        .unwrap_or(constants::DEFAULT_MEMORY_IMPORTANCE)
        .clamp(0.0, 1.0);
    let memory_id = note
        .memory_id
        .clone()
        .unwrap_or_else(|| format!("{}{}", constants::MEMORY_ID_PREFIX, now_unix_secs));
    let source = format!("{}{memory_id}", constants::MEMORY_SOURCE_PREFIX);

    DocumentChunk {
        text: note.text.clone(),
        source: Some(source),
        source_type: Some(MEMORY_SOURCE_TYPE.to_string()),
        collection: Some(collection.to_string()),
        tags: note.tags.clone(),
        timestamp: Some(now_unix_secs.to_string()),
        project: note.project.clone(),
        last_modified: Some(now_unix_secs.to_string()),
        chunk_index: Some(0),
        total_chunks: Some(1),
        importance: Some(importance),
        memory_id: Some(memory_id),
        scope: None,
        clearance: None,
    }
}

/// Attach memory-specific payload fields (importance, last_accessed, memory_id) as a
/// JSON map merge helper for tests / docs. Production store path uses DocumentChunk
/// tags/project and sets these keys on the point after `build_point_payload`.
pub fn memory_extra_payload(
    importance: f32,
    last_accessed_unix: u64,
    memory_id: &str,
) -> serde_json::Map<String, serde_json::Value> {
    let mut m = serde_json::Map::new();
    // Store as string so Qdrant StringValue round-trip preserves the value without
    // relying on Number→String coercion of `v.to_string()` in upsert_points.
    m.insert(
        memory_payload::IMPORTANCE.to_string(),
        serde_json::Value::String(format!("{}", importance.clamp(0.0, 1.0))),
    );
    m.insert(
        memory_payload::LAST_ACCESSED.to_string(),
        serde_json::Value::String(last_accessed_unix.to_string()),
    );
    m.insert(
        memory_payload::MEMORY_ID.to_string(),
        serde_json::Value::String(memory_id.to_string()),
    );
    m
}

/// Merge memory extras into a full point payload built by `build_point_payload`.
pub fn merge_memory_into_payload(
    mut payload: serde_json::Map<String, serde_json::Value>,
    importance: f32,
    last_accessed_unix: u64,
    memory_id: &str,
) -> serde_json::Map<String, serde_json::Value> {
    let extra = memory_extra_payload(importance, last_accessed_unix, memory_id);
    for (k, v) in extra {
        payload.insert(k, v);
    }
    // Ensure source_type is memory even if caller omitted it.
    payload.insert(
        payload_schema::SOURCE_TYPE.to_string(),
        serde_json::Value::String(MEMORY_SOURCE_TYPE.to_string()),
    );
    payload
}

/// Parse unix seconds from a string payload field.
pub fn parse_unix_secs(s: Option<&str>) -> Option<u64> {
    s.and_then(|v| v.parse().ok())
}

/// Blend semantic score with importance and recency.
///
/// All components in ~[0,1]. `now_unix` and `last_accessed` are unix seconds.
/// `half_life_secs` controls recency decay (larger = slower decay).
///
/// `blended = w_sem * score + w_imp * importance + w_rec * recency`
/// with weights defaulting to 0.6 / 0.25 / 0.15 when `use_recency` is true.
pub fn blend_memory_score(
    semantic_score: f32,
    importance: f32,
    last_accessed_unix: Option<u64>,
    now_unix: u64,
    use_recency: bool,
    half_life_secs: f32,
) -> f32 {
    let imp = importance.clamp(0.0, 1.0);
    // Cosine scores are often ~0..1; clamp for safety.
    let sem = semantic_score.clamp(0.0, 1.0);

    if !use_recency {
        return constants::MEMORY_BLEND_SEM_WEIGHT * sem + constants::MEMORY_BLEND_IMP_WEIGHT * imp;
    }

    let half = half_life_secs.max(1.0);
    let recency = match last_accessed_unix {
        Some(ts) => {
            let age = now_unix.saturating_sub(ts) as f32;
            // Exponential decay: 1 at age 0, 0.5 at half_life.
            (-(age / half) * std::f32::consts::LN_2)
                .exp()
                .clamp(0.0, 1.0)
        }
        None => 0.0,
    };

    constants::MEMORY_BLEND_RECENCY_SEM_WEIGHT * sem
        + constants::MEMORY_BLEND_RECENCY_IMP_WEIGHT * imp
        + constants::MEMORY_BLEND_RECENCY_REC_WEIGHT * recency
}

/// Convert search hits into `MemoryHit`s and optionally re-sort by blended score.
pub fn rank_memory_hits(
    results: &[SearchResult],
    now_unix: u64,
    use_recency: bool,
    half_life_secs: f32,
) -> Vec<MemoryHit> {
    let mut hits: Vec<MemoryHit> = results
        .iter()
        .map(|r| {
            let importance = r
                .payload
                .get(memory_payload::IMPORTANCE)
                .and_then(parse_importance_value)
                .unwrap_or(constants::DEFAULT_MEMORY_IMPORTANCE);
            // last_accessed is always written as a string; also accept number-ish forms.
            let last_accessed = r.payload.get(memory_payload::LAST_ACCESSED).and_then(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .or_else(|| v.as_u64().map(|n| n.to_string()))
                    .or_else(|| v.as_i64().map(|n| n.to_string()))
            });
            let last_ts = parse_unix_secs(last_accessed.as_deref());
            let blended = blend_memory_score(
                r.score,
                importance,
                last_ts,
                now_unix,
                use_recency,
                half_life_secs,
            );
            let memory_id = r
                .payload
                .get(memory_payload::MEMORY_ID)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| {
                    r.payload
                        .get(payload_schema::SOURCE)
                        .and_then(|v| v.as_str())
                        .map(|s| {
                            s.trim_start_matches(constants::MEMORY_SOURCE_PREFIX)
                                .to_string()
                        })
                });
            let tags = r.payload.get(payload_schema::TAGS).and_then(|v| {
                v.as_array().map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
            });
            let project = r
                .payload
                .get(payload_schema::PROJECT)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            MemoryHit {
                text: r.text.clone(),
                score: r.score,
                blended_score: blended,
                importance: Some(importance),
                last_accessed,
                memory_id,
                tags,
                project,
                payload: r.payload.clone(),
            }
        })
        .collect();

    if use_recency {
        hits.sort_by(|a, b| {
            b.blended_score
                .partial_cmp(&a.blended_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn memory_note_to_chunk_sets_schema() {
        let note = MemoryNote {
            text: "User prefers dark mode".into(),
            importance: Some(0.9),
            tags: Some(vec!["prefs".into()]),
            project: Some("ui".into()),
            memory_id: Some("pref-dark".into()),
        };
        let chunk = memory_note_to_chunk(&note, DEFAULT_MEMORY_COLLECTION, 1_700_000_000);
        assert_eq!(chunk.text, "User prefers dark mode");
        assert_eq!(chunk.collection.as_deref(), Some(DEFAULT_MEMORY_COLLECTION));
        assert_eq!(chunk.source_type.as_deref(), Some(MEMORY_SOURCE_TYPE));
        assert_eq!(chunk.source.as_deref(), Some("memory://pref-dark"));
        assert_eq!(chunk.project.as_deref(), Some("ui"));
        assert_eq!(chunk.tags.as_ref().unwrap(), &vec!["prefs".to_string()]);
        assert_eq!(chunk.chunk_index, Some(0));
        assert_eq!(chunk.total_chunks, Some(1));
    }

    #[test]
    fn merge_memory_payload_has_keys() {
        let base = crate::qdrant::build_point_payload(
            &DocumentChunk {
                text: "x".into(),
                source: Some("memory://id".into()),
                source_type: Some(MEMORY_SOURCE_TYPE.into()),
                collection: Some(DEFAULT_MEMORY_COLLECTION.into()),
                tags: None,
                timestamp: None,
                project: None,
                last_modified: None,
                chunk_index: Some(0),
                total_chunks: Some(1),
                importance: Some(0.8),
                memory_id: Some("id".into()),
                scope: None,
                clearance: None,
            },
            0,
            1,
            "fake",
        );
        // build_point_payload writes importance as string for Qdrant round-trip.
        let built_imp = parse_importance_value(&base[memory_payload::IMPORTANCE]).unwrap();
        assert!((built_imp - 0.8).abs() < 1e-5, "built={built_imp}");
        assert!(base[memory_payload::IMPORTANCE].as_str().is_some());

        let merged = merge_memory_into_payload(base, 0.8, 100, "id");
        let imp = parse_importance_value(&merged[memory_payload::IMPORTANCE]).unwrap();
        assert!((imp - 0.8).abs() < 1e-5, "merged={imp}");
        assert_eq!(merged[memory_payload::LAST_ACCESSED], "100");
        assert_eq!(merged[memory_payload::MEMORY_ID], "id");
        assert_eq!(merged[payload_schema::SOURCE_TYPE], MEMORY_SOURCE_TYPE);
    }

    #[test]
    fn parse_importance_accepts_number_and_string() {
        assert!((parse_importance_value(&json!(0.85)).unwrap() - 0.85).abs() < 1e-5);
        assert!((parse_importance_value(&json!("0.85")).unwrap() - 0.85).abs() < 1e-5);
        // Simulates Number→to_string() as performed by upsert_points for non-strings.
        let coerced = json!(0.9).to_string();
        assert_eq!(coerced, "0.9");
        assert!(
            (parse_importance_value(&serde_json::Value::String(coerced)).unwrap() - 0.9).abs()
                < 1e-5
        );
        assert_eq!(parse_importance_value(&json!("nope")), None);
    }

    #[test]
    fn parse_importance_accepts_integer() {
        // Integer JSON values hit as_i64(), not as_f64().
        assert!(
            (parse_importance_value(&json!(1)).unwrap() - 1.0).abs() < 1e-5,
            "integer 1 should parse as 1.0"
        );
        assert!(
            (parse_importance_value(&json!(0)).unwrap() - 0.0).abs() < 1e-5,
            "integer 0 should parse as 0.0"
        );
        // Out-of-range integer gets clamped
        assert!(
            (parse_importance_value(&json!(5)).unwrap() - 1.0).abs() < 1e-5,
            "integer > 1 should be clamped to 1.0"
        );
    }

    #[test]
    fn blend_higher_importance_wins_without_recency() {
        let low = blend_memory_score(0.9, 0.1, None, 1000, false, 86400.0);
        let high = blend_memory_score(0.9, 0.9, None, 1000, false, 86400.0);
        assert!(high > low, "high={high} low={low}");
    }

    #[test]
    fn blend_recent_outranks_stale_at_same_sem_imp() {
        let now = 10_000_000_u64;
        let recent = blend_memory_score(0.8, 0.5, Some(now - 60), now, true, 86_400.0);
        let stale = blend_memory_score(0.8, 0.5, Some(now - 30 * 86_400), now, true, 86_400.0);
        assert!(recent > stale, "recent={recent} stale={stale}");
    }

    #[test]
    fn rank_memory_hits_orders_by_blend() {
        let now = 10_000_000_u64;
        let results = vec![
            SearchResult {
                text: "old".into(),
                score: 0.9,
                payload: json!({
                    "importance": 0.5,
                    "last_accessed": "100",
                    "memory_id": "old",
                    "source": "memory://old",
                }),
            },
            SearchResult {
                text: "new".into(),
                score: 0.9,
                payload: json!({
                    "importance": 0.5,
                    "last_accessed": now.to_string(),
                    "memory_id": "new",
                    "source": "memory://new",
                }),
            },
        ];
        let ranked = rank_memory_hits(&results, now, true, 86_400.0);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].text, "new");
        assert!(ranked[0].blended_score >= ranked[1].blended_score);
    }

    /// Production shape: Qdrant returns importance as a JSON *string* after StringValue round-trip.
    #[test]
    fn rank_memory_hits_reads_string_importance_qdrant_shape() {
        let now = 10_000_000_u64;
        let results = vec![
            SearchResult {
                text: "low".into(),
                score: 0.9,
                // string importance — what scored_point_to_search_result actually yields
                payload: json!({
                    "importance": "0.1",
                    "last_accessed": now.to_string(),
                    "memory_id": "low",
                    "source": "memory://low",
                }),
            },
            SearchResult {
                text: "high".into(),
                score: 0.9,
                payload: json!({
                    "importance": "0.95",
                    "last_accessed": now.to_string(),
                    "memory_id": "high",
                    "source": "memory://high",
                }),
            },
        ];
        // use_recency=false still applies 0.75*sem + 0.25*imp, so higher string importance wins.
        let ranked = rank_memory_hits(&results, now, false, 86_400.0);
        assert_eq!(ranked.len(), 2);
        assert!(
            (ranked
                .iter()
                .find(|h| h.text == "high")
                .unwrap()
                .importance
                .unwrap()
                - 0.95)
                .abs()
                < 1e-5,
            "string importance must parse, got {:?}",
            ranked
        );
        assert!(
            (ranked
                .iter()
                .find(|h| h.text == "low")
                .unwrap()
                .importance
                .unwrap()
                - 0.1)
                .abs()
                < 1e-5
        );
        // Re-rank with recency off still keeps higher blended from importance:
        let high_blend = ranked
            .iter()
            .find(|h| h.text == "high")
            .unwrap()
            .blended_score;
        let low_blend = ranked
            .iter()
            .find(|h| h.text == "low")
            .unwrap()
            .blended_score;
        assert!(
            high_blend > low_blend,
            "high_blend={high_blend} low_blend={low_blend}"
        );
    }

    /// End-to-end pure path: build_point_payload → simulate upsert string coercion → rank.
    #[test]
    fn importance_survives_qdrant_string_coercion_path() {
        let chunk = memory_note_to_chunk(
            &MemoryNote {
                text: "pref".into(),
                importance: Some(0.85),
                tags: None,
                project: None,
                memory_id: Some("p1".into()),
            },
            DEFAULT_MEMORY_COLLECTION,
            1_700_000_000,
        );
        let payload_map = crate::qdrant::build_point_payload(&chunk, 0, 1, "fake");
        // Simulate what upsert does for any non-string (and our write is already string).
        let mut restored = serde_json::Map::new();
        for (k, v) in payload_map {
            let as_search = match v {
                serde_json::Value::String(s) => serde_json::Value::String(s),
                serde_json::Value::Array(a) => serde_json::Value::Array(a),
                other => serde_json::Value::String(other.to_string()),
            };
            restored.insert(k, as_search);
        }
        let imp = parse_importance_value(restored.get(memory_payload::IMPORTANCE).unwrap())
            .expect("importance after coercion");
        assert!((imp - 0.85).abs() < 1e-4, "imp={imp}");

        let results = vec![SearchResult {
            text: "pref".into(),
            score: 0.9,
            payload: serde_json::Value::Object(restored),
        }];
        let hits = rank_memory_hits(&results, 1_700_000_000, false, 86_400.0);
        assert!((hits[0].importance.unwrap() - 0.85).abs() < 1e-4);
    }

    #[test]
    fn rank_memory_hits_empty_returns_empty() {
        let hits = rank_memory_hits(&[], 1_700_000_000, true, 86_400.0);
        assert!(hits.is_empty());
    }
}
