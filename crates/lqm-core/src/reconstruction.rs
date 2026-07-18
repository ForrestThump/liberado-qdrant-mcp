//! Source reconstruction: order, paginate, and expand chunks by `source` pointer.
//!
//! Pure helpers are unit-tested offline. Qdrant I/O lives on `RagCore`.

use crate::types::{Payload, payload_schema, payload_str};
use serde::{Deserialize, Serialize};

/// Default page size for `list_chunks` when limit is omitted.
pub const DEFAULT_LIST_CHUNKS_LIMIT: u64 = 50;

/// Default ±N window for `expand_context` when neighbors is omitted.
pub const DEFAULT_EXPAND_NEIGHBORS: u64 = 1;

/// One indexed chunk belonging to a source (agent reconstruction).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceChunk {
    pub text: String,
    pub chunk_index: Option<u64>,
    pub total_chunks: Option<u64>,
    pub source: Option<String>,
    pub source_type: Option<String>,
    pub payload: Payload,
}

/// Paginated list of chunks for one source, ordered by `chunk_index`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceChunkPage {
    pub source: String,
    pub chunks: Vec<SourceChunk>,
    pub offset: u64,
    pub limit: u64,
    pub total: u64,
    pub has_more: bool,
    pub next_offset: Option<u64>,
}

/// Full source reconstruction (all chunks in index order + joined text).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceDocument {
    pub source: String,
    pub total: u64,
    pub chunks: Vec<SourceChunk>,
    /// Chunk texts joined with blank lines (empty when no chunks).
    pub text: String,
    pub source_type: Option<String>,
}

/// Parse `chunk_index` / `total_chunks` from a Qdrant-restored payload value.
///
/// Upsert coerces non-strings to `StringValue`, so numbers often return as JSON
/// strings (`"0"`). Accept integer, float-as-int, and decimal strings.
pub fn parse_chunk_index_value(v: &serde_json::Value) -> Option<u64> {
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    if let Some(n) = v.as_i64() {
        return if n >= 0 { Some(n as u64) } else { None };
    }
    if let Some(f) = v.as_f64() {
        if f.is_finite() && f >= 0.0 && f.fract() == 0.0 {
            return Some(f as u64);
        }
        return None;
    }
    if let Some(s) = v.as_str() {
        let t = s.trim();
        if let Ok(n) = t.parse::<u64>() {
            return Some(n);
        }
        if let Ok(f) = t.parse::<f64>()
            && f.is_finite()
            && f >= 0.0
            && f.fract() == 0.0
        {
            return Some(f as u64);
        }
    }
    None
}

/// Build a [`SourceChunk`] from a stored payload object.
pub fn source_chunk_from_payload(payload: Payload) -> SourceChunk {
    let text = payload_str(&payload, payload_schema::TEXT).unwrap_or_default();
    let chunk_index = payload
        .get(payload_schema::CHUNK_INDEX)
        .and_then(parse_chunk_index_value);
    let total_chunks = payload
        .get(payload_schema::TOTAL_CHUNKS)
        .and_then(parse_chunk_index_value);
    let source = payload_str(&payload, payload_schema::SOURCE);
    let source_type = payload_str(&payload, payload_schema::SOURCE_TYPE);
    SourceChunk {
        text,
        chunk_index,
        total_chunks,
        source,
        source_type,
        payload,
    }
}

/// Sort key for reconstruction: known indices ascending; missing index sorts last.
pub fn chunk_sort_key(chunk: &SourceChunk) -> u64 {
    chunk.chunk_index.unwrap_or(u64::MAX)
}

/// Sort chunks in place by `chunk_index` (missing last). Stable for equal keys.
pub fn sort_source_chunks(chunks: &mut [SourceChunk]) {
    chunks.sort_by(|a, b| {
        chunk_sort_key(a)
            .cmp(&chunk_sort_key(b))
            .then_with(|| a.text.cmp(&b.text))
    });
}

/// Sort then slice with search-style pagination (`limit=0` → empty page).
pub fn paginate_source_chunks(
    mut chunks: Vec<SourceChunk>,
    source: impl Into<String>,
    offset: u64,
    limit: u64,
) -> SourceChunkPage {
    let source = source.into();
    sort_source_chunks(&mut chunks);
    let total = chunks.len() as u64;
    let offset = offset.min(total);

    if limit == 0 {
        return SourceChunkPage {
            source,
            chunks: vec![],
            offset,
            limit: 0,
            total,
            has_more: offset < total,
            next_offset: if offset < total { Some(offset) } else { None },
        };
    }

    let start = offset as usize;
    let end = (offset.saturating_add(limit) as usize).min(chunks.len());
    let page: Vec<SourceChunk> = if start >= chunks.len() {
        vec![]
    } else {
        chunks[start..end].to_vec()
    };
    let next = offset.saturating_add(page.len() as u64);
    let has_more = next < total;
    SourceChunkPage {
        source,
        chunks: page,
        offset,
        limit,
        total,
        has_more,
        next_offset: if has_more { Some(next) } else { None },
    }
}

/// Chunks whose `chunk_index` lies in `[center - neighbors, center + neighbors]`.
///
/// Only chunks with a known index are included. Result is sorted by index.
/// Does not invent content outside the provided set.
pub fn expand_chunk_neighbors(
    chunks: &[SourceChunk],
    center_chunk_index: u64,
    neighbors: u64,
) -> Vec<SourceChunk> {
    let lo = center_chunk_index.saturating_sub(neighbors);
    let hi = center_chunk_index.saturating_add(neighbors);
    let mut out: Vec<SourceChunk> = chunks
        .iter()
        .filter(|c| c.chunk_index.map(|i| i >= lo && i <= hi).unwrap_or(false))
        .cloned()
        .collect();
    sort_source_chunks(&mut out);
    out
}

/// Build a full-source document from already-fetched payloads for one source.
pub fn source_document_from_chunks(
    source: impl Into<String>,
    mut chunks: Vec<SourceChunk>,
) -> SourceDocument {
    let source = source.into();
    sort_source_chunks(&mut chunks);
    let source_type = chunks.iter().find_map(|c| c.source_type.clone());
    let text = chunks
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let total = chunks.len() as u64;
    SourceDocument {
        source,
        total,
        chunks,
        text,
        source_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chunk(text: &str, index: Option<u64>) -> SourceChunk {
        SourceChunk {
            text: text.to_string(),
            chunk_index: index,
            total_chunks: Some(3),
            source: Some("docs/a.md".into()),
            source_type: Some("markdown".into()),
            payload: json!({
                "text": text,
                "source": "docs/a.md",
                "chunk_index": index.map(|i| i.to_string()).unwrap_or_else(|| "missing".into()),
            }),
        }
    }

    #[test]
    fn parse_chunk_index_accepts_number_and_string() {
        assert_eq!(parse_chunk_index_value(&json!(2)), Some(2));
        assert_eq!(parse_chunk_index_value(&json!("3")), Some(3));
        assert_eq!(parse_chunk_index_value(&json!("0")), Some(0));
        assert_eq!(parse_chunk_index_value(&json!(1.0)), Some(1));
        assert_eq!(parse_chunk_index_value(&json!("1.0")), Some(1));
        assert_eq!(parse_chunk_index_value(&json!("nope")), None);
        assert_eq!(parse_chunk_index_value(&json!(-1)), None);
        assert_eq!(parse_chunk_index_value(&json!(1.5)), None);
    }

    #[test]
    fn source_chunk_from_payload_reads_qdrant_string_coercion() {
        let payload = json!({
            "text": "hello",
            "source": "s://x",
            "source_type": "text",
            "chunk_index": "1",
            "total_chunks": "4",
        });
        let c = source_chunk_from_payload(payload);
        assert_eq!(c.text, "hello");
        assert_eq!(c.source.as_deref(), Some("s://x"));
        assert_eq!(c.source_type.as_deref(), Some("text"));
        assert_eq!(c.chunk_index, Some(1));
        assert_eq!(c.total_chunks, Some(4));
    }

    #[test]
    fn sort_missing_chunk_index_last() {
        let mut chunks = vec![
            chunk("c", Some(2)),
            chunk("a", None),
            chunk("b", Some(0)),
            chunk("d", Some(1)),
        ];
        sort_source_chunks(&mut chunks);
        assert_eq!(
            chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
            vec!["b", "d", "c", "a"]
        );
    }

    #[test]
    fn paginate_orders_and_slices() {
        let chunks = vec![
            chunk("two", Some(2)),
            chunk("zero", Some(0)),
            chunk("one", Some(1)),
            chunk("three", Some(3)),
        ];
        let page = paginate_source_chunks(chunks, "docs/a.md", 1, 2);
        assert_eq!(page.source, "docs/a.md");
        assert_eq!(page.total, 4);
        assert_eq!(page.offset, 1);
        assert_eq!(page.limit, 2);
        assert_eq!(page.chunks.len(), 2);
        assert_eq!(page.chunks[0].text, "one");
        assert_eq!(page.chunks[1].text, "two");
        assert!(page.has_more);
        assert_eq!(page.next_offset, Some(3));

        let last = paginate_source_chunks(
            vec![
                chunk("two", Some(2)),
                chunk("zero", Some(0)),
                chunk("one", Some(1)),
                chunk("three", Some(3)),
            ],
            "docs/a.md",
            3,
            2,
        );
        assert_eq!(last.chunks.len(), 1);
        assert_eq!(last.chunks[0].text, "three");
        assert!(!last.has_more);
        assert_eq!(last.next_offset, None);
    }

    #[test]
    fn paginate_limit_zero_empty_with_has_more() {
        let chunks = vec![chunk("a", Some(0)), chunk("b", Some(1))];
        let page = paginate_source_chunks(chunks, "s", 0, 0);
        assert!(page.chunks.is_empty());
        assert_eq!(page.total, 2);
        assert!(page.has_more);
        assert_eq!(page.next_offset, Some(0));
    }

    #[test]
    fn paginate_empty_source() {
        let page = paginate_source_chunks(vec![], "gone", 0, 10);
        assert!(page.chunks.is_empty());
        assert_eq!(page.total, 0);
        assert!(!page.has_more);
        assert_eq!(page.next_offset, None);
    }

    #[test]
    fn expand_neighbors_window_same_source_ordered() {
        let chunks = vec![
            chunk("0", Some(0)),
            chunk("1", Some(1)),
            chunk("2", Some(2)),
            chunk("3", Some(3)),
            chunk("4", Some(4)),
            chunk("no-idx", None),
        ];
        let win = expand_chunk_neighbors(&chunks, 2, 1);
        assert_eq!(
            win.iter().map(|c| c.chunk_index).collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3)]
        );
        assert!(win.iter().all(|c| c.text != "no-idx"));
        // Does not invent indices outside the set
        let edge = expand_chunk_neighbors(&chunks, 0, 2);
        assert_eq!(
            edge.iter()
                .map(|c| c.chunk_index.unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn expand_empty_when_center_missing() {
        let chunks = vec![chunk("0", Some(0)), chunk("1", Some(1))];
        let win = expand_chunk_neighbors(&chunks, 99, 1);
        assert!(win.is_empty());
    }

    #[test]
    fn source_document_joins_ordered_text() {
        let doc = source_document_from_chunks(
            "docs/a.md",
            vec![chunk("second", Some(1)), chunk("first", Some(0))],
        );
        assert_eq!(doc.source, "docs/a.md");
        assert_eq!(doc.total, 2);
        assert_eq!(doc.text, "first\n\nsecond");
        assert_eq!(doc.chunks[0].chunk_index, Some(0));
        assert_eq!(doc.source_type.as_deref(), Some("markdown"));
    }
}
