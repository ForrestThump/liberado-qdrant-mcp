//! Agent-friendly formatting of search results as LLM-ready context.

use crate::types::SearchResult;
use serde::Serialize;

/// Structured companion to the markdown context string.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ContextSource {
    pub index: usize,
    pub score: f32,
    pub source: Option<String>,
    pub source_type: Option<String>,
    pub text_preview: String,
}

/// Result of formatting search hits for agent consumption.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FormattedContext {
    /// Markdown-style context with numbered passages, citations, and scores.
    pub context: String,
    /// Compact structured list of sources (for tools that prefer JSON).
    pub sources: Vec<ContextSource>,
    pub passage_count: usize,
}

/// Format search results into numbered passages with source/citation metadata.
///
/// Empty results produce a short notice so agents never get a blank string.
/// `max_chars_per_passage` truncates long texts (0 = no truncation).
pub fn format_relevant_context(
    query: &str,
    results: &[SearchResult],
    max_chars_per_passage: Option<usize>,
) -> FormattedContext {
    let max_chars = max_chars_per_passage.unwrap_or(0);
    if results.is_empty() {
        return FormattedContext {
            context: format!(
                "# Relevant Context\n\n_No passages matched query: {}_\n",
                escape_md_inline(query)
            ),
            sources: vec![],
            passage_count: 0,
        };
    }

    let mut body = String::new();
    body.push_str("# Relevant Context\n\n");
    body.push_str(&format!("_Query: {}_\n\n", escape_md_inline(query)));

    let mut sources = Vec::with_capacity(results.len());

    for (i, result) in results.iter().enumerate() {
        let index = i + 1;
        let source = payload_str(&result.payload, "source");
        let source_type = payload_str(&result.payload, "source_type");
        let project = payload_str(&result.payload, "project");

        let passage = if max_chars > 0 && result.text.chars().count() > max_chars {
            let truncated: String = result.text.chars().take(max_chars).collect();
            format!("{truncated}…")
        } else {
            result.text.clone()
        };

        body.push_str(&format!("## Passage {index}\n\n"));
        body.push_str(&format!("- **score**: {:.4}\n", result.score));
        if let Some(ref s) = source {
            body.push_str(&format!("- **source**: `{s}`\n"));
        }
        if let Some(ref st) = source_type {
            body.push_str(&format!("- **source_type**: `{st}`\n"));
        }
        if let Some(ref p) = project {
            body.push_str(&format!("- **project**: `{p}`\n"));
        }
        body.push('\n');
        body.push_str(&passage);
        body.push_str("\n\n");

        let preview: String = result.text.chars().take(160).collect();
        sources.push(ContextSource {
            index,
            score: result.score,
            source,
            source_type,
            text_preview: if result.text.chars().count() > 160 {
                format!("{preview}…")
            } else {
                preview
            },
        });
    }

    FormattedContext {
        context: body,
        sources,
        passage_count: results.len(),
    }
}

fn payload_str(payload: &serde_json::Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn escape_md_inline(s: &str) -> String {
    // Keep query readable in italics; strip characters that would break the line badly.
    s.replace('\n', " ").replace('*', "\\*")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hit(text: &str, score: f32, source: &str, source_type: &str) -> SearchResult {
        SearchResult {
            text: text.to_string(),
            score,
            payload: json!({
                "text": text,
                "source": source,
                "source_type": source_type,
            }),
        }
    }

    #[test]
    fn format_empty_results() {
        let fmt = format_relevant_context("rust ownership", &[], None);
        assert_eq!(fmt.passage_count, 0);
        assert!(fmt.sources.is_empty());
        assert!(fmt.context.contains("No passages matched"));
        assert!(fmt.context.contains("rust ownership"));
    }

    #[test]
    fn format_numbered_passages_with_scores_and_sources() {
        let results = vec![
            hit(
                "First passage about vectors.",
                0.91,
                "https://ex.com/a",
                "webpage",
            ),
            hit(
                "Second passage about Qdrant.",
                0.75,
                "docs/qdrant.md",
                "text",
            ),
        ];
        let fmt = format_relevant_context("vector search", &results, None);
        assert_eq!(fmt.passage_count, 2);
        assert_eq!(fmt.sources.len(), 2);
        assert!(fmt.context.contains("# Relevant Context"));
        assert!(fmt.context.contains("## Passage 1"));
        assert!(fmt.context.contains("## Passage 2"));
        assert!(fmt.context.contains("0.9100"));
        assert!(fmt.context.contains("0.7500"));
        assert!(fmt.context.contains("`https://ex.com/a`"));
        assert!(fmt.context.contains("`docs/qdrant.md`"));
        assert!(fmt.context.contains("First passage about vectors."));
        assert!(fmt.context.contains("Second passage about Qdrant."));
        assert_eq!(fmt.sources[0].index, 1);
        assert!((fmt.sources[0].score - 0.91).abs() < f32::EPSILON);
        assert_eq!(fmt.sources[0].source.as_deref(), Some("https://ex.com/a"));
    }

    #[test]
    fn format_truncates_when_max_chars_set() {
        let long = "x".repeat(500);
        let results = vec![hit(&long, 0.5, "s", "text")];
        let fmt = format_relevant_context("q", &results, Some(50));
        assert!(
            fmt.context.contains('…'),
            "expected ellipsis: {}",
            fmt.context
        );
        // Full 500-char run must not appear; truncated body is 50 x's + ellipsis.
        assert!(!fmt.context.contains(&long), "full passage leaked");
        assert!(
            fmt.context.contains(&format!("{}…", "x".repeat(50))),
            "expected truncated run in context: {}",
            fmt.context
        );
    }
}
