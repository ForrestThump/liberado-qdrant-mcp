//! Agent-friendly formatting of search results as LLM-ready context.

use crate::constants;
use crate::types::{ContextOptions, SearchResult, payload_str};
use serde::Serialize;
use std::collections::HashSet;

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
    /// True when max_total_chars stopped us before all candidates were included.
    pub truncated_by_budget: bool,
}

/// Format search results into numbered passages with source/citation metadata.
///
/// Empty results produce a short notice so agents never get a blank string.
/// `max_chars_per_passage` truncates long texts (0 / None = no truncation).
pub fn format_relevant_context(
    query: &str,
    results: &[SearchResult],
    max_chars_per_passage: Option<usize>,
) -> FormattedContext {
    format_relevant_context_with(
        query,
        results,
        &ContextOptions {
            max_chars_per_passage,
            ..Default::default()
        },
    )
}

/// Format with full context options (budget, optional MMR).
pub fn format_relevant_context_with(
    query: &str,
    results: &[SearchResult],
    opts: &ContextOptions,
) -> FormattedContext {
    let mut working: Vec<SearchResult> = results.to_vec();
    if opts.mmr && !working.is_empty() {
        let lambda = opts
            .mmr_lambda
            .unwrap_or(constants::DEFAULT_MMR_LAMBDA)
            .clamp(0.0, 1.0);
        let k = working.len();
        working = mmr_rerank(working, k, lambda);
    }

    let max_chars = opts.max_chars_per_passage.unwrap_or(0);
    let max_total = opts.max_total_chars.filter(|n| *n > 0);

    if working.is_empty() {
        return FormattedContext {
            context: format!(
                "# Relevant Context\n\n_No passages matched query: {}_\n\n\
                 _Tip: try a broader query, lower min_score, or clear source/tag filters._\n",
                escape_md_inline(query)
            ),
            sources: vec![],
            passage_count: 0,
            truncated_by_budget: false,
        };
    }

    let mut body = String::new();
    body.push_str("# Relevant Context\n\n");
    body.push_str(&format!("_Query: {}_\n\n", escape_md_inline(query)));

    let mut sources = Vec::new();
    let mut truncated_by_budget = false;
    let mut included = 0usize;

    for (i, result) in working.iter().enumerate() {
        let index = i + 1;
        let source = payload_str(&result.payload, payload_schema::SOURCE);
        let source_type = payload_str(&result.payload, payload_schema::SOURCE_TYPE);
        let project = payload_str(&result.payload, payload_schema::PROJECT);

        let passage = if max_chars > 0 && result.text.chars().count() > max_chars {
            let truncated: String = result.text.chars().take(max_chars).collect();
            format!("{truncated}…")
        } else {
            result.text.clone()
        };

        let mut block = String::new();
        block.push_str(&format!("## Passage {index}\n\n"));
        block.push_str(&format!("- **score**: {:.4}\n", result.score));
        if let Some(ref s) = source {
            block.push_str(&format!("- **source**: `{s}`\n"));
        }
        if let Some(ref st) = source_type {
            block.push_str(&format!("- **source_type**: `{st}`\n"));
        }
        if let Some(ref p) = project {
            block.push_str(&format!("- **project**: `{p}`\n"));
        }
        block.push('\n');
        block.push_str(&passage);
        block.push_str("\n\n");

        if let Some(budget) = max_total {
            // Always try to fit at least one passage; further ones respect the budget.
            if included > 0 && body.len() + block.len() > budget {
                truncated_by_budget = true;
                break;
            }
        }

        body.push_str(&block);
        included += 1;

        let preview: String = result
            .text
            .chars()
            .take(constants::TEXT_PREVIEW_CHARS)
            .collect();
        sources.push(ContextSource {
            index,
            score: result.score,
            source,
            source_type,
            text_preview: if result.text.chars().count() > constants::TEXT_PREVIEW_CHARS {
                format!("{preview}…")
            } else {
                preview
            },
        });
    }

    if truncated_by_budget {
        body.push_str("_…additional passages omitted to stay within max_total_chars budget._\n");
    }

    FormattedContext {
        context: body,
        sources,
        passage_count: included,
        truncated_by_budget,
    }
}

/// Maximal Marginal Relevance over text hits using score + token-set Jaccard.
///
/// Pure post-process: does not re-query Qdrant. `lambda` balances relevance (1.0)
/// vs diversity (0.0).
pub fn mmr_rerank(results: Vec<SearchResult>, k: usize, lambda: f32) -> Vec<SearchResult> {
    if results.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(results.len());
    let lambda = lambda.clamp(0.0, 1.0);

    let max_score = results
        .iter()
        .map(|r| r.score)
        .fold(0.0_f32, f32::max)
        .max(1e-6);

    let mut remaining: Vec<SearchResult> = results;
    let mut selected: Vec<SearchResult> = Vec::with_capacity(k);

    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = 0usize;
        let mut best_mmr = f32::NEG_INFINITY;

        for (i, cand) in remaining.iter().enumerate() {
            let rel = cand.score / max_score;
            let div = if selected.is_empty() {
                0.0
            } else {
                selected
                    .iter()
                    .map(|s| token_jaccard(&cand.text, &s.text))
                    .fold(0.0_f32, f32::max)
            };
            let mmr = lambda * rel - (1.0 - lambda) * div;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }
        selected.push(remaining.remove(best_idx));
    }
    selected
}

fn token_jaccard(a: &str, b: &str) -> f32 {
    let ta = tokenize(a);
    let tb = tokenize(b);
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count() as f32;
    let union = ta.union(&tb).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

fn tokenize(s: &str) -> HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_string())
        .collect()
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
        assert!(fmt.context.contains("broader query") || fmt.context.contains("min_score"));
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
        assert!(!fmt.context.contains(&long), "full passage leaked");
        assert!(
            fmt.context.contains(&format!("{}…", "x".repeat(50))),
            "expected truncated run in context: {}",
            fmt.context
        );
    }

    #[test]
    fn format_respects_total_char_budget() {
        let results = vec![
            hit(
                "AAAA passage one content that is moderately long.",
                0.9,
                "a",
                "text",
            ),
            hit(
                "BBBB passage two content that is moderately long.",
                0.8,
                "b",
                "text",
            ),
            hit(
                "CCCC passage three content that is moderately long.",
                0.7,
                "c",
                "text",
            ),
        ];
        // Small budget: should not fit all three full passages.
        let fmt = format_relevant_context_with(
            "q",
            &results,
            &ContextOptions {
                max_total_chars: Some(280),
                ..Default::default()
            },
        );
        assert!(fmt.passage_count >= 1);
        assert!(fmt.passage_count < 3 || fmt.truncated_by_budget);
        if fmt.truncated_by_budget {
            assert!(fmt.context.contains("omitted") || fmt.context.contains("budget"));
        }
    }

    #[test]
    fn mmr_prefers_diverse_texts() {
        let results = vec![
            hit("cats cats cats feline animals", 1.0, "1", "text"),
            hit("cats cats cats more felines", 0.99, "2", "text"),
            hit("quantum computing qubits gates", 0.5, "3", "text"),
        ];
        // Low lambda → diversity; quantum doc should enter selection early.
        let reranked = mmr_rerank(results, 2, 0.2);
        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].payload["source"], "1");
        assert_eq!(
            reranked[1].payload["source"],
            "3",
            "diverse hit should beat near-duplicate: {:?}",
            reranked
                .iter()
                .map(|r| &r.payload["source"])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn mmr_edge_k_zero_and_empty() {
        let results = vec![hit("a", 1.0, "1", "text")];
        assert!(mmr_rerank(results.clone(), 0, 0.5).is_empty());
        assert!(mmr_rerank(vec![], 3, 0.5).is_empty());
    }

    #[test]
    fn mmr_edge_k_larger_than_input() {
        let results = vec![hit("a", 1.0, "1", "text"), hit("b", 0.5, "2", "text")];
        let out = mmr_rerank(results, 10, 0.7);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn mmr_lambda_extremes() {
        let results = vec![
            hit("cats cats cats", 1.0, "1", "text"),
            hit("cats cats more", 0.9, "2", "text"),
            hit("quantum computing", 0.4, "3", "text"),
        ];
        // lambda=1.0 → pure relevance order (top-k by score).
        let pure_rel = mmr_rerank(results.clone(), 2, 1.0);
        assert_eq!(pure_rel[0].payload["source"], "1");
        assert_eq!(pure_rel[1].payload["source"], "2");
        // lambda=0.0 → pure diversity after first (highest score) pick.
        let pure_div = mmr_rerank(results, 2, 0.0);
        assert_eq!(pure_div.len(), 2);
        assert_eq!(pure_div[0].payload["source"], "1");
    }

    #[test]
    fn search_filter_empty_detection() {
        use crate::types::SearchFilter;
        assert!(SearchFilter::default().is_empty());
        assert!(
            !SearchFilter {
                source: Some("x".into()),
                ..Default::default()
            }
            .is_empty()
        );
    }
}
