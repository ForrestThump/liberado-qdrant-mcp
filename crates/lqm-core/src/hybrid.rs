//! Hybrid dense + keyword retrieval helpers (pure, unit-tested offline).
//!
//! Dense vector search can miss exact rare tokens. Hybrid mode over-fetches
//! dense hits, merges keyword candidates from a configurable backend, and fuses
//! scores (weighted + RRF). Keyword backends:
//! - [`HybridKeywordBackend::KeywordIndex`] — payload text index (default)
//! - [`HybridKeywordBackend::Sparse`] — native Qdrant sparse vectors
//! - [`HybridKeywordBackend::Scroll`] — legacy full-collection payload scroll

use crate::constants;
use crate::types::{Payload, SearchResult};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// Default reciprocal-rank fusion constant (standard RRF uses k=60).
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Default dense weight when hybrid is on (`score = α·dense + (1-α)·keyword`).
pub const DEFAULT_HYBRID_ALPHA: f32 = 0.6;

/// How many dense hits to over-fetch before hybrid fusion (multiplied by limit).
pub const HYBRID_DENSE_OVERFETCH: u64 = 5;

/// Cap on dense over-fetch for hybrid candidate pool.
pub const HYBRID_DENSE_OVERFETCH_CAP: u64 = 80;

/// How hybrid obtains keyword candidates when `hybrid=true`.
///
/// Selected via `LQM_HYBRID_KEYWORD_BACKEND` / [`crate::types::RagConfig`].
/// Dense-only search (`hybrid=false`) ignores this setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HybridKeywordBackend {
    /// Full-collection payload scroll + client-side token scoring (legacy O(n)).
    Scroll,
    /// Native Qdrant sparse-vector ANN for lexical candidates.
    Sparse,
    /// Payload full-text index on `text` (`MatchTextAny`); index-backed, not full scroll.
    #[default]
    KeywordIndex,
}

impl HybridKeywordBackend {
    pub const ALL: &'static [HybridKeywordBackend] = &[
        HybridKeywordBackend::Scroll,
        HybridKeywordBackend::Sparse,
        HybridKeywordBackend::KeywordIndex,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            HybridKeywordBackend::Scroll => "scroll",
            HybridKeywordBackend::Sparse => "sparse",
            HybridKeywordBackend::KeywordIndex => "keyword_index",
        }
    }

    /// Whether new collections should be created with a sparse vector schema.
    pub fn needs_sparse_schema(self) -> bool {
        matches!(self, HybridKeywordBackend::Sparse)
    }

    /// Whether a full-text payload index on `text` should be ensured.
    pub fn needs_text_index(self) -> bool {
        matches!(
            self,
            HybridKeywordBackend::KeywordIndex | HybridKeywordBackend::Scroll
        )
    }
}

impl fmt::Display for HybridKeywordBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for HybridKeywordBackend {
    type Err = UnknownHybridKeywordBackend;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "scroll" | "legacy" | "o_n" | "on" => Ok(HybridKeywordBackend::Scroll),
            "sparse" | "sparse_vector" | "sparse_vectors" => Ok(HybridKeywordBackend::Sparse),
            "keyword_index" | "keyword-index" | "text_index" | "text" | "index" => {
                Ok(HybridKeywordBackend::KeywordIndex)
            }
            other => Err(UnknownHybridKeywordBackend(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownHybridKeywordBackend(pub String);

impl fmt::Display for UnknownHybridKeywordBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown hybrid keyword backend '{}'; expected one of: {}",
            self.0,
            HybridKeywordBackend::ALL
                .iter()
                .map(|b| b.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl std::error::Error for UnknownHybridKeywordBackend {}

/// Resolve backend from env (`LQM_HYBRID_KEYWORD_BACKEND`), defaulting to keyword_index.
pub fn hybrid_keyword_backend_from_env() -> HybridKeywordBackend {
    match std::env::var(constants::ENV_HYBRID_KEYWORD_BACKEND) {
        Ok(raw) => raw
            .parse()
            .unwrap_or_else(|e: UnknownHybridKeywordBackend| {
                log::warn!("{e}; using keyword_index");
                HybridKeywordBackend::KeywordIndex
            }),
        Err(_) => HybridKeywordBackend::KeywordIndex,
    }
}

/// Sparse bag-of-words encoding (indices + values) for Qdrant sparse vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseEncoding {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseEncoding {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.values.is_empty()
    }
}

/// Deterministic FNV-1a token hash mapped into [`constants::SPARSE_HASH_MODULUS`].
pub fn hash_token(token: &str) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for b in token.as_bytes() {
        h ^= u32::from(*b);
        h = h.wrapping_mul(16_777_619);
    }
    h % constants::SPARSE_HASH_MODULUS
}

/// Encode text as L2-normalized term-frequency sparse vector (pure, offline).
///
/// Tokens use [`tokenize_for_keyword`]. Colliding hashes sum TF before normalize.
/// Caps at [`constants::SPARSE_MAX_DIMS`] highest-weight dims when needed.
pub fn encode_sparse_tf(text: &str) -> SparseEncoding {
    let tokens = tokenize_for_keyword(text);
    if tokens.is_empty() {
        return SparseEncoding {
            indices: vec![],
            values: vec![],
        };
    }
    let mut counts: HashMap<u32, f32> = HashMap::new();
    for t in tokens {
        let idx = hash_token(&t);
        *counts.entry(idx).or_insert(0.0) += 1.0;
    }
    let mut pairs: Vec<(u32, f32)> = counts.into_iter().collect();
    if pairs.len() > constants::SPARSE_MAX_DIMS {
        pairs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        pairs.truncate(constants::SPARSE_MAX_DIMS);
    }
    pairs.sort_by_key(|(i, _)| *i);
    let norm = pairs
        .iter()
        .map(|(_, v)| v * v)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    SparseEncoding {
        indices: pairs.iter().map(|(i, _)| *i).collect(),
        values: pairs.iter().map(|(_, v)| v / norm).collect(),
    }
}

/// Join tokens for Qdrant `MatchTextAny` / text-index queries.
pub fn text_index_query(tokens: &[String]) -> String {
    tokens.join(" ")
}

/// Build keyword candidates from (text, payload) pairs with score > 0.
///
/// Pure helper shared by scroll and text-index paths after payloads are fetched.
pub fn keyword_candidates_from_payloads(
    payloads: impl IntoIterator<Item = Payload>,
    query_tokens: &[String],
) -> Vec<SearchResult> {
    if query_tokens.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for payload in payloads {
        let text = payload
            .get(crate::types::payload_schema::TEXT)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if text.is_empty() {
            continue;
        }
        let kw = keyword_score(&text, query_tokens);
        if kw > 0.0 {
            out.push(SearchResult {
                text,
                score: 0.0,
                payload,
            });
        }
    }
    out
}

/// Lowercase alphanumeric tokens of length ≥ 2 (ASCII-oriented, deterministic).
pub fn tokenize_for_keyword(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if cur.len() >= 2 {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

/// Keyword relevance in [0, 1]: fraction of unique query tokens found in `text`,
/// with a small boost when the full lowercased query substring appears.
pub fn keyword_score(text: &str, query_tokens: &[String]) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let text_tokens: std::collections::HashSet<String> =
        tokenize_for_keyword(text).into_iter().collect();
    let unique_q: std::collections::HashSet<&String> = query_tokens.iter().collect();
    if unique_q.is_empty() {
        return 0.0;
    }
    let hits = unique_q
        .iter()
        .filter(|t| text_tokens.contains(t.as_str()))
        .count();
    let mut score = hits as f32 / unique_q.len() as f32;

    let q_joined: String = query_tokens.join(" ");
    if !q_joined.is_empty() {
        let hay = text.to_ascii_lowercase();
        if hay.contains(&q_joined) {
            score = (score + constants::KEYWORD_PHRASE_BONUS).min(1.0);
        }
    }
    score.clamp(0.0, 1.0)
}

/// Reciprocal rank fusion contribution for 0-based rank.
pub fn rrf_contribution(rank: usize, k: f32) -> f32 {
    let k = k.max(1.0);
    1.0 / (k + rank as f32)
}

/// Stable identity for deduping candidates (prefer ingest_hash, else text).
pub fn result_identity(r: &SearchResult) -> String {
    r.payload
        .get("ingest_hash")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| r.text.clone())
}

/// Normalize scores to [0, 1] by max in the list (empty → empty).
pub fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(0.0_f32, f32::max);
    if max <= f32::EPSILON {
        return scores.iter().map(|_| 0.0).collect();
    }
    scores.iter().map(|s| (s / max).clamp(0.0, 1.0)).collect()
}

/// Fuse dense-ordered hits with keyword ranking.
///
/// - Builds keyword scores from each hit's `text` vs `query`.
/// - Combines **weighted normalized scores** and **RRF** so pure keyword boosts
///   affect order even when dense scores are tied (e.g. FakeEmbedder zeros).
///
/// `alpha`: weight of dense signal in [0, 1] (keyword weight = 1 - alpha).
/// `rrf_k`: RRF constant (default [`DEFAULT_RRF_K`]).
///
/// Returns a new `Vec` sorted by fused score descending; each `score` is the fused value.
pub fn fuse_dense_keyword(
    dense_results: &[SearchResult],
    query: &str,
    alpha: f32,
    rrf_k: f32,
) -> Vec<SearchResult> {
    if dense_results.is_empty() {
        return Vec::new();
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let tokens = tokenize_for_keyword(query);

    let dense_scores: Vec<f32> = dense_results.iter().map(|r| r.score).collect();
    let dense_norm = normalize_scores(&dense_scores);

    let kw_scores: Vec<f32> = dense_results
        .iter()
        .map(|r| keyword_score(&r.text, &tokens))
        .collect();
    let kw_norm = normalize_scores(&kw_scores);

    // Dense rank: preserve input order (caller should pass score-desc dense hits).
    // Keyword rank: sort indices by kw_score desc, then original index for stability.
    let mut kw_order: Vec<usize> = (0..dense_results.len()).collect();
    kw_order.sort_by(|&a, &b| {
        kw_scores[b]
            .partial_cmp(&kw_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut kw_rank = vec![0usize; dense_results.len()];
    for (rank, &idx) in kw_order.iter().enumerate() {
        kw_rank[idx] = rank;
    }

    let mut fused: Vec<(usize, f32)> = (0..dense_results.len())
        .map(|i| {
            let weighted = alpha * dense_norm[i] + (1.0 - alpha) * kw_norm[i];
            let rrf = alpha * rrf_contribution(i, rrf_k)
                + (1.0 - alpha) * rrf_contribution(kw_rank[i], rrf_k);
            let score =
                constants::HYBRID_FUSE_WEIGHTED * weighted + constants::HYBRID_FUSE_RRF * rrf;
            (i, score)
        })
        .collect();

    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    fused
        .into_iter()
        .map(|(i, score)| {
            let mut r = dense_results[i].clone();
            r.score = score;
            r
        })
        .collect()
}

/// Merge dense hits with extra keyword candidates (e.g. from scroll), then fuse.
///
/// Extra candidates keep their given `score` as the dense component (often 0).
/// Dedupes by [`result_identity`]; prefers the dense entry when both exist.
pub fn merge_and_fuse_hybrid(
    dense_results: &[SearchResult],
    keyword_candidates: &[SearchResult],
    query: &str,
    alpha: f32,
    rrf_k: f32,
) -> Vec<SearchResult> {
    let mut by_id: HashMap<String, SearchResult> = HashMap::new();
    for r in dense_results {
        by_id.insert(result_identity(r), r.clone());
    }
    for r in keyword_candidates {
        let id = result_identity(r);
        by_id.entry(id).or_insert_with(|| r.clone());
    }
    // Dense-first order for RRF dense ranks: dense list order, then extras by keyword.
    let mut pool: Vec<SearchResult> = dense_results.to_vec();
    let dense_ids: std::collections::HashSet<String> =
        dense_results.iter().map(result_identity).collect();
    let tokens = tokenize_for_keyword(query);
    let mut extras: Vec<SearchResult> = keyword_candidates
        .iter()
        .filter(|r| !dense_ids.contains(&result_identity(r)))
        .cloned()
        .collect();
    extras.sort_by(|a, b| {
        keyword_score(&b.text, &tokens)
            .partial_cmp(&keyword_score(&a.text, &tokens))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pool.extend(extras);
    fuse_dense_keyword(&pool, query, alpha, rrf_k)
}

/// Suggested dense fetch size when hybrid is enabled (before fusion/truncate).
pub fn hybrid_dense_fetch_limit(requested_limit: u64) -> u64 {
    let base = requested_limit
        .saturating_mul(HYBRID_DENSE_OVERFETCH)
        .max(requested_limit);
    base.clamp(1, HYBRID_DENSE_OVERFETCH_CAP)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hit(text: &str, score: f32, hash: &str) -> SearchResult {
        SearchResult {
            text: text.into(),
            score,
            payload: json!({ "text": text, "ingest_hash": hash }),
        }
    }

    #[test]
    fn tokenize_splits_and_lowercases() {
        assert_eq!(
            tokenize_for_keyword("Hello, WORLD! x a12"),
            vec!["hello", "world", "a12"]
        );
        assert!(tokenize_for_keyword("").is_empty());
        assert!(tokenize_for_keyword("a b").is_empty()); // single-char dropped
    }

    #[test]
    fn keyword_score_prefers_token_overlap() {
        let q = tokenize_for_keyword("zyxwv unique token");
        let high = keyword_score("doc with zyxwv unique token present", &q);
        let low = keyword_score("unrelated fluffy cloud material", &q);
        assert!(high > low, "high={high} low={low}");
        assert!(high > 0.9, "full overlap should be high: {high}");
        assert_eq!(keyword_score("anything", &[]), 0.0);
    }

    #[test]
    fn keyword_score_phrase_boost() {
        let q = tokenize_for_keyword("alpha beta");
        let with_phrase = keyword_score("prefix alpha beta suffix", &q);
        let tokens_only = keyword_score("beta somewhere and alpha elsewhere", &q);
        assert!(
            with_phrase >= tokens_only,
            "phrase={with_phrase} tokens={tokens_only}"
        );
    }

    #[test]
    fn rrf_higher_rank_scores_higher() {
        assert!(rrf_contribution(0, 60.0) > rrf_contribution(5, 60.0));
    }

    #[test]
    fn fuse_boosts_keyword_when_dense_tied() {
        // Equal dense scores: keyword must decide order.
        let dense = vec![
            hit("generic weather discussion about rain", 0.9, "h1"),
            hit("contains rarekeytoken for hybrid smoke", 0.9, "h2"),
        ];
        let fused = fuse_dense_keyword(&dense, "rarekeytoken hybrid", 0.5, DEFAULT_RRF_K);
        assert_eq!(fused.len(), 2);
        assert!(
            fused[0].text.contains("rarekeytoken"),
            "keyword-bearing chunk should rank first: {:?}",
            fused.iter().map(|r| &r.text).collect::<Vec<_>>()
        );
        assert!(fused[0].score >= fused[1].score);
    }

    #[test]
    fn fuse_alpha_one_preserves_dense_order() {
        let dense = vec![
            hit("first dense", 0.95, "a"),
            hit("second rarekeytoken", 0.5, "b"),
        ];
        let fused = fuse_dense_keyword(&dense, "rarekeytoken", 1.0, DEFAULT_RRF_K);
        assert_eq!(fused[0].text, "first dense");
    }

    #[test]
    fn fuse_alpha_zero_prefers_keyword() {
        let dense = vec![
            hit("semantically top but no match", 0.99, "a"),
            hit("bottom dense with rarekeytoken xyz", 0.1, "b"),
        ];
        let fused = fuse_dense_keyword(&dense, "rarekeytoken", 0.0, DEFAULT_RRF_K);
        assert!(
            fused[0].text.contains("rarekeytoken"),
            "pure keyword should promote rare token: {:?}",
            fused[0].text
        );
    }

    #[test]
    fn fuse_empty_and_no_panic() {
        assert!(fuse_dense_keyword(&[], "q", 0.6, 60.0).is_empty());
        let one = vec![hit("solo", 0.5, "s")];
        let fused = fuse_dense_keyword(&one, "", 0.6, 60.0);
        assert_eq!(fused.len(), 1);
    }

    #[test]
    fn merge_and_fuse_includes_keyword_only_candidate() {
        let dense = vec![hit("only semantic fluff about clouds", 0.9, "d1")];
        let extra = vec![hit("scroll candidate with rarekeytoken999", 0.0, "k1")];
        let fused = merge_and_fuse_hybrid(&dense, &extra, "rarekeytoken999", 0.3, DEFAULT_RRF_K);
        assert_eq!(fused.len(), 2);
        assert!(
            fused.iter().any(|r| r.text.contains("rarekeytoken999")),
            "keyword-only candidate must survive merge: {fused:?}"
        );
        // With low alpha, keyword candidate should rank high.
        assert!(
            fused[0].text.contains("rarekeytoken999"),
            "expected keyword hit first: {}",
            fused[0].text
        );
    }

    #[test]
    fn hybrid_dense_fetch_limit_scales_and_caps() {
        assert_eq!(hybrid_dense_fetch_limit(0), 1);
        assert_eq!(hybrid_dense_fetch_limit(10), 50);
        assert_eq!(hybrid_dense_fetch_limit(100), HYBRID_DENSE_OVERFETCH_CAP);
    }

    #[test]
    fn hybrid_keyword_backend_parse_aliases() {
        assert_eq!(
            "keyword_index".parse::<HybridKeywordBackend>().unwrap(),
            HybridKeywordBackend::KeywordIndex
        );
        assert_eq!(
            "text_index".parse::<HybridKeywordBackend>().unwrap(),
            HybridKeywordBackend::KeywordIndex
        );
        assert_eq!(
            "SPARSE".parse::<HybridKeywordBackend>().unwrap(),
            HybridKeywordBackend::Sparse
        );
        assert_eq!(
            "scroll".parse::<HybridKeywordBackend>().unwrap(),
            HybridKeywordBackend::Scroll
        );
        assert_eq!(
            "legacy".parse::<HybridKeywordBackend>().unwrap(),
            HybridKeywordBackend::Scroll
        );
        assert!("nope".parse::<HybridKeywordBackend>().is_err());
        assert!(HybridKeywordBackend::Sparse.needs_sparse_schema());
        assert!(!HybridKeywordBackend::KeywordIndex.needs_sparse_schema());
        assert!(HybridKeywordBackend::KeywordIndex.needs_text_index());
    }

    #[test]
    fn encode_sparse_tf_stable_and_nonempty() {
        let a = encode_sparse_tf("rarekeytokenxqzv hybrid smoke");
        let b = encode_sparse_tf("rarekeytokenxqzv hybrid smoke");
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.values, b.values);
        assert!(!a.is_empty());
        assert_eq!(a.indices.len(), a.values.len());
        // indices sorted ascending
        for w in a.indices.windows(2) {
            assert!(w[0] < w[1]);
        }
        // L2 ≈ 1
        let n: f32 = a.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4, "norm={n}");
        assert!(encode_sparse_tf("").is_empty());
        assert!(encode_sparse_tf("a b").is_empty()); // single-char tokens dropped
    }

    #[test]
    fn encode_sparse_prefers_shared_token_indices() {
        let shared = "sharedtoken";
        let e1 = encode_sparse_tf(&format!("{shared} alpha"));
        let e2 = encode_sparse_tf(&format!("{shared} beta"));
        let h = hash_token(shared);
        assert!(e1.indices.contains(&h));
        assert!(e2.indices.contains(&h));
    }

    #[test]
    fn text_index_query_joins_tokens() {
        let t = tokenize_for_keyword("Hello World");
        assert_eq!(text_index_query(&t), "hello world");
    }

    #[test]
    fn keyword_candidates_from_payloads_filters_non_matches() {
        let payloads = vec![
            json!({ "text": "generic weather discussion", "ingest_hash": "a" }),
            json!({ "text": "doc with rarekeytokenxqzv inside", "ingest_hash": "b" }),
            json!({ "text": "", "ingest_hash": "c" }),
        ];
        let tokens = tokenize_for_keyword("rarekeytokenxqzv");
        let cands = keyword_candidates_from_payloads(payloads, &tokens);
        assert_eq!(cands.len(), 1);
        assert!(cands[0].text.contains("rarekeytokenxqzv"));
        assert!(keyword_candidates_from_payloads(vec![json!({"text":"x"})], &[]).is_empty());
    }

    #[test]
    fn merge_and_fuse_works_for_sparse_style_candidates() {
        // Sparse backend also feeds SearchResult candidates into the same fuse path.
        let dense = vec![hit("semantic fluff about clouds", 0.95, "d1")];
        let sparse_hit = vec![hit("sparse candidate rarekeytokenabc", 0.0, "s1")];
        let fused =
            merge_and_fuse_hybrid(&dense, &sparse_hit, "rarekeytokenabc", 0.3, DEFAULT_RRF_K);
        assert!(
            fused.iter().any(|r| r.text.contains("rarekeytokenabc")),
            "{fused:?}"
        );
    }
}
