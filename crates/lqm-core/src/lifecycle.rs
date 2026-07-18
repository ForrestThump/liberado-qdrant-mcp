//! Document lifecycle helpers: re-ingest policy and pure decisions.
//!
//! Network/Qdrant I/O stays in `qdrant.rs`; this module is unit-testable offline.

use crate::types::ReingestAction;

/// Decide skip / replace / insert from existing vs new content hashes for one source.
///
/// Hashes are compared as sorted multisets so multi-chunk documents skip only when
/// every chunk hash matches an existing point for that source (order-independent).
pub fn decide_source_reingest(existing_hashes: &[String], new_hashes: &[String]) -> ReingestAction {
    if new_hashes.is_empty() {
        // Nothing to write — treat as skip of empty work.
        return ReingestAction::Skip;
    }
    if existing_hashes.is_empty() {
        return ReingestAction::Insert;
    }
    let mut existing: Vec<&str> = existing_hashes.iter().map(|s| s.as_str()).collect();
    let mut newh: Vec<&str> = new_hashes.iter().map(|s| s.as_str()).collect();
    existing.sort_unstable();
    newh.sort_unstable();
    if existing == newh {
        ReingestAction::Skip
    } else {
        ReingestAction::Replace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_existing_is_insert() {
        let new = vec!["aaa".to_string()];
        assert_eq!(decide_source_reingest(&[], &new), ReingestAction::Insert);
    }

    #[test]
    fn identical_hashes_skip() {
        let existing = vec!["h1".to_string(), "h2".to_string()];
        let new = vec!["h2".to_string(), "h1".to_string()]; // order independent
        assert_eq!(
            decide_source_reingest(&existing, &new),
            ReingestAction::Skip
        );
    }

    #[test]
    fn different_hashes_replace() {
        let existing = vec!["old".to_string()];
        let new = vec!["new".to_string()];
        assert_eq!(
            decide_source_reingest(&existing, &new),
            ReingestAction::Replace
        );
    }

    #[test]
    fn count_mismatch_replace() {
        let existing = vec!["h1".to_string(), "h1".to_string()];
        let new = vec!["h1".to_string()];
        assert_eq!(
            decide_source_reingest(&existing, &new),
            ReingestAction::Replace
        );
    }

    #[test]
    fn empty_new_is_skip() {
        assert_eq!(
            decide_source_reingest(&["x".to_string()], &[]),
            ReingestAction::Skip
        );
    }
}
