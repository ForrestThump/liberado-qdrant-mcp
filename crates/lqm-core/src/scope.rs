//! Clearance-safe **scoped filtering** helpers (pure, unit-tested offline).
//!
//! Agents isolate knowledge within a shared collection using:
//! - **`scope`** — exact partition key (e.g. `team-a`, `personal`). When a scope
//!   constraint is set, only points with that payload value match.
//! - **`clearance`** — ordinal sensitivity (`public` < `internal` < `confidential`
//!   < `restricted`). A `max_clearance` constraint admits points at or below that
//!   level. Missing clearance is treated as `public`.
//!
//! This is **payload isolation for agents**, not multi-user auth.

/// Ordered clearance levels (index = rank). Lower rank = less sensitive.
pub const CLEARANCE_LEVELS: &[&str] = &["public", "internal", "confidential", "restricted"];

/// Default clearance when ingest omits the field.
pub const DEFAULT_CLEARANCE: &str = "public";

/// Rank of a clearance level (`0` = public). Unknown levels return `None`.
pub fn clearance_rank(level: &str) -> Option<u8> {
    let n = level.trim().to_ascii_lowercase();
    CLEARANCE_LEVELS
        .iter()
        .position(|l| *l == n)
        .map(|i| i as u8)
}

/// Normalize a clearance string to a known level name, or `None` if invalid.
pub fn normalize_clearance(level: &str) -> Option<&'static str> {
    let rank = clearance_rank(level)?;
    Some(CLEARANCE_LEVELS[rank as usize])
}

/// Whether a point's clearance is allowed under `max_clearance`.
///
/// Missing / empty point clearance is treated as [`DEFAULT_CLEARANCE`].
/// Unknown `max_clearance` rejects everything (strict).
pub fn clearance_allowed(point_clearance: Option<&str>, max_clearance: &str) -> bool {
    let Some(max_rank) = clearance_rank(max_clearance) else {
        return false;
    };
    let point_rank = point_clearance
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(clearance_rank)
        .unwrap_or_else(|| clearance_rank(DEFAULT_CLEARANCE).unwrap_or(0));
    point_rank <= max_rank
}

/// Levels permitted when filtering with `max_clearance` (inclusive).
///
/// Empty if `max_clearance` is unknown.
pub fn allowed_clearance_levels(max_clearance: &str) -> Vec<&'static str> {
    let Some(max_rank) = clearance_rank(max_clearance) else {
        return Vec::new();
    };
    CLEARANCE_LEVELS[..=max_rank as usize].to_vec()
}

/// Exact scope match when a constraint is set.
///
/// - Constraint empty / whitespace → always matches (no constraint).
/// - Point missing scope → does **not** match a non-empty constraint.
pub fn scope_matches(point_scope: Option<&str>, required_scope: &str) -> bool {
    let req = required_scope.trim();
    if req.is_empty() {
        return true;
    }
    match point_scope.map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => s == req,
        None => false,
    }
}

/// Combined pure check used by tests and optional post-filters.
pub fn point_in_scope(
    point_scope: Option<&str>,
    point_clearance: Option<&str>,
    required_scope: Option<&str>,
    max_clearance: Option<&str>,
) -> bool {
    if let Some(scope) = required_scope
        && !scope_matches(point_scope, scope)
    {
        return false;
    }
    if let Some(max) = max_clearance {
        let max = max.trim();
        if !max.is_empty() && !clearance_allowed(point_clearance, max) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clearance_ranks_are_ordered() {
        assert!(clearance_rank("public").unwrap() < clearance_rank("internal").unwrap());
        assert!(clearance_rank("internal").unwrap() < clearance_rank("confidential").unwrap());
        assert!(clearance_rank("confidential").unwrap() < clearance_rank("restricted").unwrap());
        assert_eq!(clearance_rank("PUBLIC"), Some(0));
        assert_eq!(clearance_rank("nope"), None);
    }

    #[test]
    fn clearance_allowed_includes_lower_and_equal() {
        assert!(clearance_allowed(Some("public"), "internal"));
        assert!(clearance_allowed(Some("internal"), "internal"));
        assert!(!clearance_allowed(Some("confidential"), "internal"));
        assert!(!clearance_allowed(Some("restricted"), "public"));
        // Missing point clearance → public
        assert!(clearance_allowed(None, "public"));
        assert!(clearance_allowed(Some(""), "public"));
        assert!(!clearance_allowed(Some("internal"), "public"));
        // Unknown max rejects
        assert!(!clearance_allowed(Some("public"), "top-secret"));
    }

    #[test]
    fn allowed_clearance_levels_expand() {
        assert_eq!(allowed_clearance_levels("public"), vec!["public"]);
        assert_eq!(
            allowed_clearance_levels("confidential"),
            vec!["public", "internal", "confidential"]
        );
        assert!(allowed_clearance_levels("unknown").is_empty());
    }

    #[test]
    fn scope_matches_exact_and_excludes_other() {
        assert!(scope_matches(Some("team-a"), "team-a"));
        assert!(!scope_matches(Some("team-b"), "team-a"));
        assert!(!scope_matches(None, "team-a"));
        assert!(!scope_matches(Some(""), "team-a"));
        assert!(scope_matches(Some("anything"), ""));
        assert!(scope_matches(None, "  "));
    }

    #[test]
    fn point_in_scope_combines_rules() {
        assert!(point_in_scope(
            Some("alpha"),
            Some("internal"),
            Some("alpha"),
            Some("confidential")
        ));
        assert!(!point_in_scope(
            Some("beta"),
            Some("public"),
            Some("alpha"),
            Some("restricted")
        ));
        assert!(!point_in_scope(
            Some("alpha"),
            Some("restricted"),
            Some("alpha"),
            Some("internal")
        ));
        // No constraints → open
        assert!(point_in_scope(Some("x"), Some("restricted"), None, None));
    }

    #[test]
    fn normalize_clearance_works() {
        assert_eq!(normalize_clearance(" Internal "), Some("internal"));
        assert_eq!(normalize_clearance("weird"), None);
    }
}
