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

use serde::{Deserialize, Serialize};

/// Ordered clearance levels. Lower variant rank = less sensitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(into = "String", try_from = "String")]
pub enum Clearance {
    Public = 0,
    Internal = 1,
    Confidential = 2,
    Restricted = 3,
}

impl Clearance {
    /// All clearance levels in ascending sensitivity order.
    pub const ALL: &[Clearance] = &[
        Clearance::Public,
        Clearance::Internal,
        Clearance::Confidential,
        Clearance::Restricted,
    ];

    /// Canonical string for Qdrant payload storage.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Clearance::Public => "public",
            Clearance::Internal => "internal",
            Clearance::Confidential => "confidential",
            Clearance::Restricted => "restricted",
        }
    }

    /// Levels at or below `self` (inclusive).
    pub fn allowed_levels(&self) -> Vec<Clearance> {
        let max_rank = *self as u8;
        Clearance::ALL[..=max_rank as usize].to_vec()
    }

    /// Whether this clearance is allowed under a `max_clearance` ceiling.
    pub fn allowed_under(&self, max: Clearance) -> bool {
        *self <= max
    }
}

/// Default clearance when ingest omits the field.
pub const DEFAULT_CLEARANCE: Clearance = Clearance::Public;

impl std::fmt::Display for Clearance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<Clearance> for String {
    fn from(c: Clearance) -> Self {
        c.as_str().to_string()
    }
}

impl std::str::FromStr for Clearance {
    type Err = UnknownClearance;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "public" => Ok(Clearance::Public),
            "internal" => Ok(Clearance::Internal),
            "confidential" => Ok(Clearance::Confidential),
            "restricted" => Ok(Clearance::Restricted),
            other => Err(UnknownClearance(other.to_string())),
        }
    }
}

impl TryFrom<String> for Clearance {
    type Error = UnknownClearance;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

#[derive(Debug, Clone)]
pub struct UnknownClearance(pub String);

impl std::fmt::Display for UnknownClearance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown clearance '{}'; expected one of: {}",
            self.0,
            Clearance::ALL
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl std::error::Error for UnknownClearance {}

/// Normalize a clearance string to a known `Clearance`, or `None` if invalid.
pub fn normalize_clearance(level: &str) -> Option<Clearance> {
    level.parse().ok()
}

/// Parse an optional clearance from an MCP/HTTP boundary string.
///
/// - `None` or blank/whitespace → `Ok(None)` (absent; no filter / default ingest)
/// - known level (case-insensitive, trimmed) → `Ok(Some(Clearance))`
/// - any other non-blank token → `Err(UnknownClearance)` (fail closed; never admit-all)
pub fn parse_optional_clearance(
    value: Option<&str>,
) -> Result<Option<Clearance>, UnknownClearance> {
    match value {
        None => Ok(None),
        Some(s) if s.trim().is_empty() => Ok(None),
        Some(s) => s.parse().map(Some),
    }
}

/// Whether a point's clearance is allowed under `max_clearance`.
///
/// Missing / empty point clearance is treated as [`DEFAULT_CLEARANCE`].
pub fn clearance_allowed(point_clearance: Option<&str>, max_clearance: Clearance) -> bool {
    let point = point_clearance
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<Clearance>().ok())
        .unwrap_or(DEFAULT_CLEARANCE);
    point.allowed_under(max_clearance)
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
    max_clearance: Option<Clearance>,
) -> bool {
    if let Some(scope) = required_scope
        && !scope_matches(point_scope, scope)
    {
        return false;
    }
    if let Some(max) = max_clearance
        && !clearance_allowed(point_clearance, max)
    {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clearance_ranks_are_ordered() {
        assert!(Clearance::Public < Clearance::Internal);
        assert!(Clearance::Internal < Clearance::Confidential);
        assert!(Clearance::Confidential < Clearance::Restricted);
        assert_eq!("PUBLIC".parse::<Clearance>().unwrap(), Clearance::Public);
        assert!("nope".parse::<Clearance>().is_err());
    }

    #[test]
    fn clearance_allowed_includes_lower_and_equal() {
        assert!(clearance_allowed(Some("public"), Clearance::Internal));
        assert!(clearance_allowed(Some("internal"), Clearance::Internal));
        assert!(!clearance_allowed(
            Some("confidential"),
            Clearance::Internal
        ));
        assert!(!clearance_allowed(Some("restricted"), Clearance::Public));
        assert!(clearance_allowed(None, Clearance::Public));
        assert!(clearance_allowed(Some(""), Clearance::Public));
        assert!(!clearance_allowed(Some("internal"), Clearance::Public));
    }

    #[test]
    fn allowed_levels_expand() {
        assert_eq!(Clearance::Public.allowed_levels(), vec![Clearance::Public]);
        assert_eq!(
            Clearance::Confidential.allowed_levels(),
            vec![
                Clearance::Public,
                Clearance::Internal,
                Clearance::Confidential
            ]
        );
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
            Some(Clearance::Confidential)
        ));
        assert!(!point_in_scope(
            Some("beta"),
            Some("public"),
            Some("alpha"),
            Some(Clearance::Restricted)
        ));
        assert!(!point_in_scope(
            Some("alpha"),
            Some("restricted"),
            Some("alpha"),
            Some(Clearance::Internal)
        ));
        assert!(point_in_scope(Some("x"), Some("restricted"), None, None));
    }

    #[test]
    fn normalize_clearance_works() {
        assert_eq!(normalize_clearance(" Internal "), Some(Clearance::Internal));
        assert_eq!(normalize_clearance("weird"), None);
    }

    #[test]
    fn parse_optional_clearance_blank_valid_invalid() {
        assert_eq!(parse_optional_clearance(None).unwrap(), None);
        assert_eq!(parse_optional_clearance(Some("")).unwrap(), None);
        assert_eq!(parse_optional_clearance(Some("   ")).unwrap(), None);
        assert_eq!(
            parse_optional_clearance(Some("internal")).unwrap(),
            Some(Clearance::Internal)
        );
        assert_eq!(
            parse_optional_clearance(Some(" CONFIDENTIAL ")).unwrap(),
            Some(Clearance::Confidential)
        );
        assert_eq!(
            parse_optional_clearance(Some("PUBLIC")).unwrap(),
            Some(Clearance::Public)
        );
        let err = parse_optional_clearance(Some("nope")).unwrap_err();
        assert!(err.to_string().contains("nope"), "{err}");
        let err2 = parse_optional_clearance(Some("top-secret")).unwrap_err();
        assert!(err2.to_string().contains("top-secret"), "{err2}");
        assert!(
            err2.to_string().contains("public"),
            "error should list expected levels: {err2}"
        );
    }

    #[test]
    fn serde_roundtrip() {
        let json = serde_json::to_string(&Clearance::Confidential).unwrap();
        assert_eq!(json, "\"confidential\"");
        let parsed: Clearance = serde_json::from_str("\"RESTRICTED\"").unwrap();
        assert_eq!(parsed, Clearance::Restricted);
    }

    #[test]
    fn all_variants_parse() {
        for c in Clearance::ALL {
            let s = c.to_string();
            let parsed: Clearance = s.parse().unwrap();
            assert_eq!(*c, parsed);
        }
    }
}
