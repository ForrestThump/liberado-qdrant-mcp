//! Canonical source-type identifiers used in payload schema and tool params.
//!
//! Each variant maps to the string stored in the Qdrant payload `source_type`
//! field.  The `DocumentChunk` struct stores it as `Option<String>` for serde
//! compatibility; this enum provides type-safe construction and matching.

use core::fmt;
use std::str::FromStr;

/// Canonical source types for document chunks and extracted content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceType {
    Text,
    Webpage,
    Url,
    Pdf,
    Audio,
    Memory,
    Markdown,
    Code,
}

impl SourceType {
    /// All known variants (useful for validation / error messages).
    pub const ALL: &'static [SourceType] = &[
        SourceType::Text,
        SourceType::Webpage,
        SourceType::Url,
        SourceType::Pdf,
        SourceType::Audio,
        SourceType::Memory,
        SourceType::Markdown,
        SourceType::Code,
    ];

    pub const fn as_str(&self) -> &'static str {
        // Canonical string values live here; `constants::SOURCE_TYPE_*` re-export these.
        match self {
            SourceType::Text => "text",
            SourceType::Webpage => "webpage",
            SourceType::Url => "url",
            SourceType::Pdf => "pdf",
            SourceType::Audio => "audio",
            SourceType::Memory => "memory",
            SourceType::Markdown => "markdown",
            SourceType::Code => "code",
        }
    }
}

impl fmt::Display for SourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SourceType {
    type Err = UnknownSourceType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "text" => Ok(SourceType::Text),
            "webpage" => Ok(SourceType::Webpage),
            "url" => Ok(SourceType::Url),
            "pdf" => Ok(SourceType::Pdf),
            "audio" => Ok(SourceType::Audio),
            "memory" => Ok(SourceType::Memory),
            "markdown" => Ok(SourceType::Markdown),
            "code" => Ok(SourceType::Code),
            other => Err(UnknownSourceType(other.to_string())),
        }
    }
}

/// Error returned when a string does not match any known `SourceType`.
#[derive(Debug, Clone)]
pub struct UnknownSourceType(pub String);

impl fmt::Display for UnknownSourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown source type '{}'; expected one of: {}",
            self.0,
            SourceType::ALL
                .iter()
                .map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl std::error::Error for UnknownSourceType {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_display_parse() {
        for st in SourceType::ALL {
            let s = st.to_string();
            let parsed: SourceType = s.parse().unwrap();
            assert_eq!(st, &parsed);
        }
    }

    #[test]
    fn parse_case_insensitive() {
        assert_eq!("TEXT".parse::<SourceType>().unwrap(), SourceType::Text);
        assert_eq!(
            "WebPage".parse::<SourceType>().unwrap(),
            SourceType::Webpage
        );
    }

    #[test]
    fn unknown_returns_err() {
        assert!("unknown_type".parse::<SourceType>().is_err());
    }

    #[test]
    fn as_str_matches_display() {
        assert_eq!(SourceType::Text.as_str(), "text");
        assert_eq!(SourceType::Memory.as_str(), "memory");
    }
}
