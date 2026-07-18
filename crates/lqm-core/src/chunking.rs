//! Text chunking strategies for ingestion.
//!
//! - `chunk_text` — paragraph-aware sliding window (generic prose)
//! - `chunk_markdown` — split on AT1–H6 headings, then size-limit sections
//! - `chunk_code` — split on common def/fn/class boundaries, then size-limit
//! - `chunk_for_ingest` — pick strategy from source_type / path extension

#[derive(Debug, Clone)]
pub struct ChunkingStrategy {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl ChunkingStrategy {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
        }
    }

    pub fn text(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
        }
    }
}

/// Structural kind used to pick a chunker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    Plain,
    Markdown,
    Code,
}

/// Infer chunk kind from extractor `source_type` and optional path/extension hint.
pub fn chunk_kind_for(source_type: Option<&str>, path_hint: Option<&str>) -> ChunkKind {
    let ext = path_hint
        .and_then(|p| {
            std::path::Path::new(p)
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
        })
        .unwrap_or_default();

    let st = source_type.unwrap_or("").to_ascii_lowercase();

    if st == "markdown"
        || ext == "md"
        || ext == "mdx"
        || ext == "markdown"
        || ext == "rmd"
        || ext == "org"
    {
        return ChunkKind::Markdown;
    }

    if matches!(
        ext.as_str(),
        "rs" | "py"
            | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "go"
            | "java"
            | "c"
            | "cpp"
            | "h"
            | "hpp"
            | "rb"
            | "cs"
            | "kt"
            | "swift"
            | "scala"
            | "php"
            | "lua"
            | "zig"
    ) || st == "code"
    {
        return ChunkKind::Code;
    }

    // Plain text extractor labels code-ish files as "text" — extension still wins above.
    ChunkKind::Plain
}

/// Dispatch to the appropriate chunker.
pub fn chunk_for_ingest(
    text: &str,
    source_type: Option<&str>,
    path_hint: Option<&str>,
    strategy: &ChunkingStrategy,
) -> Vec<String> {
    match chunk_kind_for(source_type, path_hint) {
        ChunkKind::Markdown => chunk_markdown(text, strategy),
        ChunkKind::Code => chunk_code(text, strategy),
        ChunkKind::Plain => chunk_text(text, strategy),
    }
}

pub fn chunk_text(text: &str, strategy: &ChunkingStrategy) -> Vec<String> {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();

    let mut chunks = Vec::new();

    for paragraph in &paragraphs {
        if paragraph.len() <= strategy.chunk_size {
            chunks.push(paragraph.to_string());
        } else {
            chunks.extend(sliding_window_words(paragraph, strategy));
        }
    }

    chunks
}

/// Split markdown on AT1–H6 headings (ATX). Each section keeps its heading line.
/// Oversized sections fall back to paragraph/window chunking.
pub fn chunk_markdown(text: &str, strategy: &ChunkingStrategy) -> Vec<String> {
    if text.trim().is_empty() {
        return vec![];
    }

    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if is_atx_heading(line) && !current.trim().is_empty() {
            sections.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.trim().is_empty() {
        sections.push(current.trim().to_string());
    }

    if sections.is_empty() {
        return chunk_text(text, strategy);
    }

    let mut out = Vec::new();
    for section in sections {
        if section.len() <= strategy.chunk_size {
            out.push(section);
        } else {
            out.extend(chunk_text(&section, strategy));
        }
    }
    out
}

/// Split code on common definition lines (fn/def/class/struct/impl/function…).
/// Oversized blocks use a line-aware sliding window.
pub fn chunk_code(text: &str, strategy: &ChunkingStrategy) -> Vec<String> {
    if text.trim().is_empty() {
        return vec![];
    }

    let mut blocks: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if is_code_boundary(line) && !current.trim().is_empty() {
            blocks.push(current.trim_end().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.trim().is_empty() {
        blocks.push(current.trim_end().to_string());
    }

    if blocks.is_empty() {
        return chunk_text(text, strategy);
    }

    let mut out = Vec::new();
    for block in blocks {
        if block.len() <= strategy.chunk_size {
            out.push(block);
        } else {
            out.extend(sliding_window_lines(&block, strategy));
        }
    }
    out
}

fn is_atx_heading(line: &str) -> bool {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return false;
    }
    let hashes = t.chars().take_while(|c| *c == '#').count();
    if !(1..=6).contains(&hashes) {
        return false;
    }
    t.as_bytes()
        .get(hashes)
        .is_some_and(|b| *b == b' ' || *b == b'\t')
}

fn is_code_boundary(line: &str) -> bool {
    let t = line.trim_start();
    if t.is_empty() {
        return false;
    }
    // Comments are not boundaries (except shebang as file start, which is fine as first block).
    if t.starts_with("//") || (t.starts_with('#') && !t.starts_with("#!") && !t.starts_with("#[")) {
        return false;
    }

    let lower = t.to_ascii_lowercase();
    let prefixes = [
        "fn ",
        "pub fn ",
        "pub(crate) fn ",
        "async fn ",
        "pub async fn ",
        "impl ",
        "struct ",
        "pub struct ",
        "enum ",
        "pub enum ",
        "trait ",
        "pub trait ",
        "mod ",
        "pub mod ",
        "func ",
        "def ",
        "async def ",
        "class ",
        "function ",
        "export function ",
        "export default function ",
        "export class ",
        "export const ",
        "export async function ",
        "public class ",
        "private class ",
        "protected class ",
        "public interface ",
        "public static ",
        "public void ",
        "public async ",
    ];
    prefixes.iter().any(|p| lower.starts_with(p))
}

fn sliding_window_words(paragraph: &str, strategy: &ChunkingStrategy) -> Vec<String> {
    let words: Vec<&str> = paragraph.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let mut end = start;
        let mut current_len = 0;

        while end < words.len() && current_len + words[end].len() < strategy.chunk_size {
            current_len += words[end].len() + 1;
            end += 1;
        }

        if end == start {
            end = start + 1;
        }

        chunks.push(words[start..end].join(" "));

        let advance = if end - start > strategy.overlap / 5 {
            end - start - (strategy.overlap / 5).min(end - start - 1)
        } else {
            1
        };

        start += advance;
    }
    chunks
}

fn sliding_window_lines(block: &str, strategy: &ChunkingStrategy) -> Vec<String> {
    let lines: Vec<&str> = block.lines().collect();
    if lines.is_empty() {
        return vec![];
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let mut end = start;
        let mut len = 0usize;
        while end < lines.len() {
            let add = lines[end].len() + 1;
            if end > start && len + add > strategy.chunk_size {
                break;
            }
            len += add;
            end += 1;
            if len >= strategy.chunk_size {
                break;
            }
        }
        if end == start {
            end = start + 1;
        }
        chunks.push(lines[start..end].join("\n"));
        let overlap_lines = (strategy.overlap / 40).max(1);
        let advance = (end - start).saturating_sub(overlap_lines).max(1);
        start += advance;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let strategy = ChunkingStrategy::text(100, 20);
        let text = "This is a short paragraph.";
        let chunks = chunk_text(text, &strategy);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "This is a short paragraph.");
    }

    #[test]
    fn test_paragraph_split() {
        let strategy = ChunkingStrategy::text(100, 20);
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk_text(text, &strategy);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "First paragraph.");
        assert_eq!(chunks[1], "Second paragraph.");
        assert_eq!(chunks[2], "Third paragraph.");
    }

    #[test]
    fn test_long_paragraph_sliding_window() {
        let strategy = ChunkingStrategy::text(30, 10);
        let mut words = Vec::new();
        for i in 0..20 {
            words.push(format!("word{}", i));
        }
        let text = words.join(" ");
        let chunks = chunk_text(&text, &strategy);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.len() <= 30);
        }
        assert!(chunks.first().unwrap().starts_with("word0"));
        assert!(chunks.last().unwrap().contains("word19"));
    }

    #[test]
    fn test_empty_input() {
        let strategy = ChunkingStrategy::text(100, 20);
        let chunks = chunk_text("", &strategy);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_overlap_creates_multiple_chunks() {
        let strategy = ChunkingStrategy::text(10, 5);
        let text = "a b c d e f g h i j k l m n o p";
        let chunks = chunk_text(text, &strategy);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn markdown_splits_on_headings() {
        let strategy = ChunkingStrategy::text(2000, 50);
        let md = "# Intro\n\nHello world.\n\n## Details\n\nMore text here.\n\n# Outro\n\nBye.";
        let chunks = chunk_markdown(md, &strategy);
        assert!(chunks.len() >= 3, "got {:?}", chunks);
        assert!(chunks[0].starts_with("# Intro"), "{:?}", chunks[0]);
        assert!(
            chunks.iter().any(|c| c.contains("## Details")),
            "{:?}",
            chunks
        );
        assert!(chunks.iter().any(|c| c.contains("# Outro")), "{:?}", chunks);
    }

    #[test]
    fn code_splits_on_fn() {
        let strategy = ChunkingStrategy::text(2000, 50);
        let code = "fn one() {\n  1\n}\n\nfn two() {\n  2\n}\n";
        let chunks = chunk_code(code, &strategy);
        assert!(chunks.len() >= 2, "got {:?}", chunks);
        assert!(chunks[0].contains("fn one"), "{:?}", chunks[0]);
        assert!(chunks.iter().any(|c| c.contains("fn two")), "{:?}", chunks);
    }

    #[test]
    fn chunk_kind_from_extension() {
        assert_eq!(
            chunk_kind_for(Some("text"), Some("readme.md")),
            ChunkKind::Markdown
        );
        assert_eq!(
            chunk_kind_for(Some("text"), Some("main.rs")),
            ChunkKind::Code
        );
        assert_eq!(
            chunk_kind_for(Some("text"), Some("notes.txt")),
            ChunkKind::Plain
        );
        assert_eq!(chunk_kind_for(Some("webpage"), None), ChunkKind::Plain);
    }

    #[test]
    fn chunk_for_ingest_dispatches() {
        let strategy = ChunkingStrategy::text(2000, 20);
        let md = chunk_for_ingest(
            "# A\n\nx\n\n# B\n\ny",
            Some("text"),
            Some("x.md"),
            &strategy,
        );
        assert!(md.len() >= 2);
        let code = chunk_for_ingest(
            "def a():\n  pass\n\ndef b():\n  pass\n",
            Some("text"),
            Some("x.py"),
            &strategy,
        );
        assert!(code.len() >= 2);
    }
}
