//! Text chunking strategies for ingestion.
//!
//! - `chunk_text` — paragraph-aware sliding window (generic prose)
//! - `chunk_markdown` — split on AT1–H6 headings, then size-limit sections
//! - `chunk_code` — split on common def/fn/class boundaries, then size-limit
//! - `chunk_for_ingest` — pick strategy from source_type / path extension

use crate::constants;

/// File extensions treated as markdown content.
pub const MARKDOWN_EXTS: &[&str] = &["md", "mdx", "markdown", "rmd", "org"];

/// File extensions treated as code content.
pub const CODE_EXTS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp", "rb", "cs", "kt",
    "swift", "scala", "php", "lua", "zig",
];

/// Line prefixes that signal a code-block boundary (checked lowercased).
pub const CODE_BOUNDARY_PREFIXES: &[&str] = &[
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

/// Infer chunk kind from extractor `source_type`, path extension, and optional content preview.
///
/// When extension and source_type are both ambiguous (e.g. `.txt`, `.log`, empty), the
/// optional `content_hint` (first ~256 bytes of raw text) is inspected for structural
/// markdown or code patterns.
pub fn chunk_kind_for(
    source_type: Option<&str>,
    path_hint: Option<&str>,
    content_hint: Option<&str>,
) -> ChunkKind {
    let ext = path_hint
        .and_then(|p| {
            std::path::Path::new(p)
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
        })
        .unwrap_or_default();

    let st = source_type.unwrap_or("").to_ascii_lowercase();

    if st == crate::constants::SOURCE_TYPE_MARKDOWN || MARKDOWN_EXTS.contains(&ext.as_str()) {
        return ChunkKind::Markdown;
    }

    if CODE_EXTS.contains(&ext.as_str()) || st == crate::constants::SOURCE_TYPE_CODE {
        return ChunkKind::Code;
    }

    if let Some(content) = content_hint {
        if looks_like_markdown(content) {
            return ChunkKind::Markdown;
        }
        if looks_like_code(content) {
            return ChunkKind::Code;
        }
    }

    ChunkKind::Plain
}

/// Quick heuristic: does the first chunk of text look like structured markdown?
pub fn looks_like_markdown(content: &str) -> bool {
    let preview = &content[..content.len().min(constants::HTML_LOOKAHEAD_CHARS)];
    for line in preview.lines().take(10) {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            return true;
        }
        if (t.starts_with("# ") || t.starts_with("## "))
            || (t.starts_with("### ") || t.starts_with("#### "))
            || t.starts_with("##### ") || t.starts_with("###### ")
        {
            return true;
        }
    }
    false
}

/// Quick heuristic: does the first chunk of text look like code?
pub fn looks_like_code(content: &str) -> bool {
    let preview = &content[..content.len().min(constants::HTML_LOOKAHEAD_CHARS)];
    for line in preview.lines().take(10).filter(|l| !l.trim().is_empty()) {
        if is_code_boundary(line) {
            return true;
        }
    }
    false
}

/// Dispatch to the appropriate chunker.
pub fn chunk_for_ingest(
    text: &str,
    source_type: Option<&str>,
    path_hint: Option<&str>,
    strategy: &ChunkingStrategy,
) -> Vec<String> {
    match chunk_kind_for(source_type, path_hint, Some(text)) {
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

/// Detect ATX headings (`# Title` with space/tab after 1–6 hashes).
///
/// Closing hashes (`## Title ##`) are allowed by CommonMark and are still detected
/// as headings here because we only require whitespace after the opening run.
/// Lines like `##NoSpace` are not headings.
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
    if t.starts_with("//") || (t.starts_with('#') && !t.starts_with("#!") && !t.starts_with("#[")) {
        return false;
    }

    let lower = t.to_ascii_lowercase();
    CODE_BOUNDARY_PREFIXES.iter().any(|p| lower.starts_with(p))
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

        let overlap_words = strategy.overlap / constants::SLIDING_WINDOW_WORD_OVERLAP_DIV;
        let advance = if end - start > overlap_words {
            end - start - overlap_words.min(end - start - 1)
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
        let overlap_lines = (strategy.overlap / constants::SLIDING_WINDOW_LINE_OVERLAP_DIV)
            .max(constants::SLIDING_WINDOW_MIN_OVERLAP_LINES);
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
            chunk_kind_for(Some("text"), Some("readme.md"), None),
            ChunkKind::Markdown
        );
        assert_eq!(
            chunk_kind_for(Some("text"), Some("main.rs"), None),
            ChunkKind::Code
        );
        assert_eq!(
            chunk_kind_for(Some("text"), Some("notes.txt"), None),
            ChunkKind::Plain
        );
        assert_eq!(chunk_kind_for(Some("webpage"), None, None), ChunkKind::Plain);
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

    #[test]
    fn atx_heading_open_and_closed_forms() {
        assert!(is_atx_heading("## Open form"));
        // Closed ATX still has space after opening hashes → treated as heading.
        assert!(is_atx_heading("## Closed form ##"));
        assert!(!is_atx_heading("##NoSpace"));
        let strategy = ChunkingStrategy::text(2000, 20);
        let doc = "## One ##\n\nbody a\n\n## Two\n\nbody b";
        let chunks = chunk_markdown(doc, &strategy);
        assert!(
            chunks.len() >= 2,
            "closed + open ATX both split: {chunks:?}"
        );
    }

    // --- is_atx_heading variants (H1, H6, H7+, tab, indent) ---

    #[test]
    fn atx_heading_h1_and_h6_detected() {
        assert!(is_atx_heading("# H1"));
        assert!(is_atx_heading("###### H6"));
    }

    #[test]
    fn atx_heading_h7_rejected() {
        assert!(!is_atx_heading("####### too many"));
    }

    #[test]
    fn atx_heading_tab_after_hashes_accepted() {
        assert!(is_atx_heading("##\tTabbed"));
    }

    #[test]
    fn atx_heading_indented_detected() {
        assert!(is_atx_heading("  ## Indented"));
    }

    #[test]
    fn atx_heading_bare_hash_with_space() {
        assert!(is_atx_heading("# "));
    }

    // --- is_code_boundary (comments, boundary prefixes, indented) ---

    #[test]
    fn code_boundary_rejects_comments() {
        assert!(!is_code_boundary("// C-style comment"));
        assert!(!is_code_boundary("   // indented comment"));
        assert!(!is_code_boundary("# Python comment"));
    }

    #[test]
    fn code_boundary_allows_shebang_and_attribute() {
        // #! and #[ should not be treated as comments (they start with # but have special meaning)
        // #! is a shebang — let it pass through to prefix check
        assert!(!is_code_boundary("#!/usr/bin/env python3"));
        // #[ is a Rust attribute — not a comment, but doesn't match any boundary prefix
        assert!(!is_code_boundary("#[derive(Debug)]"));
    }

    #[test]
    fn code_boundary_various_prefixes() {
        assert!(is_code_boundary("fn "));
        assert!(is_code_boundary("pub fn bar()"));
        assert!(is_code_boundary("impl Foo"));
        assert!(is_code_boundary("struct Bar"));
        assert!(is_code_boundary("enum Color"));
        assert!(is_code_boundary("trait Handler"));
        assert!(is_code_boundary("mod my_module"));
        assert!(is_code_boundary("class Widget"));
        assert!(is_code_boundary("def my_func():"));
        assert!(is_code_boundary("function myFunc()"));
        assert!(is_code_boundary("export function foo()"));
        assert!(is_code_boundary("public class JClass"));
    }

    #[test]
    fn code_boundary_case_insensitive() {
        assert!(is_code_boundary("FN uppercase()"));
        assert!(is_code_boundary("CLASS Pascal()"));
        assert!(is_code_boundary("Def Python()"));
    }

    #[test]
    fn code_boundary_rejects_empty_lines() {
        assert!(!is_code_boundary(""));
        assert!(!is_code_boundary("   "));
    }

    #[test]
    fn code_boundary_indented_detected() {
        assert!(is_code_boundary("    fn indented()"));
        assert!(is_code_boundary("  class Padded"));
    }

    // --- chunk_kind_for canonical source_type dispatch ---

    #[test]
    fn chunk_kind_from_canonical_source_type() {
        assert_eq!(chunk_kind_for(Some("markdown"), None, None), ChunkKind::Markdown);
        assert_eq!(
            chunk_kind_for(Some("code"), Some("notes.txt"), None),
            ChunkKind::Code
        );
    }

    #[test]
    fn chunk_kind_from_mixed_case_source_type() {
        assert_eq!(chunk_kind_for(Some("MARKDOWN"), None, None), ChunkKind::Markdown);
        assert_eq!(chunk_kind_for(Some("Code"), None, None), ChunkKind::Code);
    }

    #[test]
    fn chunk_kind_from_uppercase_extension() {
        assert_eq!(
            chunk_kind_for(Some("text"), Some("README.MD"), None),
            ChunkKind::Markdown
        );
        assert_eq!(
            chunk_kind_for(Some("text"), Some("MAIN.RS"), None),
            ChunkKind::Code
        );
    }

    // --- chunk_for_ingest Plain dispatch ---

    #[test]
    fn chunk_for_ingest_plain_dispatches_to_chunk_text() {
        let strategy = ChunkingStrategy::text(2000, 20);
        let chunks = chunk_for_ingest("plain text notes", None, None, &strategy);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "plain text notes");
    }

    #[test]
    fn chunk_for_ingest_plain_with_text_source() {
        let strategy = ChunkingStrategy::text(2000, 20);
        let chunks = chunk_for_ingest("just some notes", Some("text"), Some("file.txt"), &strategy);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "just some notes");
    }

    // --- chunk_markdown: oversized-section → chunk_text fallback ---

    #[test]
    fn markdown_oversized_section_falls_back_to_chunk_text() {
        // chunk_size=20 forces a section with a long paragraph to be split
        let strategy = ChunkingStrategy::text(20, 5);
        let md = "# Title\n\nthis is a very long paragraph that exceeds twenty characters";
        let chunks = chunk_markdown(md, &strategy);
        assert!(
            chunks.len() >= 2,
            "long paragraph under heading should split into multiple chunks, got {chunks:?}"
        );
        for c in &chunks {
            assert!(
                c.len() <= 20 || !c.contains(' '),
                "chunk len {} exceeds 20: {c:?}",
                c.len()
            );
        }
    }

    #[test]
    fn markdown_empty_input_returns_empty() {
        let strategy = ChunkingStrategy::text(100, 20);
        assert!(chunk_markdown("   \n\n", &strategy).is_empty());
    }

    #[test]
    fn markdown_lone_heading_without_body() {
        let strategy = ChunkingStrategy::text(2000, 20);
        let chunks = chunk_markdown("# Lone heading", &strategy);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "# Lone heading");
    }

    // --- chunk_code: oversized-block → sliding_window_lines fallback ---

    #[test]
    fn code_oversized_block_sliding_window_lines() {
        // chunk_size=30 forces a long function body to be split via line-level sliding window
        let strategy = ChunkingStrategy::text(30, 10);
        let code = "fn long_fn() {\n  line_a\n  line_b\n  line_c\n  line_d\n  line_e\n  line_f\n}";
        let chunks = chunk_code(code, &strategy);
        assert!(
            chunks.len() >= 2,
            "long code block should split into multiple chunks, got {chunks:?}"
        );
        // All chunks should contain at least one line and not exceed soft cap unreasonably
        for c in &chunks {
            assert!(!c.is_empty(), "chunks should not be empty");
        }
    }

    #[test]
    fn code_empty_input_returns_empty() {
        let strategy = ChunkingStrategy::text(100, 20);
        assert!(chunk_code("   \n", &strategy).is_empty());
    }

    #[test]
    fn code_no_boundary_falls_back_to_chunk_text() {
        let strategy = ChunkingStrategy::text(2000, 20);
        // No code boundary prefix in this text → single block → chunk_text
        let code = "x = 1\ny = 2";
        let chunks = chunk_code(code, &strategy);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "x = 1\ny = 2");
    }

    // --- sliding_window_words edge cases ---

    #[test]
    fn sliding_window_words_oversized_single_word() {
        let strategy = ChunkingStrategy::text(3, 1);
        // A word that exceeds chunk_size entirely
        let chunks = chunk_text("supercalifragilisticexpialidocious", &strategy);
        assert!(!chunks.is_empty(), "oversized word should produce a chunk");
        assert_eq!(chunks[0], "supercalifragilisticexpialidocious");
    }

    #[test]
    fn sliding_window_words_small_advance_when_overlap_consumes_most() {
        let strategy = ChunkingStrategy::text(15, 12);
        // overlap/5 = 2, with small windows the advance falls to 1
        let text = "a b c d e f g h i j";
        let chunks = chunk_text(text, &strategy);
        assert!(
            chunks.len() >= 3,
            "small advance should produce many chunks, got {:?}",
            chunks
        );
    }

    // --- sliding_window_lines edge cases ---

    #[test]
    fn sliding_window_lines_oversized_single_line() {
        let strategy = ChunkingStrategy::text(3, 1);
        // Split on code boundary to trigger sliding_window_lines with a long block
        let code = "fn f() {\n  very_long_line_that_exceeds_chunk_size\n}";
        let chunks = chunk_code(code, &strategy);
        assert!(!chunks.is_empty(), "oversized line should not crash");
    }

    #[test]
    fn sliding_window_lines_empty_input_returns_empty() {
        let strategy = ChunkingStrategy::text(100, 20);
        // chunk_code on empty/whitespace-only text exercises sliding_window_lines
        // indirectly via the blocks.is_empty() fallback
        let chunks = chunk_code("\n", &strategy);
        assert!(chunks.is_empty());
    }

    // --- content-type detection ---

    #[test]
    fn looks_like_markdown_detects_headings_and_fences() {
        assert!(looks_like_markdown("# Title\n\nbody"));
        assert!(looks_like_markdown("## H2\n\npara\n\n###### H6"));
        assert!(looks_like_markdown("```rust\ncode\n```"));
        assert!(looks_like_markdown("~~~\nblock\n~~~"));
    }

    #[test]
    fn looks_like_markdown_rejects_plain() {
        assert!(!looks_like_markdown("Hello world.\n\nJust some paragraphs."));
    }

    #[test]
    fn looks_like_code_detects_patterns() {
        assert!(looks_like_code("fn main() {}\n"));
        assert!(looks_like_code("def hello():\n    pass"));
        assert!(looks_like_code("import os\n\ndef main():\n    pass"));
    }

    #[test]
    fn looks_like_code_rejects_plain() {
        assert!(!looks_like_code("Hello world.\n\nMore text here."));
    }

    #[test]
    fn chunk_kind_for_content_hint_on_ambiguous_ext() {
        assert_eq!(
            chunk_kind_for(Some("text"), Some("notes.txt"), Some("# Heading\n\nbody")),
            ChunkKind::Markdown
        );
        assert_eq!(
            chunk_kind_for(Some("text"), Some("notes.txt"), Some("fn main() {}")),
            ChunkKind::Code
        );
        assert_eq!(
            chunk_kind_for(Some("text"), Some("notes.txt"), Some("just notes")),
            ChunkKind::Plain
        );
    }
}
