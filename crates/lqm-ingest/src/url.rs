//! Remote URL / webpage text extraction for agent ingestion.
//!
//! Pure HTML→text helpers are network-free and unit-tested with fixtures.
//! `fetch_url` performs HTTP GET and reuses those extractors.

use crate::ExtractError;
use lqm_core::constants;
use std::time::Duration;

/// Default HTTP timeout for remote fetches (seconds).
pub const DEFAULT_FETCH_TIMEOUT_SECS: u64 = 30;

/// Reject response bodies larger than this (bytes, as received text).
pub const DEFAULT_MAX_FETCH_BYTES: usize = 2 * 1024 * 1024;

/// Result of fetching and extracting a remote document.
#[derive(Debug, Clone)]
pub struct FetchedDocument {
    pub url: String,
    pub text: String,
    pub content_type: Option<String>,
    /// `"webpage"` for HTML, `"url"` for plain/other text payloads.
    pub source_type: String,
    /// HTML `<title>` when present.
    pub title: Option<String>,
}

/// Strip HTML to readable plain text.
///
/// Removes script/style/noscript/nav/footer/header blocks, converts common block
/// tags to newlines, strips remaining tags, decodes entities, collapses whitespace.
/// When a `<title>` is present it is prefixed as `Title: …`.
pub fn html_to_text(html: &str) -> String {
    let title = extract_html_title(html);
    let without_blocks = strip_script_style_blocks(html);
    let with_breaks = insert_block_breaks(&without_blocks);
    let no_tags = strip_tags(&with_breaks);
    let decoded = decode_basic_entities(&no_tags);
    let body = collapse_whitespace(&decoded);
    match title {
        Some(t) if !t.is_empty() => {
            if body.is_empty() {
                format!("Title: {t}")
            } else {
                format!("Title: {t}\n\n{body}")
            }
        }
        _ => body,
    }
}

/// Extract and decode the first `<title>…</title>` if present.
pub fn extract_html_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let after = &html[start..];
    let after_lower = &lower[start..];
    let open_end = after_lower.find('>')?;
    let content_start = open_end + 1;
    let close_rel = after_lower[content_start..].find("</title>")?;
    let raw = after[content_start..content_start + close_rel].trim();
    let decoded = decode_basic_entities(&strip_tags(raw));
    let cleaned = collapse_whitespace(&decoded);
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Choose an extractor based on Content-Type (or body heuristics).
///
/// Returns `(text, source_type)` where source_type is `"webpage"` or `"url"`.
pub fn extract_response_text(content_type: Option<&str>, body: &str) -> (String, String) {
    let ct = content_type.unwrap_or("").to_ascii_lowercase();
    if ct.contains("html") || looks_like_html(body) {
        (
            html_to_text(body),
            constants::SOURCE_TYPE_WEBPAGE.to_string(),
        )
    } else {
        (
            collapse_whitespace(body),
            constants::SOURCE_TYPE_URL.to_string(),
        )
    }
}

/// Fetch a URL and extract usable text. Network I/O only — pure extraction is above.
///
/// Requires the `fetch-url` feature (default-on).
#[cfg(feature = "fetch-url")]
pub async fn fetch_url(
    url: &str,
    timeout: Option<Duration>,
) -> Result<FetchedDocument, ExtractError> {
    if url.trim().is_empty() {
        return Err(ExtractError::ExtractionFailed(
            "url must not be empty".to_string(),
        ));
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(ExtractError::ExtractionFailed(format!(
            "only http(s) URLs are supported: {url}"
        )));
    }

    let timeout = timeout.unwrap_or(Duration::from_secs(DEFAULT_FETCH_TIMEOUT_SECS));
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .user_agent(concat!("liberado-qdrant-mcp/", env!("CARGO_PKG_VERSION")))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| ExtractError::ExtractionFailed(format!("http client build failed: {e}")))?;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| ExtractError::ExtractionFailed(format!("fetch failed for {url}: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        return Err(ExtractError::ExtractionFailed(format!(
            "HTTP {status} fetching {url}"
        )));
    }

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    if let Some(len) = response.content_length()
        && (len as usize) > DEFAULT_MAX_FETCH_BYTES
    {
        return Err(ExtractError::ExtractionFailed(format!(
            "response too large ({len} bytes; max {DEFAULT_MAX_FETCH_BYTES}) for {url}"
        )));
    }

    let mut body = response
        .text()
        .await
        .map_err(|e| ExtractError::ExtractionFailed(format!("read body failed for {url}: {e}")))?;

    if body.len() > DEFAULT_MAX_FETCH_BYTES {
        body.truncate(DEFAULT_MAX_FETCH_BYTES);
        // Avoid cutting mid-codepoint.
        while !body.is_char_boundary(body.len()) {
            body.pop();
        }
    }

    let title = if content_type
        .as_deref()
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("html")
        || looks_like_html(&body)
    {
        extract_html_title(&body)
    } else {
        None
    };

    let (text, source_type) = extract_response_text(content_type.as_deref(), &body);
    if text.trim().is_empty() {
        return Err(ExtractError::ExtractionFailed(format!(
            "no extractable text from {url}"
        )));
    }

    Ok(FetchedDocument {
        url: url.to_string(),
        text,
        content_type,
        source_type,
        title,
    })
}

fn looks_like_html(body: &str) -> bool {
    let trimmed = body.trim_start();
    let lower: String = trimmed
        .chars()
        .take(constants::HTML_LOOKAHEAD_CHARS)
        .collect::<String>()
        .to_ascii_lowercase();
    lower.starts_with("<!doctype html")
        || lower.starts_with("<html")
        || lower.contains("<html")
        || (lower.contains("<head") && lower.contains("<body"))
}

fn strip_script_style_blocks(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len());
    let mut i = 0usize;
    // Walk by byte index; html and lower share the same byte lengths for ASCII tags.
    while i < html.len() {
        if let Some((tag, open_len)) = match_open_block_tag(&lower[i..]) {
            let close = format!("</{tag}>");
            let search_from = i + open_len;
            if let Some(rel) = lower[search_from..].find(&close) {
                i = search_from + rel + close.len();
                continue;
            }
            // Unclosed block: drop the remainder.
            break;
        }
        // Copy one char from the original (UTF-8 safe).
        let ch = html[i..].chars().next().expect("non-empty slice");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn match_open_block_tag(lower_slice: &str) -> Option<(&'static str, usize)> {
    // Boilerplate-heavy chrome is dropped entirely (nav/footer/header/aside/svg).
    for tag in [
        "script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "iframe",
    ] {
        let open = format!("<{tag}");
        if lower_slice.starts_with(&open) {
            let rest = &lower_slice[open.len()..];
            let next = rest.chars().next().unwrap_or('>');
            if next == '>' || next.is_whitespace() || next == '/' {
                return Some((tag, open.len()));
            }
        }
    }
    None
}

fn insert_block_breaks(html: &str) -> String {
    let mut out = html.to_string();
    for tag in [
        "</p>",
        "</div>",
        "</h1>",
        "</h2>",
        "</h3>",
        "</h4>",
        "</h5>",
        "</h6>",
        "</li>",
        "</tr>",
        "<br>",
        "<br/>",
        "<br />",
        "</section>",
        "</article>",
        "</header>",
        "</footer>",
        "</nav>",
        "</main>",
    ] {
        out = replace_ignore_case(&out, tag, "\n");
    }
    out
}

fn replace_ignore_case(haystack: &str, needle: &str, replacement: &str) -> String {
    let lower = haystack.to_ascii_lowercase();
    let needle_l = needle.to_ascii_lowercase();
    let mut out = String::with_capacity(haystack.len());
    let mut last = 0usize;
    let mut search = 0usize;
    while let Some(rel) = lower[search..].find(&needle_l) {
        let start = search + rel;
        out.push_str(&haystack[last..start]);
        out.push_str(replacement);
        last = start + needle.len();
        search = last;
    }
    out.push_str(&haystack[last..]);
    out
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

fn decode_basic_entities(s: &str) -> String {
    s.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_was_space = false;
    let mut newline_run = 0u8;
    for ch in s.chars() {
        if ch == '\r' {
            continue;
        }
        if ch == '\n' {
            newline_run = newline_run.saturating_add(1);
            prev_was_space = true;
            continue;
        }
        if ch.is_whitespace() {
            if !prev_was_space && !out.is_empty() {
                out.push(' ');
            }
            prev_was_space = true;
            newline_run = 0;
            continue;
        }
        if newline_run > 0 {
            // One newline for soft breaks, two for paragraph gaps.
            if !out.is_empty() {
                out.push('\n');
                if newline_run >= 2 {
                    out.push('\n');
                }
            }
            newline_run = 0;
        }
        out.push(ch);
        prev_was_space = false;
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_text_strips_tags_and_keeps_content() {
        let html = r#"
        <!DOCTYPE html>
        <html>
        <head>
          <title>Doc Title</title>
          <style>body { color: red; }</style>
          <script>alert("x")</script>
        </head>
        <body>
          <nav>Skip nav</nav>
          <h1>Hello World</h1>
          <p>This is a <b>paragraph</b> with &amp; entities.</p>
          <footer>Skip footer</footer>
          <script>evil()</script>
        </body>
        </html>
        "#;
        let text = html_to_text(html);
        assert!(text.contains("Title: Doc Title"), "text was: {text}");
        assert!(text.contains("Hello World"), "text was: {text}");
        assert!(
            text.contains("This is a paragraph with & entities."),
            "text was: {text}"
        );
        assert!(!text.contains("alert"), "script leaked: {text}");
        assert!(!text.contains("color: red"), "style leaked: {text}");
        assert!(!text.contains("Skip nav"), "nav leaked: {text}");
        assert!(!text.contains("Skip footer"), "footer leaked: {text}");
        assert!(!text.contains("<h1>"), "tags leaked: {text}");
    }

    #[test]
    fn extract_html_title_works() {
        assert_eq!(
            extract_html_title("<html><title>Hello &amp; Co</title></html>").as_deref(),
            Some("Hello & Co")
        );
    }

    #[test]
    fn html_to_text_emptyish() {
        assert_eq!(html_to_text(""), "");
        assert_eq!(html_to_text("<div></div>"), "");
    }

    #[test]
    fn html_to_text_preserves_unicode() {
        let html = "<p>Café 日本語</p>";
        let text = html_to_text(html);
        assert!(text.contains("Café"), "text was: {text}");
        assert!(text.contains("日本語"), "text was: {text}");
    }

    #[test]
    fn extract_response_text_html_vs_plain() {
        let (html_text, st) = extract_response_text(Some("text/html; charset=utf-8"), "<p>Hi</p>");
        assert_eq!(st, constants::SOURCE_TYPE_WEBPAGE);
        assert!(html_text.contains("Hi"));

        let (plain, st2) = extract_response_text(Some("text/plain"), "just text");
        assert_eq!(st2, constants::SOURCE_TYPE_URL);
        assert_eq!(plain, "just text");
    }

    #[test]
    fn extract_response_text_heuristic_html() {
        let body = "<!DOCTYPE html><html><body>Page</body></html>";
        let (text, st) = extract_response_text(None, body);
        assert_eq!(st, constants::SOURCE_TYPE_WEBPAGE);
        assert!(text.contains("Page"));
    }

    #[test]
    fn decode_entities() {
        assert_eq!(
            decode_basic_entities("a&nbsp;b&amp;c&lt;d&gt;e&quot;f&#39;g&apos;h"),
            "a b&c<d>e\"f'g'h"
        );
    }

    #[cfg(feature = "fetch-url")]
    #[tokio::test]
    async fn fetch_url_rejects_empty_and_non_http() {
        let empty = fetch_url("", None).await;
        assert!(empty.is_err());
        let ftp = fetch_url("ftp://example.com/x", None).await;
        assert!(ftp.is_err());
        let file = fetch_url("file:///etc/passwd", None).await;
        assert!(file.is_err());
    }

    /// Offline happy-path for response extraction (fetch_url network path uses this).
    #[test]
    fn extract_response_text_success_fixture() {
        let html = r#"<!DOCTYPE html><html><head><title>Doc</title></head>
<body><p>Hello from fixture page for offline validation.</p></body></html>"#;
        let (text, st) = extract_response_text(Some("text/html"), html);
        assert_eq!(st, constants::SOURCE_TYPE_WEBPAGE);
        assert!(text.contains("Hello from fixture"));
        assert!(text.contains("Doc") || text.contains("Title"));
    }
}
