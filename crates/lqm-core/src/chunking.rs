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
            let words: Vec<&str> = paragraph.split_whitespace().collect();
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

                let chunk = words[start..end].join(" ");
                chunks.push(chunk);

                let advance = if end - start > strategy.overlap / 5 {
                    end - start - (strategy.overlap / 5).min(end - start - 1)
                } else {
                    1
                };

                start += advance;
            }
        }
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
}
