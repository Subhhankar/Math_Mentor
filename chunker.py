"""
chunker.py
----------
Semantic double-pass chunking pipeline for mathematical PDFs.
Handles PDF loading, formula detection, problem/solution boundary detection,
metadata extraction, and Pinecone-ready document creation.

Usage:
    from chunker import process_mathematical_documents
    chunks = process_mathematical_documents("Data/")
"""

import re
import numpy as np
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ──────────────────────────────────────────────
# Embeddings (singleton — import and reuse)
# ──────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ──────────────────────────────────────────────
# PDF Loading
# ──────────────────────────────────────────────

def load_pdf_files(data_path: str) -> List[Document]:
    """Load all PDFs from a directory."""
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Strip metadata down to source only."""
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source")}
        )
        for doc in docs
    ]


# ──────────────────────────────────────────────
# Math Formula Detector
# ──────────────────────────────────────────────

class MathFormulaDetector:
    """Detect and preserve mathematical formulas in text."""

    @staticmethod
    def detect_latex_formulas(text: str) -> List[Tuple[int, int, str]]:
        """
        Find LaTeX formulas and math symbols.
        Returns list of (start_pos, end_pos, formula_text).
        """
        formulas = []

        patterns = [
            r'\$\$(.+?)\$\$',                                           # Display math $$...$$
            r'\$([^\$]+)\$',                                             # Inline math $...$
            r'\\begin\{(equation|align|gather|multline)\*?\}(.+?)\\end\{\1\*?\}',  # LaTeX envs
            r'[∫∑∏√∂∇±×÷≠≈≤≥∈∉⊂⊃∩∪∞]+',                             # Unicode symbols
        ]
        flags = [re.DOTALL, 0, re.DOTALL, 0]

        for pattern, flag in zip(patterns, flags):
            for match in re.finditer(pattern, text, flag):
                formulas.append((match.start(), match.end(), match.group(0)))

        return formulas

    @staticmethod
    def has_math_content(text: str) -> bool:
        """Return True if text contains mathematical content."""
        indicators = [
            r'\$.*?\$',
            r'\$\$.*?\$\$',
            r'\\begin\{equation\}',
            r'\\frac\{',
            r'\\int',
            r'\\sum',
            r'[∫∑∏√∂∇±×÷≠≈≤≥∈∉⊂⊃∩∪∞]',
            r'\b(theorem|lemma|proof|corollary|proposition)\b',
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in indicators)


# ──────────────────────────────────────────────
# Problem / Solution Detector
# ──────────────────────────────────────────────

class ProblemSolutionDetector:
    """Detect problem and solution boundaries in text."""

    @staticmethod
    def detect_problem_boundaries(text: str) -> List[Tuple[int, int, str]]:
        """
        Find problem statement positions.
        Returns list of (start_pos, end_pos, problem_type).
        """
        patterns = [
            (r'(?:^|\n)(Problem|Question|Exercise|Example)\s*\d*[:.]\s*', 'problem'),
            (r'(?:^|\n)Q\d+[:.]\s*', 'question'),
            (r'(?:^|\n)\d+\.\s*(?=\w)', 'numbered'),
            (r'(?:^|\n)(Theorem|Lemma|Proposition|Corollary)\s*\d*[:.]\s*', 'theorem'),
        ]
        boundaries = []
        for pattern, prob_type in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                boundaries.append((match.start(), match.end(), prob_type))
        return sorted(boundaries, key=lambda x: x[0])

    @staticmethod
    def detect_solution_markers(text: str) -> List[int]:
        """Return positions of solution markers."""
        markers = [
            r'(?:^|\n)(Solution|Answer|Proof)[:.]\s*',
            r'(?:^|\n)Sol[:.]\s*',
        ]
        positions = []
        for pattern in markers:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                positions.append(match.start())
        return sorted(positions)


# ──────────────────────────────────────────────
# Metadata Extractor
# ──────────────────────────────────────────────

class MathMetadataExtractor:
    """Extract topic, difficulty, and concept metadata from math text."""

    TOPIC_KEYWORDS = {
        "calculus":       ["integral", "derivative", "limit", "differentiation", "integration", "series"],
        "algebra":        ["polynomial", "equation", "factor", "quadratic", "roots", "algebraic"],
        "probability":    ["probability", "random", "distribution", "expectation", "variance", "event"],
        "geometry":       ["triangle", "circle", "angle", "area", "volume", "coordinate"],
        "linear_algebra": ["matrix", "vector", "eigenvalue", "determinant", "linear", "span"],
    }

    DIFFICULTY_HARD   = ["proof", "theorem", "lemma", "corollary", "advanced", "generalize"]
    DIFFICULTY_EASY   = ["basic", "simple", "introduction", "elementary", "example"]

    CONCEPT_KEYWORDS  = [
        "integration", "differentiation", "limit", "series", "theorem", "proof",
        "equation", "function", "matrix", "vector", "probability", "combinatorics",
    ]

    def extract_topic(self, text: str) -> str:
        text_lower = text.lower()
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        return "general"

    def extract_difficulty(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in self.DIFFICULTY_HARD):
            return "hard"
        if any(kw in text_lower for kw in self.DIFFICULTY_EASY):
            return "easy"
        return "medium"

    def extract_concepts(self, text: str) -> List[str]:
        text_lower = text.lower()
        return [c for c in self.CONCEPT_KEYWORDS if c in text_lower]

    def count_formulas(self, text: str) -> int:
        return len(MathFormulaDetector.detect_latex_formulas(text))


# ──────────────────────────────────────────────
# Semantic Double-Pass Chunker
# ──────────────────────────────────────────────

class SemanticDoublePassChunker:
    """
    BitPeak-inspired semantic double-pass chunking optimised for math content.

    Pass 1 — builds initial chunks by semantic similarity.
    Pass 2 — look-ahead merging to reunite formula-interrupted text.
    """

    def __init__(
        self,
        embeddings_model=None,
        initial_threshold: float = 0.75,
        merging_threshold: float = 0.65,
        chunk_size: int = 1200,
        chunk_overlap: int = 250,
    ):
        self.embeddings       = embeddings_model
        self.initial_threshold = initial_threshold
        self.merging_threshold = merging_threshold
        self.chunk_size       = chunk_size
        self.chunk_overlap    = chunk_overlap
        self.math_detector    = MathFormulaDetector()

    # ── helpers ──────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        return np.array(self.embeddings.embed_query(text))

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Sentence-split while keeping formulas intact."""
        formulas = self.math_detector.detect_latex_formulas(text)
        formula_map = {}
        processed = text

        for idx, (start, end, formula) in enumerate(formulas):
            placeholder = f"<<FORMULA_{idx}>>"
            formula_map[placeholder] = formula
            processed = processed[:start] + placeholder + processed[end:]

        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', processed)

        return [
            next(
                (s.replace(ph, f) for ph, f in formula_map.items() if ph in s),
                s
            )
            for s in sentences
        ]

    # ── pass 1 ───────────────────────────────

    def _initial_chunking(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        chunks = []
        current_chunk = sentences[0]
        current_emb   = self._embed(current_chunk)

        for sentence in sentences[1:]:
            potential = current_chunk + " " + sentence

            if len(potential) > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
                current_emb   = self._embed(sentence)
            else:
                sent_emb    = self._embed(sentence)
                similarity  = self._cosine(current_emb, sent_emb)

                if similarity > self.initial_threshold:
                    current_chunk = potential
                    current_emb   = (current_emb + sent_emb) / 2
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                    current_emb   = sent_emb

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # ── pass 2 ───────────────────────────────

    def _double_pass_merging(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 2:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            # Look-ahead merge: chunks[i] ↔ chunks[i+2] with math in the middle
            if i + 2 < len(chunks):
                emb_cur   = self._embed(chunks[i])
                emb_ahead = self._embed(chunks[i + 2])
                sim       = self._cosine(emb_cur, emb_ahead)
                mid_math  = self.math_detector.has_math_content(chunks[i + 1])

                if sim > self.merging_threshold and mid_math:
                    merged.append(chunks[i] + " " + chunks[i + 1] + " " + chunks[i + 2])
                    i += 3
                    continue

            # Standard adjacent merge
            if i + 1 < len(chunks):
                emb_cur  = self._embed(chunks[i])
                emb_next = self._embed(chunks[i + 1])
                sim      = self._cosine(emb_cur, emb_next)

                if sim > self.merging_threshold:
                    merged.append(chunks[i] + " " + chunks[i + 1])
                    i += 2
                    continue

            merged.append(chunks[i])
            i += 1

        return merged

    # ── public API ───────────────────────────

    def chunk_document(self, text: str) -> List[str]:
        sentences      = self._split_into_sentences(text)
        initial_chunks = self._initial_chunking(sentences)
        return self._double_pass_merging(initial_chunks)


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def process_mathematical_documents(
    data_path: str,
    embeddings_model=None,
) -> List[Document]:
    """
    Full pipeline: PDF → semantic chunks → LangChain Documents with metadata.

    Args:
        data_path:        Path to directory containing PDF files.
        embeddings_model: HuggingFaceEmbeddings instance (uses default if None).

    Returns:
        List of LangChain Documents ready for Pinecone upsert.
    """
    emb_model = embeddings_model or embeddings

    print("Step 1: Loading PDFs...")
    documents = load_pdf_files(data_path)
    print(f"  Loaded {len(documents)} pages")

    print("Step 2: Initialising chunker...")
    chunker = SemanticDoublePassChunker(
        embeddings_model=emb_model,
        initial_threshold=0.75,
        merging_threshold=0.65,
        chunk_size=1200,
        chunk_overlap=250,
    )
    metadata_extractor = MathMetadataExtractor()

    print("Step 3: Chunking documents...")
    processed_docs = []

    for doc_idx, doc in enumerate(documents):
        print(f"  [{doc_idx + 1}/{len(documents)}] {doc.metadata.get('source', 'unknown')}")

        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", 0)
        chunks = chunker.chunk_document(doc.page_content)

        for chunk_idx, chunk in enumerate(chunks):
            metadata = {
                "source":        source,
                "page":          page,
                "chunk_index":   chunk_idx,
                "topic":         metadata_extractor.extract_topic(chunk),
                "difficulty":    metadata_extractor.extract_difficulty(chunk),
                "concepts":      metadata_extractor.extract_concepts(chunk),
                "formula_count": metadata_extractor.count_formulas(chunk),
                "has_math":      MathFormulaDetector.has_math_content(chunk),
                "chunk_length":  len(chunk),
            }
            processed_docs.append(Document(page_content=chunk, metadata=metadata))

    print(f"\nDone. Created {len(processed_docs)} chunks from {len(documents)} pages.")
    return processed_docs


# ──────────────────────────────────────────────
# CLI entry point  (python chunker.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "Data"
    chunks = process_mathematical_documents(data_path)

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    for i, chunk in enumerate(chunks[:3]):
        m = chunk.metadata
        print(f"\n--- Chunk {i + 1} ---")
        print(f"  Source:        {m['source']}")
        print(f"  Topic:         {m['topic']}")
        print(f"  Difficulty:    {m['difficulty']}")
        print(f"  Concepts:      {', '.join(m['concepts']) or 'none'}")
        print(f"  Formula count: {m['formula_count']}")
        print(f"  Has math:      {m['has_math']}")
        print(f"  Preview:       {chunk.page_content[:200]}...")