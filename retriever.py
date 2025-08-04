from __future__ import annotations
from typing import List, Tuple, Dict


# --- Hard Negative Mining ---
def mine_hard_negatives(
    retriever: BaseRetriever,
    queries: List[Tuple[str, str]],
    k: int = 10
) -> Dict[str, List[DocumentChunk]]:
    """
    Mine hard negatives for each query.

    Parameters
    ----------
    retriever : BaseRetriever
        The retriever to use for ranking.
    queries : List[Tuple[str, str]]
        A list of (query, positive_doc_id) pairs.
    k : int
        Number of top retrieved chunks to consider as potential hard negatives.

    Returns
    -------
    Dict[str, List[DocumentChunk]]
        A dictionary mapping query strings to a list of hard negative chunks.
    """
    hard_negatives = {}
    for query_text, pos_id in queries:
        retrieved = retriever.query(query_text, k=k+5)  # Slightly overfetch
        negatives = [chunk for chunk in retrieved if chunk.id != pos_id][:k]
        hard_negatives[query_text] = negatives
    return hard_negatives
"""
Low‑level retrieval primitives for the HackRx policy QA system.

This module implements a lightweight retrieval stack built on top of
scikit‑learn.  The goal is to enable search over a collection of
``DocumentChunk`` objects while supporting dense, sparse (BM25) and
hybrid retrieval strategies.  Each retriever maintains an in‑memory
index and exposes a simple ``query`` method returning the most
relevant chunks.

Given that packages like ``faiss`` and ``sentence‑transformers`` are
not available in this execution environment, dense retrieval uses a
TF–IDF vectoriser to approximate semantic similarity.  While this is
not as powerful as transformer‑based embeddings, it provides a strong
baseline and makes the retriever service self‑contained.

The classes defined here can be extended or replaced with more
sophisticated implementations once additional dependencies are
available.  For example, you may implement a ``DPRRetriever`` using
transformer embeddings or integrate an external vector database.

Example usage:

>>> from services.retriever import DenseRetriever, DocumentChunk
>>> docs = [DocumentChunk(id="1", text="knee surgery is covered", metadata={}),
...         DocumentChunk(id="2", text="hip replacement is not covered", metadata={})]
>>> retriever = DenseRetriever()
>>> retriever.build_index(docs)
>>> results = retriever.query("knee procedure", k=1)
>>> print(results[0].id)
"1"

Author: HackRx 6.0 team
"""


import fitz  # PyMuPDF
from typing import List

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    # scikit‑learn should be installed in the notebook environment
    NearestNeighbors = None  # type: ignore


@dataclass
class DocumentChunk:
    """Representation of a single textual chunk with optional metadata.

    ``id`` is a unique identifier for the chunk (e.g. ``doc_id/page/h1/h2``).
    ``text`` contains the raw text of the chunk.  ``metadata`` stores
    arbitrary key/value pairs such as insurer name, section labels or
    page numbers.  Metadata is not directly used by the retriever but
    may be utilised downstream to filter results or for auditing.
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise whitespace and strip leading/trailing spaces
        self.text = " ".join(self.text.strip().split())


class BaseRetriever:
    """Abstract base class for retrievers.

    Subclasses must implement ``build_index`` and ``query``.
    """

    def __init__(self) -> None:
        self._chunks: List[DocumentChunk] = []

    def build_index(self, chunks: Iterable[DocumentChunk]) -> None:
        """Build the underlying index over a collection of chunks.

        Parameters
        ----------
        chunks : Iterable[DocumentChunk]
            The document chunks to index.
        """
        raise NotImplementedError

    def query(self, query_text: str, k: int = 3) -> List[DocumentChunk]:
        """Return the top ``k`` chunks matching the query.

        Parameters
        ----------
        query_text : str
            A natural language query.
        k : int, optional
            The number of results to return (default: 3).

        Returns
        -------
        List[DocumentChunk]
            The most relevant chunks, sorted by descending relevance.
        """
        raise NotImplementedError

    def _store_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        """Persist a copy of the chunks for later retrieval.

        This method is called by ``build_index`` implementations to store
        chunk objects internally.  The retriever does not modify the
        chunks beyond normalisation performed in ``DocumentChunk``.
        """
        self._chunks = list(chunks)

    @property
    def chunks(self) -> List[DocumentChunk]:
        """Return the list of indexed chunks."""
        return self._chunks


class DenseRetriever(BaseRetriever):
    """Dense retriever based on TF–IDF and cosine similarity.

    This class uses ``TfidfVectorizer`` from scikit‑learn to compute
    high‑dimensional sparse vectors for each chunk and queries.
    During querying, the query vector is compared to all stored
    document vectors using cosine similarity.  You can specify
    additional keyword arguments to the vectoriser via the constructor.
    """

    def __init__(self, *, vectorizer: Optional[TfidfVectorizer] = None, **tfidf_kwargs: Any) -> None:
        super().__init__()
        # Use a provided vectoriser or construct a default one
        self.vectorizer: TfidfVectorizer = vectorizer or TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_df=0.95,
            min_df=1,
            **tfidf_kwargs,
        )
        self._doc_matrix: Optional[np.ndarray] = None

    def build_index(self, chunks: Iterable[DocumentChunk]) -> None:
        # Persist chunks
        self._store_chunks(chunks)
        # Fit the vectoriser on all documents and compute document term matrix
        texts = [chunk.text for chunk in self._chunks]
        if not texts:
            raise ValueError("No documents provided to build the index.")
        self._doc_matrix = self.vectorizer.fit_transform(texts).astype(np.float32)

    def query(self, query_text: str, k: int = 3) -> List[DocumentChunk]:
        if self._doc_matrix is None:
            raise RuntimeError("The index has not been built; call build_index() first.")
        # Transform the query to TF–IDF space
        query_vec = self.vectorizer.transform([query_text]).astype(np.float32)
        # Compute cosine similarities
        sims = cosine_similarity(query_vec, self._doc_matrix)[0]
        # Get top k indices
        top_indices = np.argsort(sims)[::-1][:k]
        return [self._chunks[i] for i in top_indices]


class SparseBM25Retriever(BaseRetriever):
    """Sparse BM25 retriever.

    Implements a variant of BM25 using TF–IDF style weighting.  BM25 is a
    widely used ranking function in information retrieval that rewards
    term frequency while penalising long documents.  This implementation
    follows the Okapi BM25 formulation with configurable parameters
    ``k1`` and ``b``.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.k1 = k1
        self.b = b
        self._doc_freq: Dict[str, int] = {}
        self._avg_len: float = 0.0
        self._vocabulary: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._vectorizer: Optional[TfidfVectorizer] = None

    def build_index(self, chunks: Iterable[DocumentChunk]) -> None:
        self._store_chunks(chunks)
        # Build a term frequency matrix and compute document lengths
        texts = [chunk.text for chunk in self._chunks]
        if not texts:
            raise ValueError("No documents provided to build the index.")
        # Use TfidfVectorizer to obtain term counts (use idf_=False, norm=None)
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", use_idf=False, norm=None)
        tf_matrix = self._vectorizer.fit_transform(texts)
        self._vocabulary = self._vectorizer.get_feature_names_out().tolist()
        # Compute document frequency for each term
        df = np.asarray((tf_matrix > 0).sum(axis=0)).ravel()
        self._doc_freq = {term: int(df[idx]) for idx, term in enumerate(self._vocabulary)}
        # Compute average document length
        doc_lengths = np.asarray(tf_matrix.sum(axis=1)).ravel()
        self._avg_len = float(doc_lengths.mean())
        # Store raw term frequencies per document for later scoring
        self._matrix = tf_matrix.astype(np.float32)

    def _bm25_score(self, query: str) -> List[float]:
        assert self._matrix is not None and self._vectorizer is not None
        # Tokenise query using the same preprocessing as vectoriser
        analyzer = self._vectorizer.build_analyzer()
        query_terms = analyzer(query)
        n_docs = len(self._chunks)
        scores = np.zeros(n_docs, dtype=np.float32)
        # Precompute doc lengths
        doc_lengths = np.asarray(self._matrix.sum(axis=1)).ravel()
        for term in query_terms:
            if term not in self._doc_freq:
                continue
            df = self._doc_freq[term]
            # Inverse document frequency
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            term_index = self._vocabulary.index(term)
            # Term frequency vector for term across documents
            f = self._matrix[:, term_index].toarray().ravel()
            # BM25 per document
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * (doc_lengths / self._avg_len))
            scores += idf * (numerator / (denominator + 1e-8))
        return scores.tolist()

    def query(self, query_text: str, k: int = 3) -> List[DocumentChunk]:
        if self._matrix is None:
            raise RuntimeError("The index has not been built; call build_index() first.")
        scores = self._bm25_score(query_text)
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._chunks[i] for i in top_indices]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense and sparse backends.

    A weighted combination of dense and sparse retrieval scores often
    outperforms either method alone.  This class wraps a dense
    retriever (e.g. ``DenseRetriever``) and a sparse retriever (e.g.
    ``SparseBM25Retriever``), linearly combines their scores, and
    returns the top candidates.  The weighting parameter ``alpha``
    controls the relative contribution of the dense backend; ``alpha=1``
    uses only dense scores, ``alpha=0`` uses only sparse scores.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseBM25Retriever] = None,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseBM25Retriever()
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.alpha = alpha

    def build_index(self, chunks: Iterable[DocumentChunk]) -> None:
        # Both retrievers need the same chunk order for correct alignment
        chunk_list = list(chunks)
        # Persist chunks in this class as well
        self._store_chunks(chunk_list)
        # Build both indexes
        self.dense_retriever.build_index(chunk_list)
        self.sparse_retriever.build_index(chunk_list)

    def query(self, query_text: str, k: int = 3) -> List[DocumentChunk]:
        # Compute dense scores
        dense_results = self._get_scores(self.dense_retriever, query_text)
        # Compute sparse scores
        sparse_results = self._get_scores(self.sparse_retriever, query_text)
        # Combine scores
        scores = self.alpha * dense_results + (1 - self.alpha) * sparse_results
        # Return top k chunks
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._chunks[i] for i in top_indices]

    def _get_scores(self, retriever: BaseRetriever, query_text: str) -> np.ndarray:
        """Helper to get per-document similarity scores from a retriever.

        The dense retriever does not expose scores, so we recompute them.
        This method returns an array of floats of shape (n_chunks,).
        """
        if isinstance(retriever, DenseRetriever):
            # Transform all docs and query into TF–IDF space for scoring
            assert retriever._doc_matrix is not None
            q_vec = retriever.vectorizer.transform([query_text]).astype(np.float32)
            sims = cosine_similarity(q_vec, retriever._doc_matrix)[0]
            return sims
        elif isinstance(retriever, SparseBM25Retriever):
            return np.array(retriever._bm25_score(query_text), dtype=np.float32)
        else:
            raise TypeError(f"Unsupported retriever type: {type(retriever)}")


def load_chunks_from_jsonl(path: str) -> List[DocumentChunk]:
    """Load a list of ``DocumentChunk`` objects from a JSONL file.

    Each line in the JSONL should be a JSON object with at least the
    keys ``id`` and ``text``.  Any other keys are stored as metadata.

    Parameters
    ----------
    path : str
        Path to the JSONL file.

    Returns
    -------
    List[DocumentChunk]
        The loaded chunks.
    """
    chunks: List[DocumentChunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing line {line_no}: {e}") from e
            if "id" not in obj or "text" not in obj:
                raise ValueError(f"Line {line_no} missing required keys: {obj}")
            meta = {k: v for k, v in obj.items() if k not in {"id", "text"}}
            chunks.append(DocumentChunk(id=obj["id"], text=obj["text"], metadata=meta))
    return chunks

def extract_chunks_from_pdf(path: str) -> List[DocumentChunk]:
    doc = fitz.open(path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        window_size = 3
        for i in range(0, len(sentences), window_size):
            chunk_text = ". ".join(sentences[i:i + window_size]) + "."
            chunk = DocumentChunk(id=f"{page_num+1}_{i // window_size}", text=chunk_text)
            chunks.append(chunk)
    return chunks



# def demo_retriever() -> None:
#     """Standalone demo for running the retriever on example data.

#     This function can be executed as a script.  It loads sample
#     documents, builds a dense, sparse and hybrid retriever, and prints
#     out the top results for a few test queries.  Feel free to modify
#     the sample text to experiment with different retrieval behaviours.
#     """
#     sample_chunks = [
#         DocumentChunk(id="1", text="Knee surgery is covered under the policy."),
#         DocumentChunk(id="2", text="Hip replacement is not covered by this insurer."),
#         DocumentChunk(id="3", text="Shoulder arthroscopy coverage depends on the plan."),
#         DocumentChunk(id="4", text="Dental extraction is excluded except in accidents."),
#         DocumentChunk(id="5", text="Hospitalisation for knee arthroplasty is reimbursable."),
#     ]
#     query = "46-year-old with knee surgery"
#     print(f"Query: {query}\n")

#     # Dense retriever
#     dense = DenseRetriever()
#     dense.build_index(sample_chunks)
#     print("Dense top 3 results:")
#     for doc in dense.query(query, k=3):
#         print(f"  {doc.id}: {doc.text}")
#     print()

#     # Sparse retriever
#     sparse = SparseBM25Retriever()
#     sparse.build_index(sample_chunks)
#     print("BM25 top 3 results:")
#     for doc in sparse.query(query, k=3):
#         print(f"  {doc.id}: {doc.text}")
#     print()

#     # Hybrid retriever
#     hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, alpha=0.5)
#     # No need to rebuild: hybrid uses existing dense/sparse indexes
#     hybrid._store_chunks(sample_chunks)
#     print("Hybrid top 3 results:")
#     for doc in hybrid.query(query, k=3):
#         print(f"  {doc.id}: {doc.text}")


# if __name__ == "__main__":  # pragma: no cover
#     demo_retriever()


def demo_retriever_from_pdfs(pdf_paths: List[str], query: str) -> None:
    all_chunks = []
    for pdf_path in pdf_paths:
        chunks = extract_chunks_from_pdf(pdf_path)
        all_chunks.extend(chunks)
    print(f"Loaded {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
    print(f"\nQuery: {query}\n")

    # --- Validate positive chunk IDs ---
    labeled_pairs = [
        ("46-year-old, knee surgery, Pune, 3-month-old policy", "3_2"),
        ("Knee arthroplasty claim after accident", "5_1"),
        ("Expenses before hospital admission", "2_1"),
        ("Reimbursement after discharge", "2_3"),
        ("Delivery by caesarean", "4_0"),
        ("Hospital bills for newborn baby", "4_2"),
        ("Can I claim for CPAP machine?", "6_1"),
        ("Treatment taken at home for asthma", "7_0"),
        ("Is hip replacement surgery covered?", "3_4"),
        ("Policy waiting time for orthopedic procedures", "1_2"),
        ("Does this policy cover ambulance?", "2_0"),
        ("Is air ambulance allowed?", "2_2"),
        ("Organ donor surgery expenses", "5_3"),
        ("Transplant donor fees", "5_3"),
        ("Maternity coverage under this plan", "4_0"),
        ("Miscarriage medical bills", "4_1"),
        ("Waiting period for maternity claims", "4_4"),
        ("Is circumcision covered?", "6_4"),
        ("What surgeries are excluded?", "6_5"),
        ("Treatment for cosmetic dental issues", "6_2"),
        ("Self-inflicted injury coverage", "6_6"),
        ("Policy exclusion: substance abuse", "6_6"),
        ("Health checkup reimbursement", "5_5"),
        ("Are routine health checkups included?", "5_5"),
        ("Rehabilitation expenses", "7_1"),
        ("Pain management therapy costs", "7_1"),
        ("Coverage for AYUSH treatment", "5_6"),
        ("Homeopathic hospitalisation", "5_6"),
        ("Automatic sum insured restoration", "5_7"),
        ("Clauses for restoring coverage limit", "5_7"),
        ("Coverage for ICU expenses", "3_5"),
        ("Room rent and shared accommodation", "3_6"),
        ("Chronic asthma care at home", "7_0"),
        ("In utero fetal surgery covered?", "4_5"),
        ("Repatriation of mortal remains", "7_2"),
        ("Companion travel for emergencies", "7_3"),
        ("Assisted reproduction reimbursement", "4_6"),
        ("Coverage for infertility treatment", "4_6"),
        ("Dental surgery after accident", "3_3"),
        ("Implantable device claim", "6_1"),
        ("Modern treatment costs", "5_8"),
        ("Treatment abroad eligibility", "6_7"),
        ("Claim for cancer treatment", "3_2"),
        ("Critical illness lump sum", "4_8"),
        ("Is dengue hospitalisation covered?", "3_0"),
        ("Policy for chronic kidney disease", "6_8"),
        ("Waiting period for pre-existing diseases", "1_1"),
        ("Grace period for policy renewal", "1_3"),
        ("Eligibility for cumulative bonus", "5_9"),
        ("Free medical second opinion", "5_4"),
        ("What is not covered in this policy?", "6_5"),
    ]

    chunk_ids = set(chunk.id for chunk in all_chunks)
    missing_ids = [pid for _, pid in labeled_pairs if pid not in chunk_ids]

    if missing_ids:
        print("⚠️ WARNING: Some positive chunk IDs not found in current index:")
        for pid in missing_ids:
            print(f"  - Missing: {pid}")
    else:
        print("✅ All positive chunk IDs are present in the indexed chunks.")

    dense = DenseRetriever()
    dense.build_index(all_chunks)
    print("Dense top 20 results:")
    for doc in dense.query(query, k=20):
        print(f"  {doc.id}: {doc.text}")
    print()

    sparse = SparseBM25Retriever()
    sparse.build_index(all_chunks)
    print("BM25 top 20 results:")
    for doc in sparse.query(query, k=20):
        print(f"  {doc.id}: {doc.text}")
    print()

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, alpha=0.5)
    hybrid._store_chunks(all_chunks)
    print("Hybrid top 20 results:")
    for doc in hybrid.query(query, k=20):
        print(f"  {doc.id}: {doc.text}")

    # --- Hard Negative Mining ---
    queries = [
        ("46-year-old with knee surgery", "3_2"),  # (query, positive_chunk_id)
        ("hip replacement after 6 months", "7_1"),
    ]
    hard_negs = mine_hard_negatives(hybrid, queries, k=5)
    for q, negs in hard_negs.items():
        print(f"\nHard negatives for: '{q}'")
        for neg in negs:
            print(f"  {neg.id}: {neg.text}")
        # --- Rerank with LlamaRank via Together API ---
    import together
    import os

    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not TOGETHER_API_KEY:
        print("❌ TOGETHER_API_KEY not found in environment variables.")
        return

    client = together.Together(api_key=TOGETHER_API_KEY)

    print("\n🔁 Reranking hybrid top 20 using LlamaRank...\n")
    top_docs = hybrid.query(query, k=20)
    texts = [doc.text for doc in top_docs]

    llama_result = client.rerank.create(
        model="Salesforce/Llama-Rank-V1",
        query=query,
        documents=texts,
        top_n=5
    )

    print("🔝 LlamaRank Top 5 Chunks:")
    for item in llama_result.results:
       rank = item.index
       print(f"{rank + 1}. {top_docs[rank].id}: {top_docs[rank].text}")

if __name__ == "__main__":
    import os

    pdf_dir = "/Users/apple/Downloads"
    pdf_files = [
        "EDLHLGA23009V012223.pdf",
        "MCIHLIA23023V012223.pdf",
        "SGLHLGP23014V012223.pdf",
        "NAVHLIP23003V012223.pdf",
        "RQBHLGP23036V012223.pdf",
        "ICIHLIP22012V012223.pdf",
        "SGLHLGP23026V012223.pdf",
        "SBIHLGP23005V012223.pdf",
        "NBHTGBP22011V012223.pdf",
        "NBHHLIP23007V052223.pdf",
        "BAJHLIP23020V012223-2.pdf",
        "HDFHLIP23024V072223.pdf",
        "CHOTGDP23004V012223.pdf",
        "RSAHLIP23029V012223.pdf",
        "ICIHLGP23027V012223.pdf",
        "SHAHLGP23015V012223.pdf",
        "BAJHLIP23020V012223.pdf",
        "SHAHLIP23017V012223.pdf",
    ]
    query = "46-year-old with knee surgery"
    full_paths = [os.path.join(pdf_dir, f) for f in pdf_files]

    demo_retriever_from_pdfs(full_paths, query)

   