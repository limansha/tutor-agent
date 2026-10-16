from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
class ChatFilters(BaseModel):
    """Metadata filters to shrink the vector search space before cosine search."""

    book_no: int | None = None
    year: int | None = None
    reading_no: int | None = None
    module_no: str | None = None
    lo_code: str | None = None
    section: str | None = Field(
        default=None,
        description="Restrict to one part of the notes: exam_focus, module_content, "
        "key_concepts, formulas, module_quiz, answer_key.",
    )


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to search against the corpus.")
    filters: ChatFilters | None = None
    top_k: int = Field(default=3, ge=1, le=50)
    parent_expansion: bool = Field(
        default=True,
        description="Fetch parent chunks for retrieved children and pass the full section context.",
    )
    section_expansion: bool = Field(
        default=False,
        description="Also return every sibling chunk of the matched module (same book/reading/module) "
        "so the caller gets the complete section content.",
    )


class MCQChatRequest(BaseModel):
    """Agentic chat request: a question plus multiple-choice options."""

    question: str = Field(..., min_length=1)
    options: dict[str, str] = Field(..., min_length=2)
    filters: ChatFilters | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    parent_expansion: bool = Field(default=True)
    include_sources: bool = Field(
        default=False,
        description="Include the retrieved `sources` array in the response.",
    )


class OptionRejection(BaseModel):
    option_label: str  # e.g. "A"
    option_text: str
    reason: str  # why this option is NOT correct


class MCQAnswer(BaseModel):
    """Structured answer produced by the agent (response_format schema)."""

    selected_option_label: str
    selected_option_text: str
    explanation: str  # why the selected option is correct
    rejections: list[OptionRejection]  # why the other options are wrong


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk (enables filtering and referencing)."""

    book_no: int | None = None
    book_title: str | None = None
    source_path: str | None = None
    year: int | None = None
    reading_no: int | None = None
    reading_title: str | None = None
    module_no: str | None = None
    module_title: str | None = None
    lo_code: str | None = None
    section: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None
    token_count: int | None = None
    doc_url: str | None = None
    parent_chunk_id: int | None = None


class SourceChunk(BaseModel):
    chunk_id: int | None = None
    role: str  # 'parent' | 'child'
    text: str
    score: float | None = None  # cosine similarity (1 = identical)
    metadata: ChunkMetadata


class FileIngestResult(BaseModel):
    filename: str
    book_no: int | None = None
    pages: int = 0
    parents: int = 0
    children: int = 0
    tokens: int = 0
    status: str = "ok"  # ok | skipped | error
    detail: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    sources: list[SourceChunk]
    elapsed_ms: int


class MCQChatResponse(BaseModel):
    question: str
    answer: MCQAnswer
    sources: list[SourceChunk] | None = None
    model: str | None = None
    elapsed_ms: int


class ServiceUnavailable(BaseModel):
    error: str
    hint: str | None = None


# ---------------------------------------------------------------------------
# Questions (fetch by requested topics with exact-first/semantic-fallback)
# ---------------------------------------------------------------------------
class QuestionsRequest(BaseModel):
    topics: list[str] = Field(
        default_factory=list,
        description="Exact topic tags to match against question_topics (GIN).",
    )
    search_topics: list[str] = Field(
        default_factory=list,
        description="Topics to match semantically; only runs when non-empty.",
    )
    page_number: int = Field(default=1, ge=1, description="1-based batch number.")
    page_size: int = Field(default=100, ge=1, le=100,
                           description="Questions per batch (max 100).")


class Question(BaseModel):
    """One frm_questions row as returned by GET /api/questions."""

    qid: int
    question_text: str
    options: dict[str, str]
    answer: str
    has_answer_in_doc: bool = False
    question_topics: list[str] = Field(default_factory=list)
    prerequisites: dict = Field(default_factory=dict)
    pdf_name: str
    source_path: str
    page_number: int


class TopicMatch(BaseModel):
    """How a requested topic was resolved."""

    topic: str
    match_type: str  # 'exact' | 'semantic'


class UncoveredTopic(BaseModel):
    topic: str
    detail: str = "no questions found for this topic"


class QuestionsResponse(BaseModel):
    topics: list[TopicMatch]
    total: int
    has_more: bool = False
    uncovered_topics: list[UncoveredTopic]
    questions: list[Question]
    elapsed_ms: int


# ---------------------------------------------------------------------------
# Check answer (deterministic validation against frm_questions.answer)
# ---------------------------------------------------------------------------
class AnswerCheckItem(BaseModel):
    qid: int
    answer: str | None = Field(
        default=None, description="The option label the user chose; null if unanswered."
    )


class AnswerCheckRequest(BaseModel):
    answers: list[AnswerCheckItem] = Field(
        ..., min_length=1, max_length=100,
        description="One entry per question to validate.",
    )


class AnswerCheckResult(BaseModel):
    qid: int
    chosen_answer: str | None = None
    correct_answer: str | None = None
    is_correct: bool = False


class AnswerCheckResponse(BaseModel):
    results: list[AnswerCheckResult]
    total: int
    correct: int
    elapsed_ms: int