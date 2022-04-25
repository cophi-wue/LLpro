from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from pydantic import BaseModel, Field

from sapienzanlp.data.model_io.sentence import SrlSentence
from sapienzanlp.data.model_io.word import Word


class DocumentIn(BaseModel):
    text: str
    lang: Optional[str] = None


class BatchIn(BaseModel):
    texts: List[DocumentIn]


class WordOut(BaseModel):
    index: int
    raw_text: str = Field(description="Raw text of the token", alias="rawText")

    class Config:
        allow_population_by_field_name = True


class ArgumentOut(BaseModel):
    role: str
    score: float
    span: List[int]
    # start_index: int
    # end_index: int


class PredicateOut(BaseModel):
    frame_name: str = Field(description="Name of the Frame", alias="frameName")
    roles: List[ArgumentOut] = None

    class Config:
        allow_population_by_field_name = True


class AnnotationOut(BaseModel):
    token_index: int = Field(description="Index the token", alias="tokenIndex")
    verbatlas: PredicateOut = Field(
        default=PredicateOut(frame_name="_", roles=[]), description="Index the token"
    )
    english_propbank: PredicateOut = Field(
        default=PredicateOut(frame_name="_", roles=[]),
        description="Index the token",
        alias="englishPropbank",
    )
    chinese_propbank: PredicateOut = Field(
        default=PredicateOut(frame_name="_", roles=[]),
        description="Index the token",
        alias="chinesePropbank",
    )
    german_propbank: PredicateOut = Field(
        default=PredicateOut(frame_name="_", roles=[]),
        description="Index the token",
        alias="germanPropbank",
    )
    pdt_vallex: PredicateOut = Field(
        default=PredicateOut(frame_name="_", roles=[]),
        description="Index the token",
        alias="pdtVallex",
    )
    spanish_ancora: PredicateOut = Field(
        default=PredicateOut(frame_name="_", roles=[]),
        description="Index the token",
        alias="spanishAncora",
    )
    catalan_ancora: PredicateOut = Field(
        default=PredicateOut(frame_name="_", roles=[]),
        description="Index the token",
        alias="catalanAncora",
    )

    class Config:
        allow_population_by_field_name = True


class DocumentOut(BaseModel):
    tokens: List[WordOut] = []
    annotations: List[AnnotationOut] = []


class BatchOut(BaseModel):
    sentences: List[DocumentOut] = []


@dataclass
class Doc:
    doc_id: int
    sid: int
    lang: str
    text: str
    tokens: Optional[List[Word]] = None
    annotations: Dict[str, SrlSentence] = field(default_factory=lambda: defaultdict(list))
    # ca_annotations: Optional[SrlSentence] = None
    # cs_annotations: Optional[SrlSentence] = None
    # de_annotations: Optional[SrlSentence] = None
    # en_annotations: Optional[SrlSentence] = None
    # es_annotations: Optional[SrlSentence] = None
    # zh_annotations: Optional[SrlSentence] = None
