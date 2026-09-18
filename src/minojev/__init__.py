"""minojev: a minimal generation-free decision model.

States and questions in, complete probability distributions out, with zero
output-token decoding. The package supports three primitives (choice,
boolean, score), parallel decisions over shared state, a trainable decision
head, and a dependency-free tiny backbone for offline reproduction.
"""

from .backbone import HFBackbone, TinyConfig, TinyLM, load_hf_backbone, resolve_device
from .bench import BenchOptions, benchmark_model, compare_modes
from .calibrate import Calibration, fit_calibration
from .encoding import EncodedBatch, Path, QuestionGroup, encode_requests
from .heads import DecisionHead, GroupOutput
from .metrics import aggregate
from .model import DecisionModel, ScoreOptions
from .synth import generate_request, generate_split
from .tokenizer import ByteTokenizer
from .types import (
    BOOLEAN,
    CHOICE,
    MAX_CHOICE,
    MAX_LEVELS,
    MIN_CHOICE,
    MIN_LEVELS,
    SCORE,
    Candidate,
    Question,
    Request,
    ValidationError,
    make_boolean_question,
    make_choice_question,
    make_score_question,
    request_from_object,
    request_to_object,
)

__version__ = "0.1.0"

__all__ = [
    "BOOLEAN",
    "CHOICE",
    "SCORE",
    "MIN_CHOICE",
    "MAX_CHOICE",
    "MIN_LEVELS",
    "MAX_LEVELS",
    "BenchOptions",
    "ByteTokenizer",
    "Calibration",
    "Candidate",
    "DecisionHead",
    "DecisionModel",
    "EncodedBatch",
    "GroupOutput",
    "HFBackbone",
    "Path",
    "Question",
    "QuestionGroup",
    "Request",
    "ScoreOptions",
    "TinyConfig",
    "TinyLM",
    "ValidationError",
    "aggregate",
    "benchmark_model",
    "compare_modes",
    "encode_requests",
    "fit_calibration",
    "generate_request",
    "generate_split",
    "load_hf_backbone",
    "make_boolean_question",
    "make_choice_question",
    "make_score_question",
    "request_from_object",
    "request_to_object",
    "resolve_device",
]
