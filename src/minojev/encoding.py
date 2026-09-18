"""Turn requests into candidate paths for one forward pass.

Every question becomes one path per candidate. A path is the state text
followed by the question text and one candidate description. The decision is
read from the hidden state at the final path token, so no output token is ever
decoded.

The split between ``state_ids`` and ``suffix_ids`` enables shared-prefix
scoring: a state is prefilled once and reused across every question and
candidate that references it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import BOOLEAN, CHOICE, SCORE, Question, Request, serialize_state

PATH_TEMPLATE = "Candidate: {description}\nDecision:"


@dataclass
class Path:
    state_ids: list[int]
    suffix_ids: list[int]
    request_index: int
    question_index: int
    candidate_index: int
    kind: str

    @property
    def token_ids(self) -> list[int]:
        return self.state_ids + self.suffix_ids


@dataclass
class QuestionGroup:
    request_index: int
    question_index: int
    kind: str
    path_indices: list[int]
    candidate_ids: list[str]


@dataclass
class EncodedBatch:
    requests: list[Request]
    paths: list[Path]
    groups: list[QuestionGroup]
    state_keys: list[str] = field(default_factory=list)


def state_text(state) -> str:
    return f"State:\n{serialize_state(state)}\n"


def question_text(question: Question) -> str:
    return "\n".join([f"Question type: {question.kind}", "Question:", question.instructions]) + "\n"


def candidate_text(question: Question, candidate_index: int) -> str:
    description = question.candidates[candidate_index].description
    return PATH_TEMPLATE.format(description=description)


def encode_request(request: Request, tokenizer, add_eos: bool = True) -> tuple[list[Path], list[QuestionGroup]]:
    state_ids = tokenizer.encode(state_text(request.state))
    paths: list[Path] = []
    groups: list[QuestionGroup] = []
    for question_index, question in enumerate(request.questions):
        skeleton = question_text(question)
        indices = []
        for candidate_index in range(len(question.candidates)):
            suffix = tokenizer.encode(skeleton + candidate_text(question, candidate_index))
            if add_eos:
                suffix = suffix + [tokenizer.eos_id]
            indices.append(len(paths))
            paths.append(
                Path(
                    state_ids=list(state_ids),
                    suffix_ids=suffix,
                    request_index=0,
                    question_index=question_index,
                    candidate_index=candidate_index,
                    kind=question.kind,
                )
            )
        groups.append(
            QuestionGroup(
                request_index=0,
                question_index=question_index,
                kind=question.kind,
                path_indices=indices,
                candidate_ids=question.candidate_ids,
            )
        )
    return paths, groups


def encode_requests(requests: list[Request], tokenizer, add_eos: bool = True) -> EncodedBatch:
    paths: list[Path] = []
    groups: list[QuestionGroup] = []
    for request_index, request in enumerate(requests):
        request_paths, request_groups = encode_request(request, tokenizer, add_eos=add_eos)
        offset = len(paths)
        for path in request_paths:
            path.request_index = request_index
        for group in request_groups:
            group.request_index = request_index
            group.path_indices = [index + offset for index in group.path_indices]
        paths.extend(request_paths)
        groups.extend(request_groups)
    return EncodedBatch(requests=requests, paths=paths, groups=groups, state_keys=[request.state_key for request in requests])


def pad_paths(paths: list[Path], pad_id: int, device) -> tuple:
    import torch

    lengths = [len(path.token_ids) for path in paths]
    width = max(lengths)
    tokens = torch.full((len(paths), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(paths), width), dtype=torch.long, device=device)
    for row, path in enumerate(paths):
        ids = path.token_ids
        tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[row, : len(ids)] = 1
    return tokens, mask


def pad_sequences(sequences: list[list[int]], pad_id: int, device) -> tuple:
    import torch

    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths)
    tokens = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[row, : len(sequence)] = 1
    return tokens, mask
