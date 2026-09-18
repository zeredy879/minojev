import torch

from minojev import CHOICE, Request, make_boolean_question, make_choice_question, make_score_question


def _groups_for(request, model):
    encoded = model.encode([request])
    hidden = model.forward_paths(encoded)
    kinds = [group.kind for group in encoded.groups]
    outputs = model.head(hidden, encoded.groups, kinds)
    return encoded, hidden, outputs


def test_probability_semantics(tiny_model):
    request = Request(
        id="r",
        state="state",
        questions=[
            make_choice_question("c", "pick", {"a": "alpha", "b": "beta", "c": "gamma"}),
            make_boolean_question("b", "is it true?"),
            make_score_question("s", "rate", ["l0", "l1", "l2", "l3"]),
        ],
    )
    encoded, hidden, outputs = _groups_for(request, tiny_model)
    for output in outputs:
        assert torch.allclose(output.probabilities.sum(), torch.tensor(1.0), atol=1e-5)
        assert (output.probabilities >= 0).all()
    choice, boolean, score = outputs
    assert choice.logits.shape == (3,)
    assert boolean.probabilities.shape == (2,)
    assert torch.allclose(score.expected, torch.sum(torch.arange(4.0) * score.probabilities), atol=1e-5)

    normalized = tiny_model.head.norm(hidden)
    scalar = tiny_model.head.scalar(normalized).squeeze(-1).detach().float()
    boolean_pair = encoded.groups[1].path_indices
    expected_true = torch.softmax(torch.tensor([0.0, float(scalar[boolean_pair[1]] - scalar[boolean_pair[0]])]), -1)[1]
    assert torch.allclose(boolean.probabilities[1], expected_true, atol=1e-5)
    assert torch.allclose(boolean.probabilities[0], 1 - boolean.probabilities[1], atol=1e-5)


def test_choice_permutation_equivariance(tiny_model):
    request = Request(
        id="r",
        state="state",
        questions=[make_choice_question("c", "pick", {"a": "alpha", "b": "beta", "c": "gamma", "d": "delta"})],
    )
    encoded, hidden, outputs = _groups_for(request, tiny_model)
    base = outputs[0].probabilities
    permutation = [2, 0, 3, 1]
    group = encoded.groups[0]
    reordered = type(group)(
        request_index=group.request_index,
        question_index=group.question_index,
        kind=group.kind,
        path_indices=[group.path_indices[index] for index in permutation],
        candidate_ids=[group.candidate_ids[index] for index in permutation],
    )
    permuted = tiny_model.head(hidden, [reordered], [CHOICE])[0].probabilities
    assert torch.allclose(base[permutation], permuted, atol=1e-5)


def test_boolean_is_a_two_path_comparison(tiny_model):
    request = Request(
        id="r",
        state="state",
        questions=[make_boolean_question("b", "the claim", {"true": "claim holds", "false": "claim fails"})],
    )
    encoded, hidden, outputs = _groups_for(request, tiny_model)
    assert len(encoded.paths) == 2
    normalized = tiny_model.head.norm(hidden)
    scalar = tiny_model.head.scalar(normalized).squeeze(-1).float()
    false_index, true_index = encoded.groups[0].path_indices
    assert torch.allclose(
        outputs[0].probabilities[1],
        torch.sigmoid(scalar[true_index] - scalar[false_index]),
        atol=1e-5,
    )
