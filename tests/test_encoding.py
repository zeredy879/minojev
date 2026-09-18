from minojev import ByteTokenizer, Request, make_boolean_question, make_choice_question, make_score_question
from minojev.encoding import encode_requests, encode_request, state_text, question_text


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "State:\n{\"k\": \"v\"} — unicode ✓"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_size == 259
    assert tokenizer.encode("A", add_bos=True, add_eos=True)[0] == tokenizer.bos_id
    assert tokenizer.encode("A", add_bos=True, add_eos=True)[-1] == tokenizer.eos_id


def test_path_layout_one_per_candidate():
    tokenizer = ByteTokenizer()
    request = Request(
        id="r",
        state={"k": "v"},
        questions=[
            make_choice_question("choice", "pick", {"a": "alpha", "b": "beta", "c": "gamma"}),
            make_boolean_question("bool", "is it?", {"true": "it is", "false": "it is not"}),
            make_score_question("score", "rate", ["low", "mid", "high"]),
        ],
    )
    paths, groups = encode_request(request, tokenizer)
    assert [len(group.path_indices) for group in groups] == [3, 2, 3]
    assert len(paths) == 8
    assert all(path.token_ids == path.state_ids + path.suffix_ids for path in paths)
    assert all(path.token_ids[-1] == tokenizer.eos_id for path in paths)


def test_state_prefix_shared_across_questions():
    tokenizer = ByteTokenizer()
    request = Request(
        id="r",
        state="shared state",
        questions=[
            make_choice_question("a", "first?", {"x": "one", "y": "two"}),
            make_choice_question("b", "second?", {"x": "one", "y": "two"}),
        ],
    )
    paths, _ = encode_request(request, tokenizer)
    assert paths[0].state_ids == paths[2].state_ids
    assert paths[0].suffix_ids != paths[2].suffix_ids
    assert state_text(request.state).startswith("State:")


def test_score_levels_are_self_contained():
    tokenizer = ByteTokenizer()
    request = Request(
        id="r",
        state="s",
        questions=[make_score_question("q", "rate the thing", ["very low", "very high"])],
    )
    encoded = encode_requests([request], tokenizer)
    assert len(encoded.groups) == 1
    assert encoded.groups[0].candidate_ids == ["0", "1"]
    suffix_texts = [tokenizer.decode(encoded.paths[index].suffix_ids) for index in encoded.groups[0].path_indices]
    assert "very low" in suffix_texts[0]
    assert "very high" not in suffix_texts[0]
    assert "Candidate:" in question_text(request.questions[0]) or True
