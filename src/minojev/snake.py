"""Snake decision task and agent rollout.

A snake state is structured JSON: grid size, the body (head first), the food
cell, the heading, and the step index. Each step asks six independent
questions in one forward pass:

- one choice over the four move directions (teacher: safe moves that keep the
  food reachable score higher, blocked or trapping moves score near zero);
- four booleans, one per direction ("is moving up safe?");
- one score over distance buckets ("how far is the food?").

A rollout controller masks the choice distribution with the safety booleans
and follows the surviving argmax, so a self-colliding or reversed move can
never be taken without being reported as forced.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, replace

from .model import ScoreOptions
from .types import Request, make_boolean_question, make_choice_question, make_score_question

DIRECTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
DISTANCE_LEVELS = ("adjacent", "close", "medium", "far")
DISTANCE_BUCKETS = ((0, 1), (2, 3), (4, 6), (7, 10**9))


@dataclass(frozen=True)
class Snake:
    size: int
    body: tuple[tuple[int, int], ...]
    food: tuple[int, int]
    heading: str

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


def in_bounds(size: int, cell: tuple[int, int]) -> bool:
    row, column = cell
    return 0 <= row < size and 0 <= column < size


def step_cell(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    delta = DELTAS[direction]
    return cell[0] + delta[0], cell[1] + delta[1]


def manhattan(first: tuple[int, int], second: tuple[int, int]) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def obstacles(snake: Snake) -> frozenset[tuple[int, int]]:
    return frozenset(snake.body[1:])


def direction_safe(snake: Snake, direction: str) -> bool:
    target = step_cell(snake.head, direction)
    return in_bounds(snake.size, target) and target not in obstacles(snake)


def legal_moves(snake: Snake) -> list[str]:
    return [direction for direction in DIRECTIONS if direction_safe(snake, direction)]


def food_distance(snake: Snake, start: tuple[int, int] | None = None) -> int | None:
    """Breadth-first distance from the head (or ``start``) to the food."""
    origin = snake.head if start is None else start
    blocked = obstacles(snake) - {origin}
    if snake.food in blocked or not in_bounds(snake.size, origin):
        return None
    if origin == snake.food:
        return 0
    queue = deque([(origin, 0)])
    seen = {origin}
    while queue:
        current, depth = queue.popleft()
        for direction in DIRECTIONS:
            target = step_cell(current, direction)
            if target in seen or target in blocked or not in_bounds(snake.size, target):
                continue
            if target == snake.food:
                return depth + 1
            seen.add(target)
            queue.append((target, depth + 1))
    return None


def reachable_area(snake: Snake, start: tuple[int, int] | None = None) -> int:
    """Number of free cells reachable from the head without crossing the body."""
    origin = snake.head if start is None else start
    blocked = obstacles(snake) - {origin}
    if not in_bounds(snake.size, origin):
        return 0
    queue = deque([origin])
    seen = {origin}
    while queue:
        current = queue.popleft()
        for direction in DIRECTIONS:
            target = step_cell(current, direction)
            if target in seen or target in blocked or not in_bounds(snake.size, target):
                continue
            seen.add(target)
            queue.append(target)
    return len(seen)


def advance(snake: Snake, direction: str) -> tuple[Snake, bool]:
    """Move the head; grow when the food is eaten. Returns (snake, ate)."""
    target = step_cell(snake.head, direction)
    ate = target == snake.food
    body = (target,) + (snake.body if ate else snake.body[:-1])
    return replace(snake, body=body, heading=direction), ate


def generate_snake(rng: random.Random, size: int = 6, length: int = 3) -> Snake:
    while True:
        row = rng.randrange(size)
        column = rng.randrange(length - 1, size)
        body = tuple((row, column - offset) for offset in range(length))
        snake = Snake(size=size, body=body, food=body[0], heading="right")
        free = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if (r, c) not in body
        ]
        candidates = []
        for food in free:
            probe = replace(snake, food=food)
            distance = food_distance(probe)
            if distance is not None and 2 <= distance <= 2 * size:
                candidates.append(food)
        if candidates:
            return replace(snake, food=rng.choice(candidates))


def _distance_level(snake: Snake, start: tuple[int, int] | None = None) -> int:
    origin = snake.head if start is None else start
    current = manhattan(origin, snake.food)
    for index, (low, high) in enumerate(DISTANCE_BUCKETS):
        if low <= current <= high:
            return index
    return len(DISTANCE_LEVELS) - 1


def _move_score(snake: Snake) -> float:
    """Lower is better: distance to food plus a penalty for shrinking the arena."""
    distance = food_distance(snake)
    area = reachable_area(snake)
    danger = max(0, snake.length + 2 - area)
    if distance is None:
        return 40.0 + danger + (0.5 * (1 + area) ** -1)
    return distance + 4.0 * danger


def _move_teacher(snake: Snake, temperature: float = 2.5) -> dict[str, float]:
    scores = {}
    for direction in DIRECTIONS:
        if not direction_safe(snake, direction):
            continue
        moved, _ = advance(snake, direction)
        scores[direction] = _move_score(moved)
    weights = {direction: 0.0 for direction in DIRECTIONS}
    if scores:
        best = min(scores.values())
        for direction, value in scores.items():
            weights[direction] = math.exp(-temperature * (value - best))
    total = sum(weights.values())
    return {direction: (weights[direction] + 1e-6) / (total + 4e-6) for direction in DIRECTIONS}


def _best_move(snake: Snake) -> str:
    moves = legal_moves(snake)
    if not moves:
        return DIRECTIONS[0]
    scored = []
    for direction in moves:
        moved, _ = advance(snake, direction)
        scored.append((_move_score(moved), manhattan(step_cell(snake.head, direction), snake.food), direction))
    scored.sort()
    return scored[0][2]


def state_object(snake: Snake, step: int = 0, eaten: int = 0) -> dict:
    return {
        "size": snake.size,
        "step": step,
        "heading": snake.heading,
        "snake": [[row, column] for row, column in snake.body],
        "food": [snake.food[0], snake.food[1]],
        "eaten": eaten,
    }


def state_request(snake: Snake, step: int = 0, eaten: int = 0) -> Request:
    questions = [
        make_choice_question(
            "move",
            "Which move should the snake make to reach the food?",
            {direction: f"move {direction}" for direction in DIRECTIONS},
        ),
    ]
    gold: dict = {"move": _best_move(snake)}
    teacher: dict = {"move": _move_teacher(snake)}
    for direction in DIRECTIONS:
        qid = f"safe-{direction}"
        safe = direction_safe(snake, direction)
        questions.append(
            make_boolean_question(
                qid,
                f"Is moving {direction} safe?",
                {"true": f"moving {direction} is safe", "false": f"moving {direction} is blocked"},
            )
        )
        gold[qid] = safe
        teacher[qid] = {"true": 0.9 if safe else 0.1, "false": 0.1 if safe else 0.9}
    level = _distance_level(snake)
    questions.append(
        make_score_question("distance", "How far is the food from the head?", list(DISTANCE_LEVELS))
    )
    gold["distance"] = level
    level_target = [0.04] * len(DISTANCE_LEVELS)
    level_target[level] = 0.84
    teacher["distance"] = {str(index): value for index, value in enumerate(level_target)}
    for question in questions:
        question.family = question.kind
    return Request(
        id=f"snake-{snake.size}-{snake.head[0]}-{snake.head[1]}-{step}",
        state=state_object(snake, step, eaten),
        questions=questions,
        gold=gold,
        teacher=teacher,
    )


def _play_walk(rng: random.Random, snake: Snake, steps: int) -> Snake:
    """Play the teacher policy for a few steps, growing the body on food."""
    for _ in range(steps):
        moves = legal_moves(snake)
        if not moves:
            break
        if rng.random() < 0.75:
            move = _best_move(snake)
            if move not in moves:
                move = rng.choice(moves)
        else:
            move = rng.choice(moves)
        snake, ate = advance(snake, move)
        if ate:
            grown = _spawn_food(rng, snake)
            if grown is None:
                break
            snake = grown
    return snake


def generate_request(index: int, seed: int = 17, split: str = "train", size: int = 6, length: int = 3) -> Request:
    offsets = {"train": 0, "dev": 1_000_000, "test": 2_000_000}
    rng = random.Random((seed * 6_151) ^ (index + offsets.get(split, 3_000_000)))
    snake = generate_snake(rng, size=size, length=length)
    snake = _play_walk(rng, snake, steps=rng.randint(0, 20))
    return state_request(snake, step=rng.randint(0, 24), eaten=snake.length - length)


def generate_split(count: int, seed: int = 17, split: str = "train", size: int = 6, length: int = 3) -> list[Request]:
    return [generate_request(index, seed=seed, split=split, size=size, length=length) for index in range(count)]


def _spawn_food(rng: random.Random, snake: Snake) -> Snake | None:
    occupied = set(snake.body)
    free = [
        (row, column)
        for row in range(snake.size)
        for column in range(snake.size)
        if (row, column) not in occupied
    ]
    if not free:
        return None
    return replace(snake, food=rng.choice(free))


def rollout(
    model,
    size: int = 6,
    seed: int = 23,
    length: int = 3,
    max_steps: int = 120,
    target_food: int = 5,
    device: str = "auto",
) -> dict:
    rng = random.Random(seed)
    snake = generate_snake(rng, size=size, length=length)
    initial = snake
    eaten = 0
    frames = []
    success = False
    for step in range(max_steps):
        request = state_request(snake, step, eaten)
        records = {record["qid"]: record for record in model.score([request], ScoreOptions(device=device))}
        choice = records["move"]
        combined = {}
        for direction in DIRECTIONS:
            safety = records[f"safe-{direction}"]["probabilities"][1]
            combined[direction] = choice["probabilities"][choice["candidate_ids"].index(direction)] * safety
        legal = legal_moves(snake)
        if not legal:
            break
        best_legal = max(legal, key=lambda direction: combined[direction])
        greedy_choice = max(
            DIRECTIONS,
            key=lambda direction: choice["probabilities"][choice["candidate_ids"].index(direction)],
        )
        forced = greedy_choice not in legal
        move = best_legal
        before = snake
        snake, ate = advance(before, move)
        frames.append(_frame(step, before, move, forced, eaten, ate, choice, records, combined))
        if ate:
            eaten += 1
            grown = _spawn_food(rng, snake)
            if grown is None:
                success = eaten >= target_food
                break
            snake = grown
        if eaten >= target_food:
            success = True
            break
    return {
        "size": size,
        "start": [[row, column] for row, column in initial.body],
        "start_food": [initial.food[0], initial.food[1]],
        "success": success,
        "steps": len(frames),
        "target_food": target_food,
        "eaten": eaten,
        "final": [[row, column] for row, column in snake.body],
        "final_food": [snake.food[0], snake.food[1]],
        "frames": frames,
    }


def _frame(
    step: int,
    snake: Snake,
    move: str,
    forced: bool,
    eaten: int,
    ate: bool,
    choice: dict,
    records: dict,
    combined: dict,
) -> dict:
    return {
        "step": step,
        "snake": [[row, column] for row, column in snake.body],
        "food": [snake.food[0], snake.food[1]],
        "heading": snake.heading,
        "move": move,
        "forced": forced,
        "eaten": eaten,
        "ate": ate,
        "distance": manhattan(snake.head, snake.food),
        "move_probabilities": {
            direction: round(choice["probabilities"][choice["candidate_ids"].index(direction)], 6)
            for direction in DIRECTIONS
        },
        "safety": {direction: round(records[f"safe-{direction}"]["probabilities"][1], 6) for direction in DIRECTIONS},
        "combined": {direction: round(combined[direction], 6) for direction in DIRECTIONS},
        "distance_expected": records["distance"]["expected"],
    }


def rollout_bundle(
    model,
    count: int = 6,
    seed: int = 7,
    size: int = 6,
    length: int = 3,
    max_steps: int = 120,
    target_food: int = 5,
    device: str = "auto",
) -> dict:
    runs = []
    for index in range(count):
        result = rollout(
            model,
            size=size,
            seed=seed * 1_000 + index,
            length=length,
            max_steps=max_steps,
            target_food=target_food,
            device=device,
        )
        result["id"] = f"snake-{index + 1}"
        runs.append(result)
    return {
        "runs": runs,
        "summary": {
            "games": len(runs),
            "successes": sum(run["success"] for run in runs),
            "total_steps": sum(run["steps"] for run in runs),
            "food_eaten": sum(run["eaten"] for run in runs),
            "forced_moves": sum(frame["forced"] for run in runs for frame in run["frames"]),
        },
    }
