"""Grid-world decision task and agent rollout.

A maze state is structured JSON: grid size, agent and goal coordinates, step
index, and the wall list. Each step asks nine independent questions in one
forward pass:

- one choice over the four move directions (teacher: closer moves score
  higher, walls score near zero);
- four booleans, one per direction ("is moving up safe?");
- one score over distance buckets ("how far is the goal?").

A rollout controller masks the choice distribution with the safety booleans
and follows the surviving argmax, so an untrusted move can never be taken
without being reported as forced.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

from .model import ScoreOptions
from .types import Request, make_boolean_question, make_choice_question, make_score_question

DIRECTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
DISTANCE_LEVELS = ("adjacent", "close", "medium", "far")
DISTANCE_BUCKETS = ((0, 1), (2, 3), (4, 6), (7, 10**9))


@dataclass(frozen=True)
class Maze:
    size: int
    start: tuple[int, int]
    goal: tuple[int, int]
    walls: frozenset[tuple[int, int]]

    @property
    def wall_list(self) -> list[tuple[int, int]]:
        return sorted(self.walls)


def in_bounds(maze: Maze, cell: tuple[int, int]) -> bool:
    row, column = cell
    return 0 <= row < maze.size and 0 <= column < maze.size


def is_wall(maze: Maze, cell: tuple[int, int]) -> bool:
    return cell in maze.walls


def step_cell(maze: Maze, cell: tuple[int, int], direction: str) -> tuple[int, int]:
    delta = DELTAS[direction]
    return cell[0] + delta[0], cell[1] + delta[1]


def distance(maze: Maze, cell: tuple[int, int]) -> int:
    return abs(cell[0] - maze.goal[0]) + abs(cell[1] - maze.goal[1])


def direction_safe(maze: Maze, cell: tuple[int, int], direction: str) -> bool:
    target = step_cell(maze, cell, direction)
    return in_bounds(maze, target) and not is_wall(maze, target)


def legal_moves(maze: Maze, cell: tuple[int, int]) -> list[str]:
    return [direction for direction in DIRECTIONS if direction_safe(maze, cell, direction)]


def bfs_distance(maze: Maze, cell: tuple[int, int]) -> int | None:
    if is_wall(maze, cell) or not in_bounds(maze, cell):
        return None
    queue = deque([(cell, 0)])
    seen = {cell}
    while queue:
        current, depth = queue.popleft()
        if current == maze.goal:
            return depth
        for direction in DIRECTIONS:
            target = step_cell(maze, current, direction)
            if target in seen or not direction_safe(maze, current, direction):
                continue
            seen.add(target)
            queue.append((target, depth + 1))
    return None


def generate_maze(rng: random.Random, size: int = 6, density: float = 0.16) -> Maze:
    start, goal = (0, 0), (size - 1, size - 1)
    while True:
        walls = {
            (row, column)
            for row in range(size)
            for column in range(size)
            if (row, column) not in {start, goal} and rng.random() < density
        }
        maze = Maze(size=size, start=start, goal=goal, walls=frozenset(walls))
        if bfs_distance(maze, start) is not None:
            return maze


def state_object(maze: Maze, agent: tuple[int, int], step: int) -> dict:
    return {
        "size": maze.size,
        "agent": [agent[0], agent[1]],
        "goal": [maze.goal[0], maze.goal[1]],
        "step": step,
        "walls": [[row, column] for row, column in maze.wall_list],
    }


def _distance_level(maze: Maze, agent: tuple[int, int]) -> int:
    current = distance(maze, agent)
    for index, (low, high) in enumerate(DISTANCE_BUCKETS):
        if low <= current <= high:
            return index
    return len(DISTANCE_LEVELS) - 1


def _move_teacher(maze: Maze, agent: tuple[int, int], temperature: float = 3.0) -> dict[str, float]:
    distances = {}
    for direction in DIRECTIONS:
        if direction_safe(maze, agent, direction):
            distances[direction] = distance(maze, step_cell(maze, agent, direction))
    best = min(distances.values())
    weights = {direction: math.exp(-temperature * (value - best)) for direction, value in distances.items()}
    totals = sum(weights.values())
    return {direction: (weights.get(direction, 1e-6) / totals) for direction in DIRECTIONS}


def state_request(maze: Maze, agent: tuple[int, int], step: int) -> Request:
    questions = [
        make_choice_question(
            "move",
            "Which move should the agent make to reach the goal?",
            {direction: f"move {direction}" for direction in DIRECTIONS},
        ),
    ]
    gold: dict = {"move": _best_move(maze, agent)}
    teacher: dict = {"move": _move_teacher(maze, agent)}
    for direction in DIRECTIONS:
        qid = f"safe-{direction}"
        safe = direction_safe(maze, agent, direction)
        questions.append(
            make_boolean_question(
                qid,
                f"Is moving {direction} safe?",
                {"true": f"moving {direction} is safe", "false": f"moving {direction} is blocked"},
            )
        )
        gold[qid] = safe
        teacher[qid] = {"true": 0.9 if safe else 0.1, "false": 0.1 if safe else 0.9}
    level = _distance_level(maze, agent)
    questions.append(
        make_score_question("distance", "How far is the goal from the agent?", list(DISTANCE_LEVELS))
    )
    gold["distance"] = level
    level_target = [0.04] * len(DISTANCE_LEVELS)
    level_target[level] = 0.84
    teacher["distance"] = {str(index): value for index, value in enumerate(level_target)}
    for question in questions:
        question.family = question.kind
    return Request(
        id=f"maze-{maze.size}-{agent[0]}-{agent[1]}-{step}",
        state=state_object(maze, agent, step),
        questions=questions,
        gold=gold,
        teacher=teacher,
    )


def _best_move(maze: Maze, agent: tuple[int, int]) -> str:
    moves = legal_moves(maze, agent)
    if not moves:
        return DIRECTIONS[0]
    best = min(distance(maze, step_cell(maze, agent, direction)) for direction in moves)
    tied = [direction for direction in moves if distance(maze, step_cell(maze, agent, direction)) == best]
    if len(tied) == 1:
        return tied[0]
    row_gap = maze.goal[0] - agent[0]
    column_gap = maze.goal[1] - agent[1]
    if abs(row_gap) > abs(column_gap):
        preferred = "down" if row_gap > 0 else "up"
    elif abs(column_gap) > abs(row_gap):
        preferred = "right" if column_gap > 0 else "left"
    else:
        preferred = None
    if preferred in tied:
        return preferred
    return tied[0]


def _random_reachable(maze: Maze, rng: random.Random, max_step: int) -> tuple[int, int]:
    cell = maze.start
    for _ in range(rng.randint(0, max_step)):
        moves = legal_moves(maze, cell)
        if not moves:
            break
        cell = step_cell(maze, cell, rng.choice(moves))
    return cell


def generate_request(index: int, seed: int = 17, split: str = "train", size: int = 6) -> Request:
    offsets = {"train": 0, "dev": 1_000_000, "test": 2_000_000}
    rng = random.Random((seed * 7_919) ^ (index + offsets.get(split, 3_000_000)))
    maze = generate_maze(rng, size=size)
    agent = _random_reachable(maze, rng, max_step=24)
    return state_request(maze, agent, rng.randint(0, 24))


def generate_split(count: int, seed: int = 17, split: str = "train", size: int = 6) -> list[Request]:
    return [generate_request(index, seed=seed, split=split, size=size) for index in range(count)]


def rollout(model, maze: Maze, max_steps: int = 64, device: str = "auto") -> dict:
    agent = maze.start
    frames = []
    success = False
    for step in range(max_steps):
        request = state_request(maze, agent, step)
        records = {record["qid"]: record for record in model.score([request], ScoreOptions(device=device))}
        choice = records["move"]
        combined = {}
        for direction in DIRECTIONS:
            safety = records[f"safe-{direction}"]["probabilities"][1]
            combined[direction] = choice["probabilities"][choice["candidate_ids"].index(direction)] * safety
        legal = legal_moves(maze, agent)
        if not legal:
            break
        best_legal = max(legal, key=lambda direction: combined[direction])
        greedy_choice = max(DIRECTIONS, key=lambda direction: choice["probabilities"][choice["candidate_ids"].index(direction)])
        forced = greedy_choice not in legal
        move = best_legal
        target = step_cell(maze, agent, move)
        frames.append(
            {
                "step": step,
                "agent": [agent[0], agent[1]],
                "move": move,
                "forced": forced,
                "distance": distance(maze, agent),
                "move_probabilities": {direction: round(choice["probabilities"][choice["candidate_ids"].index(direction)], 6) for direction in DIRECTIONS},
                "safety": {direction: round(records[f"safe-{direction}"]["probabilities"][1], 6) for direction in DIRECTIONS},
                "combined": {direction: round(combined[direction], 6) for direction in DIRECTIONS},
                "distance_expected": records["distance"]["expected"],
            }
        )
        agent = target
        if agent == maze.goal:
            success = True
            break
    optimal = bfs_distance(maze, maze.start)
    return {
        "size": maze.size,
        "walls": [[row, column] for row, column in maze.wall_list],
        "start": list(maze.start),
        "goal": list(maze.goal),
        "success": success,
        "steps": len(frames),
        "optimal_steps": optimal,
        "final": [agent[0], agent[1]],
        "frames": frames,
    }


def rollout_bundle(model, count: int = 6, seed: int = 23, size: int = 6, device: str = "auto") -> dict:
    rng = random.Random(seed)
    runs = []
    for index in range(count):
        maze = generate_maze(rng, size=size)
        result = rollout(model, maze, device=device)
        result["id"] = f"maze-{index}"
        runs.append(result)
    return {
        "runs": runs,
        "summary": {
            "mazes": len(runs),
            "successes": sum(run["success"] for run in runs),
            "total_steps": sum(run["steps"] for run in runs),
            "forced_moves": sum(frame["forced"] for run in runs for frame in run["frames"]),
        },
    }
