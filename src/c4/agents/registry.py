"""Build agents from compact spec strings, e.g. ``minimax:d=4`` or ``az:ckpt=models/x.pt,sims=200``.

Spec grammar: ``name[:key=value[,key=value...]]``. Values are parsed as int, float or bool where
possible. A ``seed`` key is accepted by every agent.
"""

from __future__ import annotations

from typing import Any

from c4.agents.base import Agent

ALIASES = {
    "minimax": {"d": "depth"},
    "mcts": {"it": "iterations"},
    "az": {"ckpt": "checkpoint", "t": "temperature"},
    "ppo": {"ckpt": "checkpoint"},
}


def _parse_value(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    name, _, rest = spec.partition(":")
    kwargs: dict[str, Any] = {}
    if rest:
        for part in rest.split(","):
            k, _, v = part.partition("=")
            k = ALIASES.get(name, {}).get(k.strip(), k.strip())
            kwargs[k] = _parse_value(v.strip())
    return name.strip(), kwargs


def make_agent(spec: str, seed: int | None = None) -> Agent:
    name, kw = parse_spec(spec)
    if seed is not None:
        kw.setdefault("seed", seed)
    if name == "random":
        from c4.agents.random_agent import RandomAgent

        agent: Any = RandomAgent(**kw)
    elif name == "minimax":
        from c4.agents.minimax import MinimaxAgent

        agent = MinimaxAgent(**kw)
    elif name == "mcts":
        from c4.agents.mcts_uct import MCTSAgent

        agent = MCTSAgent(**kw)
    elif name == "solver":
        from c4.agents.solver_agent import SolverAgent

        agent = SolverAgent(**kw)
    elif name == "az":
        from c4.agents.az_agent import AZAgent

        agent = AZAgent(**kw)
    elif name == "ppo":
        from c4.baselines.ppo.agent import PPOAgent

        agent = PPOAgent(**kw)
    else:
        raise ValueError(f"unknown agent '{name}' in spec '{spec}'")
    agent.name = spec
    return agent
