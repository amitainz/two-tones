from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Protocol, runtime_checkable, Self
import random

@runtime_checkable
class Showable(Protocol):
    def show(self) -> str:
        ...

ObsT = TypeVar('ObsT', bound=Showable)
OutT = TypeVar('OutT', bound='Outcome')

@runtime_checkable
class Actionable(Showable, Protocol, Generic[ObsT]):
    @classmethod
    def all_possibilities(cls, obs: ObsT) -> frozenset[Self]:
        ...

ActT = TypeVar('ActT', bound=Actionable)

class Outcome(Showable, Protocol):
    @property
    @abstractmethod
    def reward(self) -> float:
        ...

class World(ABC, Generic[ObsT, ActT, OutT]):
    @property
    @abstractmethod
    def observation_distribution(self) -> dict[ObsT, float]:
        """Probability distribution over observations."""
        raise NotImplementedError

    @abstractmethod
    def marginal_outcome_distribution(
        self,
        observation: ObsT,
        action: ActT,
    ) -> dict[OutT, float]:
        """Conditional outcome distribution given an observation and action."""
        raise NotImplementedError

# Global world distribution: list of tuples (World, probability)
world_distribution: list[tuple[World, float]] = []

def sample_world() -> World:
    worlds, weights = zip(*world_distribution)
    return random.choices(worlds, weights=weights, k=1)[0]

# Streamlit UI
import streamlit as st

st.title("🌍 Interactive World Simulation")

if 'total_reward' not in st.session_state:
    st.session_state.total_reward = 0.0
    st.session_state.current_world = sample_world()
    st.session_state.current_observation = random.choices(
    list(st.session_state.current_world.observation_distribution.keys()),
    weights=st.session_state.current_world.observation_distribution.values(),
    k=1
)[0])
    )

st.write(f"Observation: {st.session_state.current_observation.show()}")

# Assume ActT implements Actionable with all_possibilities
possible_actions = ActT.all_possibilities(st.session_state.current_observation)
action_choice = st.selectbox(
    "Choose an action:",
    options=list(possible_actions),
    format_func=lambda a: a.show()
)

if st.button("Submit Action"):
    outcome_distribution = st.session_state.current_world.marginal_outcome_distribution(
        st.session_state.current_observation,
        action_choice
    )
    outcome = random.choices(
    list(outcome_distribution.keys()),
    weights=outcome_distribution.values(),
    k=1
)[0]

    st.session_state.total_reward += outcome.reward

    st.success(f"Outcome: {outcome.show()} | Reward: {outcome.reward}")
    st.info(f"Total Reward: {st.session_state.total_reward}")

    # Sample next observation
    st.session_state.current_observation = random.choice(
        list(st.session_state.current_world.observation_distribution.keys())
    )

st.write("Total Reward:", st.session_state.total_reward)
