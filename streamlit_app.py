from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Protocol, runtime_checkable, Self

@runtime_checkable
class Showable(Protocol):
    def show(self) -> str:
        ...

ObsT = TypeVar('ObsT', bound=Showable)
OutT = TypeVar('OutT', bound=Showable)

@runtime_checkable
class Actionable(Showable, Protocol, Generic[ObsT]):
    @classmethod
    def all_possibilities(cls, obs: ObsT) -> frozenset[Self]:
        ...

ActT = TypeVar('ActT', bound=Actionable)

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

# Streamlit UI
import streamlit as st

st.title("🎈 Making a change")
st.write("Why don't I see the change?")
