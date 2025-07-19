import streamlit as st

class World(ABC):

    @property
    @abstractmethod
    def observation_distribution(self) -> dict[str, float]
        raise NotImplemented

    @property
    @abstractmethod
    def possible_actions(self) -> frozenset[str]
        raise NotImplemented

    @property
    @abstractmethod
    def marginal_outcome_distribution(self, observation: str, action: str) -> dict[str, float]
        raise NotImplemented
        

st.title("🎈 Making a change")
st.write(
    "Why don't I see the change?"
)
