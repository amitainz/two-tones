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
class Actionable(Showable, Generic[ObsT]):
    @classmethod
    def all_possibilities(cls, obs: ObsT) -> frozenset[Self]:
        ...


ActT = TypeVar('ActT', bound=Actionable)


class Outcome(Showable):
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


# Concrete implementations
class WeatherObservation:
    def __init__(self, description: str):
        self.description = description

    def show(self) -> str:
        return self.description

    def __hash__(self):
        return hash(self.description)

    def __eq__(self, other):
        return isinstance(other, WeatherObservation) and self.description == other.description




class UmbrellaAction(Actionable[WeatherObservation]):
    def __init__(self, description: str):
        self.description = description

    def show(self) -> str:
        return self.description

    def __hash__(self):
        return hash(self.description)

    def __eq__(self, other):
        return isinstance(other, UmbrellaAction) and self.description == other.description

    @classmethod
    def all_possibilities(cls, obs: WeatherObservation) -> frozenset[Self]:
        return frozenset({cls("Take Umbrella"), cls("Don't Take Umbrella")})

TAKE = UmbrellaAction("Take Umbrella")
DONT_TAKE = UmbrellaAction("Dont Take Umbrella")


class SimpleOutcome(Outcome):
    def __init__(self, description: str, reward: float):
        self.description = description
        self._reward = reward

    def show(self) -> str:
        return self.description

    @property
    def reward(self) -> float:
        return self._reward

    def __hash__(self):
        return hash(self.description)

    def __eq__(self, other):
        return isinstance(other, SimpleOutcome) and self.description == other.description


class SimpleWorld(World[WeatherObservation, UmbrellaAction, SimpleOutcome]):
    def __init__(self, obs_distribution: dict[WeatherObservation, float],
                 outcomes: dict[tuple[WeatherObservation, UmbrellaAction], dict[SimpleOutcome, float]]):
        self._obs_distribution = obs_distribution
        self._outcomes = outcomes

    @property
    def observation_distribution(self) -> dict[WeatherObservation, float]:
        return self._obs_distribution

    def marginal_outcome_distribution(
        self,
        observation: WeatherObservation,
        action: UmbrellaAction
    ) -> dict[SimpleOutcome, float]:
        return self._outcomes.get((observation, action), {})


# Define worlds
CLOUDY = WeatherObservation("Cloudy")
CLEAR = WeatherObservation("Clear")

DRY_BY_UMBRELLA = SimpleOutcome("It rained but I stayed dry", 50)
UNNECESSARY_BURDEN = SimpleOutcome("No rain, I took an unnecessary load", 70)
SOAKED = SimpleOutcome("I got soaked", -100)
LIGHT_AS_A_FEATHER = SimpleOutcome("No rain, light as a feather.", 100)

world1 = SimpleWorld(
    obs_distribution={CLOUDY: 0.6, CLEAR: 0.4},
    outcomes={
        (CLOUDY, TAKE): {
            DRY_BY_UMBRELLA: 0.3,
            UNNECESSARY_BURDEN: 0.7
        },
        (CLOUDY, DONT_TAKE): {
            SOAKED: 0.3,
            LIGHT_AS_A_FEATHER: 0.7,
        },
        (CLEAR, TAKE): {
            DRY_BY_UMBRELLA: 0.1,
            UNNECESSARY_BURDEN: 0.9
        },
        (CLEAR, DONT_TAKE): {
            SOAKED: 0.1,
            LIGHT_AS_A_FEATHER: 0.9,
        }
    },
)

world2 = SimpleWorld(
    obs_distribution={CLOUDY: 0.5, CLEAR: 0.5},
    outcomes=world1._outcomes
)

world_distribution: list[tuple[World, float]] = [
    (world1, 0.5),
    (world2, 0.5)
]


def sample_world() -> None:
    worlds, weights = zip(*world_distribution)
    st.session_state.current_world = random.choices(worlds, weights=weights, k=1)[0]


def sample_observation() -> None:
    st.session_state.current_observation = random.choices(
        list(st.session_state.current_world.observation_distribution.keys()),
        weights=st.session_state.current_world.observation_distribution.values(),
        k=1
    )[0]


def sample_outcome() -> OutT:
    outcome_distribution = st.session_state.current_world.marginal_outcome_distribution(
        st.session_state.current_observation,
        action_choice
    )
    return random.choices(
        list(outcome_distribution.keys()),
        weights=outcome_distribution.values(),
        k=1
    )[0]


# Streamlit UI
import streamlit as st

st.title("🌍 Interactive World Simulation")

# Initialize session state
if 'total_reward' not in st.session_state:
    st.session_state.total_reward = 0.0
    sample_world()
    sample_observation()
    st.session_state.awaiting_play_again = False

# Reward display box
st.sidebar.header("Game Stats")
st.sidebar.metric("Total Reward", f"{st.session_state.total_reward:.2f}")

# Play Again logic
if st.session_state.awaiting_play_again:
    if st.button("Play Again"):
        # Sample new world and observation
        # st.session_state.current_world = sample_world()
        st.session_state.sample_observation()
        st.session_state.awaiting_play_again = False

    else:
        st.write("Click 'Play Again' to start a new round.")
else:
    # Show current observation
    st.write(f"**Observation:** {st.session_state.current_observation.show()}")

    # Action selection
    possible_actions = UmbrellaAction.all_possibilities(st.session_state.current_observation)
    action_choice = st.selectbox(
        "Choose an action:",
        options=list(possible_actions),
        format_func=lambda a: a.show()
    )

    if st.button("Submit Action"):
        # Compute outcome
        outcome = sample_outcome()

        # Update reward
        st.session_state.total_reward += outcome.reward

        # Display result
        st.success(f"Outcome: {outcome.show()} | Reward: {outcome.reward}")

        # Flag to await next round
        st.session_state.awaiting_play_again = True
