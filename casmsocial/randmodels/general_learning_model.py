import random
from typing import ClassVar

import networkx as nx
import numpy as np
from repast4py import core, schedule, space


class SocialLearningAgent(core.Agent):
    # class attributes
    behaviors: ClassVar[list[str]] = ["A", "Ā"]  # Possible behaviors

    @classmethod
    def get_behaviors(cls):
        """Returns the list of possible behaviors."""
        return cls.behaviors

    @classmethod
    def set_behaviors(cls, behaviors):
        """Sets the list of possible behaviors."""
        cls.behaviors = behaviors

    def __init__(self, agent_id, rank, initial_behavior, network):
        super().__init__(agent_id, rank)
        self.behavior = initial_behavior  # A or Ā
        self.propensity = np.random.uniform(-1, 1)  # Initial behavioral propensity
        self.memory = 0.85  # Memory parameter (retains past experiences)
        self.network = network  # Social connections
        self.experience_weights = {  # Outcome-dependent experience weights
            "positive": 0.5,
            "negative": -1.0,
            "rare_bad": -3.0,  # Rare catastrophic event
        }

    def __repr__(self):
        """String representation of the agent."""
        return f"Agent(id={self.id}, rank={self.rank}, behavior={self.behavior})"

    def __str__(self):
        """String representation of the agent."""
        return f"Agent(id={self.id}, rank={self.rank}, behavior={self.behavior})"

    def step(self, context):
        """
        - Update behavior based on past experiences and social influence.
        - Transmit behavior and/or outcome information to neighbors.
        """
        self.adapt_behavior(context)
        self.transmit_information(context)

    def adapt_behavior(self, context):
        """Adjusts behavior based on individual and social experiences."""
        neighbors = self.network.neighbors(self.id)
        social_experiences = [context.agent_by_id(n).behavior for n in neighbors]

        # Calculate new propensity based on personal memory and social network influence
        personal_experience = (
            self.experience_weights["positive"] if self.behavior == "A" else self.experience_weights["negative"]
        )
        social_influence = sum(
            [
                self.experience_weights["positive"] if b == "A" else self.experience_weights["negative"]
                for b in social_experiences
            ]
        ) / len(neighbors)

        # Update propensity using memory decay
        self.propensity = self.memory * self.propensity + personal_experience + social_influence

        # Convert propensity into behavior probability
        behavior_probability = 1 / (1 + np.exp(-self.propensity))
        self.behavior = "A" if random.random() < behavior_probability else "Ā"  # Non-cryptographic use for simulation

    def transmit_information(self, context):
        """Agents share their experiences with neighbors in their network."""
        for neighbor_id in self.network.neighbors(self.id):
            neighbor = context.agent_by_id(neighbor_id)
            if random.random() < 0.7:  # 70% chance of sharing information (non-cryptographic use)
                if random.random() < 0.5:  # Non-cryptographic randomness for simulation
                    neighbor.receive_information(self.behavior)  # Behavior-only
                else:
                    neighbor.receive_information(
                        ("behavior", self.behavior, "outcome", "positive" if self.behavior == "A" else "negative")
                    )

    def receive_information(self, info):
        """Processes received social information to update beliefs."""
        if isinstance(info, str):  # Behavior-only transmission
            self.propensity += (
                self.experience_weights["positive"] if info == "A" else self.experience_weights["negative"]
            )
        elif isinstance(info, tuple):  # Behavior-outcome pair
            outcome_weight = self.experience_weights[info[3]]
            self.propensity += outcome_weight


class SocialLearningModel:
    def __init__(self, num_agents, network_type="small-world"):
        self.schedule = schedule.Schedule()
        self.context = core.Context()
        self.space = space.SharedGrid("grid", 50, 50, False)
        self.context.add_projection(self.space)

        # Create social network (Watts-Strogatz small-world network)
        self.network = (
            nx.watts_strogatz_graph(num_agents, k=6, p=0.1)
            if network_type == "small-world"
            else nx.erdos_renyi_graph(num_agents, p=0.1)
        )

        # Initialize agents
        for i in range(num_agents):
            initial_behavior = "A" if random.random() > 0.5 else "Ā"  # Non-cryptographic randomness for simulation
            agent = SocialLearningAgent(i, 1, initial_behavior, self.network)
            self.context.add(agent)
            self.schedule.add(agent)

    def step(self):
        """Runs one iteration of the model."""
        self.schedule.execute()


# Run simulation
model = SocialLearningModel(100)
for _ in range(100):
    model.step()
