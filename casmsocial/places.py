""" Places Class """
from dataclasses  import dataclass
from typing import (
    Type,
    List,
    NamedTuple
)

from casmsocial.place import Place


# NamedTuple for PlacesConfig
PlacesConfig = NamedTuple(
    'PlacesConfig',
    [
        ('name', str),
        ('type', Type[Place]),
        ('dataType', Type[dataclass]),
        ('personPlaceField', str)
    ]
)


class Places:
    """Configurations for places."""

    # List of PlacesConfigs
    __configs: List[PlacesConfig] = []

    @classmethod
    def register_place_config(cls, config: PlacesConfig):
        """Add a PlacesConfig to the list of configs."""
        cls.__configs.append(config)
    
    @classmethod
    def get_place_configs(cls) -> List[PlacesConfig]:
        """Get the list of PlacesConfigs."""
        return cls.__configs

    @classmethod
    def get_place_config(cls, idx: int) -> PlacesConfig:
        """Get a PlacesConfig from the list of configs."""
        return cls.__configs[idx]

    @classmethod
    def get_place_config_idx(cls, name: str) -> int:
        """Get the index of a PlacesConfig in the list of configs."""
        for idx, config in enumerate(cls.__configs):
            if config.name == name:
                return idx
        return -1

    @classmethod
    def get_num_configs(cls) -> int:
        """Get the number of PlacesConfigs in the list of configs."""
        return len(cls.__configs)
