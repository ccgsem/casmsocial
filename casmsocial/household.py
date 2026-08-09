from dataclasses import dataclass

import repast4py.core as core

from casmsocial.data_utilities import create_dataclass_record_from_dict


@dataclass(slots=True)
class HouseholdData:
    """Household data class"""

    household_id: int = 0
    place_id: int = 0
    household_size: int = 0
    household_income: float = 0.0
    household_type: str = ""
    household_race: str = ""
    household_age: int = 0


class Household(core.Agent):
    """Household class"""

    TYPE = 2
    __household_data_class: type[dataclass] = HouseholdData

    @classmethod
    def getHouseholdDataClass(cls) -> type[dataclass]:
        """Get the household data class."""
        return cls.__household_data_class

    @classmethod
    def setHouseholdDataClass(cls, household_data_class: type[dataclass]) -> None:
        """Set the household data class."""
        cls.__household_data_class = household_data_class

    def __init__(self, initDict: dict, householdDataClass: type[dataclass]):
        """Initialize the household."""
        init_dict = self._normalize_init_dict(initDict)
        local_id = init_dict["household_id"]
        rank = init_dict.get("rank", 0)

        super().__init__(local_id, Household.TYPE, rank)

        if "rank" not in init_dict:
            init_dict["rank"] = 0
        self.rank = init_dict["rank"]
        self.place = None
        self.members = []

        self.household_data = create_dataclass_record_from_dict(householdDataClass, init_dict)

    @staticmethod
    def _normalize_init_dict(init_dict: dict) -> dict:
        """Normalize common source-table aliases to HouseholdData fields."""
        normalized = dict(init_dict)
        if "household_id" not in normalized:
            normalized["household_id"] = normalized.get("sp_id", normalized.get("hh_id"))
        if normalized["household_id"] is None:
            raise ValueError("Household rows require household_id, sp_id, or hh_id")

        if "place_id" not in normalized or normalized["place_id"] is None:
            normalized["place_id"] = normalized.get("sp_home_id", normalized.get("sp_id"))
        if normalized["place_id"] is None:
            normalized["place_id"] = normalized["household_id"]

        aliases = {
            "household_size": ("hh_size", "size"),
            "household_income": ("hh_income", "income"),
            "household_type": ("hh_type", "type"),
            "household_race": ("hh_race", "race"),
            "household_age": ("hh_age", "age"),
        }
        for field_name, alias_names in aliases.items():
            if field_name in normalized:
                continue
            for alias_name in alias_names:
                if alias_name in normalized:
                    normalized[field_name] = normalized[alias_name]
                    break
        return normalized

    @property
    def household_id(self) -> int:
        """Household identifier."""
        return self.household_data.household_id

    @property
    def place_id(self) -> int:
        """Physical place associated with this household."""
        return self.household_data.place_id

    def setPlace(self, place: core.Agent) -> None:
        """Link this household to its physical place."""
        self.place = place

    def getPlace(self) -> core.Agent | None:
        """Get the physical place linked to this household."""
        return self.place

    def addMember(self, person: core.Agent) -> None:
        """Link a person agent to this household."""
        if person not in self.members:
            self.members.append(person)

    def getHouseholdData(self) -> dataclass:
        """Get the household data."""
        return self.household_data

    def setHouseholdData(self, household_data: dataclass) -> None:
        """Set the household data."""
        self.household_data = household_data

    def getHouseholdMembers(self) -> list[core.Agent]:
        """Get the household members."""
        return list(self.members)
