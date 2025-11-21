"""data utility functions for casmsocial"""

from dataclasses import dataclass, fields
from typing import Optional, Union


# extract_dataclass_fields
def extract_dataclass_attribute_names(dataclass: type[dataclass]) -> list[str]:
    """Extract dataclass fields as a dictionary."""
    # Get a list of field objects
    field_objects = fields(dataclass)

    # Extract attribute names from field objects
    attribute_names = [field.name for field in field_objects]
    return attribute_names


def get_attribute_names_from_data(data: dataclass) -> list[str]:
    """Get a list of attribute names from a dataclass instance."""
    return [f.name for f in fields(data)]


# create_dataclass_record_from_dict
def create_dataclass_record_from_dict(dataclass: type[dataclass], initDict: dict) -> dataclass:
    """Create a dataclass record from a dictionary.

    Steps:

    1. Get a list of dataclass attributes name
    2. Remove keys from initDict that are not in attribute_names
    3. Combine inputDict and initDict
    4. Create a dataclass record from the combined dictionary

    Arguments:
        dataclass: A dataclass type.
        inputDict: A dictionary of input attribute values.
        initDict: A dictionary of initialization attribute values.

    Returns:
        A dictionary of attribute values.
    """
    # Get a list of dataclass attributes name
    attribute_names = extract_dataclass_attribute_names(dataclass)

    # remove keys from initDict that are not in attribute_names
    unwanted_keys = set(initDict.keys()) - set(attribute_names)
    for unwanted_key in unwanted_keys:
        del initDict[unwanted_key]

    parameters = initDict

    return dataclass(**parameters)


def convert_to_int(x: Union[int, str, None]) -> Optional[int]:
    """Convert a string to an integer if possible.

    Arguments:
        x: Optional[int, str]: The value to convert.
    Returns:
        int: The integer value of the string if possible, otherwise None
    """
    if not x:
        return None
    try:
        return int(x)
    except ValueError:
        return None
