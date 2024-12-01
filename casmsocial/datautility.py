""" data utility functions for casmsocial """
from dataclasses  import dataclass, fields
from typing import Type, List, Dict, NamedTuple


# extract_dataclass_fields
def extract_dataclass_attribute_names(
        dataclass: Type[dataclass]) -> List[str]:
    """Extract dataclass fields as a dictionary."""
     # Get a list of field objects
    field_objects = fields(dataclass)

    # Extract attribute names from field objects
    attribute_names = [field.name for field in field_objects]
    return attribute_names


# extract_dataclass_fields
def create_dataclass_record_from_dicts(
        dataclass: Type[dataclass],
        initDict: Dict,
        inputDict: Dict,
        ) -> dataclass:
    """Create a dataclass record from dictionaries.

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
    
    parameters = inputDict | initDict
    
    return dataclass(**parameters)