"""data utility functions for casmsocial"""

from dataclasses import dataclass, fields
from functools import cache

import duckdb


class InvalidTableIdentifier(ValueError):
    """Exception raised for invalid table identifiers."""

    def __init__(self, table_name: str) -> None:
        super().__init__(f"Invalid table identifier: {table_name}")


# extract_dataclass_fields
def extract_dataclass_attribute_names(dataclass: type[dataclass]) -> list[str]:
    """Extract dataclass fields as a dictionary."""
    # Get a list of field objects
    field_objects = fields(dataclass)

    # Extract attribute names from field objects
    attribute_names = [field.name for field in field_objects]
    return attribute_names


@cache
def _dataclass_attribute_name_set(dataclass: type[dataclass]) -> frozenset[str]:
    """Return cached dataclass field names for repeated row materialization."""
    return frozenset(field.name for field in fields(dataclass))


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
    # remove keys from initDict that are not in attribute_names
    unwanted_keys = set(initDict.keys()) - _dataclass_attribute_name_set(dataclass)
    for unwanted_key in unwanted_keys:
        del initDict[unwanted_key]

    parameters = initDict

    return dataclass(**parameters)


def convert_to_int(x: int | str | None) -> int | None:
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


def check_if_table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    table_name_parts = table_name.split(".")

    # table_name includes an optional schema prepended to the table name
    # after split should be no more than 2 parts
    if len(table_name_parts) > 2:
        return False

    schema_name = "main"
    if len(table_name_parts) == 2:
        schema_name = table_name_parts[0]
        table_name = table_name_parts[1]

    result = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM duckdb_tables
            WHERE schema_name = ?
            AND table_name = ?
        );
        """,
        [schema_name, table_name],
    ).fetchone()[0]
    return result


def quote_table_identifier(table_name: str) -> str:
    """Safely quote a schema-qualified table identifier.

    Only allows alphanumeric and underscore identifiers, with an optional
    schema prefix. Returns a double-quoted identifier suitable for direct
    SQL interpolation.
    """
    table_name_parts = table_name.split(".")

    if len(table_name_parts) > 2:
        raise InvalidTableIdentifier(table_name)

    def _validate(part: str) -> str:
        if not part:
            raise InvalidTableIdentifier(table_name)
        if not part.replace("_", "").isalnum() or part[0].isdigit():
            raise InvalidTableIdentifier(table_name)
        return f'"{part}"'

    if len(table_name_parts) == 2:
        return f"{_validate(table_name_parts[0])}.{_validate(table_name_parts[1])}"

    return _validate(table_name_parts[0])
