from casmsocial.datasets.colorado_front_range import (
    cbsa_by_county_geoid,
    list_profiles,
    load_boundary,
    load_profile,
)


def test_bundled_boundary_covers_all_profile_cbsas_and_counties():
    boundary = load_boundary()
    mapping = cbsa_by_county_geoid()

    assert boundary.boundary_id == "co-front-range-six-metros-cbsa-2023"
    assert len(boundary.metropolitan_statistical_areas) == 6
    assert len(mapping) == 16
    assert mapping["08013"] == "14540"
    assert mapping["08031"] == "19740"
    assert mapping["08041"] == "17820"
    assert mapping["08069"] == "22660"
    assert mapping["08101"] == "39380"
    assert mapping["08123"] == "24540"

    bundled_cbsas = {area.cbsa_code for area in boundary.metropolitan_statistical_areas}
    for profile_name in list_profiles():
        assert set(load_profile(profile_name).geography.home_cbsa_codes) <= bundled_cbsas
