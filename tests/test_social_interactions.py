import pytest

from casmsocial.social_interactions import (
    PresenceInterval,
    SocialTie,
    generate_in_person_events,
    generate_remote_message_opportunities,
)


def test_in_person_event_requires_shared_place_and_time():
    ties = [SocialTie(1, 2, "work", 0.4)]
    presences = [
        PresenceInterval(1, 10, 480, 600),
        PresenceInterval(2, 10, 540, 660),
        PresenceInterval(2, 11, 480, 600),
    ]

    events = generate_in_person_events(ties, presences)

    assert [(event.channel, event.place_id, event.start_minute, event.end_minute) for event in events] == [
        ("in_person", 10, 540, 600)
    ]


def test_social_tie_without_colocation_does_not_create_physical_event():
    ties = [SocialTie(1, 2, "household")]
    presences = [PresenceInterval(1, 10, 480, 600), PresenceInterval(2, 11, 480, 600)]

    assert generate_in_person_events(ties, presences) == []


def test_remote_messages_require_availability_but_not_colocation():
    ties = [SocialTie(1, 2, "school", 0.7)]

    events = generate_remote_message_opportunities(ties, [1, 2], minute=555, duration_minutes=5)

    assert len(events) == 1
    assert events[0].channel == "remote"
    assert events[0].place_id is None
    assert (events[0].start_minute, events[0].end_minute) == (555, 560)


def test_remote_messages_require_both_people_to_be_available():
    assert generate_remote_message_opportunities([SocialTie(1, 2, "work")], [1], minute=100) == []


def test_interaction_inputs_reject_invalid_intervals_and_ties():
    with pytest.raises(ValueError, match="canonical"):
        SocialTie(2, 1, "work")
    with pytest.raises(ValueError, match="after"):
        PresenceInterval(1, 1, 60, 60)
    with pytest.raises(ValueError, match="positive"):
        generate_remote_message_opportunities([], [], minute=60, duration_minutes=0)
