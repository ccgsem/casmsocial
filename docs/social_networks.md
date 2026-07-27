# Social-network input contract

`social_networks.table` is an optional input of potential relationships. It is
not a contact-event schedule and therefore has no `hour` or `place_id` column.

| Column | Required | Meaning |
| --- | --- | --- |
| `person_id_a` | yes | Lower endpoint of a canonical undirected tie. |
| `person_id_b` | yes | Higher endpoint of the same tie. |
| `network_kind` | yes | Relationship source or type, such as `household`, `school`, `work`, or `daycare`. |
| `tie_strength` | no | Non-negative model input used by a behavior layer when selecting among ties. |

Rows must have non-null endpoints with `person_id_a < person_id_b`. Each
`(person_id_a, person_id_b, network_kind)` must be unique. Additional
provenance columns are allowed.

CASMSocial expands each row into an in-memory adjacency map at startup. This
map is a set of possible social channels only. A future interaction generator
applies separate mechanisms:

- planned co-location and schedule overlap for physical encounters;
- availability and tie strength for messages or other remote contacts.

This separation prevents a relationship edge from being treated as a guaranteed
in-person encounter at an arbitrary fixed hour.

The initial implementation is in `casmsocial.social_interactions`:
`generate_in_person_events` requires an overlapping `PresenceInterval` at the
same place, while `generate_remote_message_opportunities` requires only that
both people are available. The behavior layer remains responsible for choosing
whether an eligible opportunity becomes an actual message or contact.

CASMSocial can optionally turn eligible remote opportunities into ordinary
two-way `CHECK_IN` messages with:

```yaml
social_networks.remote_messages.enabled: true
social_networks.remote_messages.interval_minutes: 60
```

The sender direction alternates by simulation tick and each available local
sender selects at most one tied peer, preferring the strongest tie. This is a
deterministic baseline policy, not a calibrated communication model.

For calibration and validation, enable the privacy-safe aggregate observer:

```yaml
observers.social_interaction_log.enabled: true
observers.social_interaction_log_file: 'social_interactions.parquet'
```

It writes only `run_id`, random seed, tick, rank, channel, network kind, and
event count. It never writes person identifiers, edge pairs, locations, or
message content.
