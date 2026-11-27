"""Couple casmsocial plan legs to the vehicle mobsim.

When a person starts a routed leg in the mobsim's mode, the coupler spawns a
vehicle for it and holds the person in transit until the vehicle arrives.
``Person.move_at`` checks the hold, so behavior engines that move people
themselves cannot skip it. A person whose car arrives early still waits for
the next activity's scheduled start, as before. A person whose car arrives late
enters the activity on the first tick after arrival.

The person stays on the rank of its origin place while held: casmsocial keeps
``rank_place_id`` at the origin during a leg, and only the vehicle migrates.
Arrivals are routed back to that rank (``VehicleMobsim.route_arrivals_home``).

Position: each ``advance`` also brings home where every held traveller's car
is, as of the tick time. ``transit_place`` turns that into the nearest place
snapped to the road network, so per-place data such as weather can be looked
up for people on the road.

Timing: ``advance`` runs at the start of a tick, before anyone moves, so a
released person enters the activity on the same tick their car arrives. Legs
are discovered during the tick in which they start. They keep their scheduled
departure time, but they see link loads as of that tick, so congestion lags
departures by up to one tick.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from loguru import logger
from mpi4py import MPI

from casmsocial.activities import Act, Leg
from casmsocial.road_network import RoadNetwork
from casmsocial.traffic.mobsim import Arrival, VehicleMobsim, VehiclePosition
from casmsocial.traffic.network import TrafficNetwork
from casmsocial.traffic.partition import assign_node_ranks, nearest_place_by_node

MINUTES_PER_DAY = 24 * 60


@dataclass(frozen=True)
class TripRecord:
    """One completed leg, as seen by the traveller's rank."""

    person_id: int
    trip_id: int
    origin_place_id: int | None
    destination_place_id: int | None
    depart_s: float
    scheduled_arrival_s: float
    arrive_s: float
    free_flow_s: float
    n_reroutes: int
    n_migrations: int

    @property
    def travel_time_s(self) -> float:
        return self.arrive_s - self.depart_s

    @property
    def delay_s(self) -> float:
        """Seconds the traveller reached the activity after its scheduled start (0 when early)."""
        return max(0.0, self.arrive_s - self.scheduled_arrival_s)


@dataclass
class _PendingTrip:
    trip_id: int
    origin_place_id: int | None
    destination_place_id: int | None
    depart_s: float
    scheduled_arrival_s: float


class LegTrafficCoupler:
    """Spawn a vehicle for each routed leg and hold its traveller until arrival.

    ``advance`` is collective; every rank must call it once per tick.
    """

    def __init__(self, mobsim: VehicleMobsim, mode: str = "drive", track_positions: bool = True) -> None:
        self.mobsim = mobsim
        self.mode = mode
        self.track_positions = track_positions
        self.rank = mobsim.rank
        self.held: dict[int, _PendingTrip] = {}
        # person_id -> where their car was at the last advance (held travellers on this rank only).
        self.positions: dict[int, VehiclePosition] = {}
        self._place_by_node: dict[int, int] | None = None
        self.trip_records: list[TripRecord] = []
        self.skipped_legs = 0
        self._tick_min = 0.0
        self._prev_tick_min = 0.0
        self._minute_of_day = 0

    @classmethod
    def from_road_network(
        cls,
        comm: MPI.Intracomm,
        road_network: RoadNetwork,
        place_to_rank: Mapping[int, int],
        *,
        mode: str = "drive",
        capacity_vph: Mapping[int, float] | None = None,
        link_model: str = "implied_flow",
        sync_window_s: float = 5.0,
        strategic_share: float = 0.0,
        seed: int = 0,
    ) -> LegTrafficCoupler:
        network = TrafficNetwork(road_network, mode=mode, capacity_vph=capacity_vph, link_model=link_model)
        node_rank = assign_node_ranks(network, place_to_rank)
        mobsim = VehicleMobsim(
            comm,
            network,
            node_rank,
            sync_window_s=sync_window_s,
            strategic_share=strategic_share,
            seed=seed,
        )
        return cls(mobsim, mode=mode)

    # ------------------------------------------------------------------
    # Called by the model
    # ------------------------------------------------------------------
    def advance(self, tick_min: float, minute_of_day: int) -> list[TripRecord]:
        """Run vehicles up to the tick time and release travellers that arrived (collective).

        ``tick_min`` is ``SimTime.tick``: minutes since the start of the run.
        Returns the trips completed for travellers on this rank.
        """
        self._prev_tick_min = self._tick_min
        self._tick_min = float(tick_min)
        self._minute_of_day = int(minute_of_day)
        self.mobsim.advance_to(self._tick_min * 60.0)
        arrivals, positions = self.mobsim.route_home(with_positions=self.track_positions)
        completed = [self._release(arrival) for arrival in arrivals]
        completed = [record for record in completed if record is not None]
        held = self.held
        self.positions = {
            position.person_id: position
            for position in positions
            if position.person_id in held and held[position.person_id].trip_id == position.trip_id
        }
        self.trip_records.extend(completed)
        return completed

    def is_held(self, person_id: int) -> bool:
        return person_id in self.held

    def transit_node(self, person_id: int) -> int | None:
        """Road node a held traveller's car was at, or leaving, at the last ``advance``."""
        position = self.positions.get(person_id)
        return position.node_id if position is not None else None

    def transit_place(self, person_id: int) -> int | None:
        """Nearest snapped place to a held traveller's car; None when not held.

        Before the car's first reported position (it was spawned this tick),
        this is the leg's origin place.
        """
        pending = self.held.get(person_id)
        if pending is None:
            return None
        node_id = self.transit_node(person_id)
        if node_id is None:
            return pending.origin_place_id
        if self._place_by_node is None:
            self._place_by_node = nearest_place_by_node(self.mobsim.network)
        return self._place_by_node.get(node_id, pending.origin_place_id)

    def on_leg_start(
        self,
        person_id: int,
        leg: Leg,
        previous_activity: Act | None,
        next_activity: Act | None,
        minute_of_day: int,
    ) -> bool:
        """Spawn a vehicle for a leg the person just started. Returns True if the person is now held."""
        if (
            person_id in self.held
            or leg.mode != self.mode
            or leg.origin_node_id is None
            or leg.destination_node_id is None
            or previous_activity is None
            or next_activity is None
        ):
            self.skipped_legs += 1
            return False

        # Minutes since the start of the run. The leg began when the previous
        # activity ended, which is at most one tick before it was seen; a leg
        # already under way when the run started departs at the first tick.
        day_start_min = self._tick_min - self._minute_of_day
        depart_min = day_start_min + previous_activity.endtime_min
        if depart_min > day_start_min + minute_of_day:
            depart_min -= MINUTES_PER_DAY  # the previous activity ended yesterday
        depart_min = max(depart_min, self._prev_tick_min)
        gap_min = (next_activity.starttime_min - previous_activity.endtime_min) % MINUTES_PER_DAY
        trip_id = int(round(depart_min))

        spawned = self.mobsim.spawn(
            person_id,
            trip_id,
            int(leg.origin_node_id),
            int(leg.destination_node_id),
            depart_min * 60.0,
            home_rank=self.rank,
        )
        if not spawned:
            self.skipped_legs += 1
            return False
        self.held[person_id] = _PendingTrip(
            trip_id=trip_id,
            origin_place_id=leg.origin_place_id,
            destination_place_id=leg.destination_place_id,
            depart_s=depart_min * 60.0,
            scheduled_arrival_s=(depart_min + gap_min) * 60.0,
        )
        return True

    def global_summary(self) -> dict[str, float]:
        """Mobsim totals plus trip delay over all ranks (collective)."""
        summary: dict[str, float] = dict(self.mobsim.global_stats())
        comm = self.mobsim.comm
        delays = [record.delay_s for record in self.trip_records]
        summary["trips_completed"] = comm.allreduce(len(delays), op=MPI.SUM)
        summary["trips_delayed"] = comm.allreduce(sum(1 for d in delays if d > 0), op=MPI.SUM)
        summary["total_delay_s"] = comm.allreduce(sum(delays), op=MPI.SUM)
        summary["held_now"] = comm.allreduce(len(self.held), op=MPI.SUM)
        summary["skipped_legs"] = comm.allreduce(self.skipped_legs, op=MPI.SUM)
        return summary

    def trip_rows(self) -> list[dict]:
        """Completed trips on this rank as plain dicts (for logging or a casmdb channel)."""
        return [asdict(record) | {"delay_s": record.delay_s} for record in self.trip_records]

    # ------------------------------------------------------------------
    def _release(self, arrival: Arrival) -> TripRecord | None:
        pending = self.held.get(arrival.person_id)
        if pending is None or pending.trip_id != arrival.trip_id:
            logger.warning(
                "Rank {}: arrival for person {} trip {} has no matching held leg.",
                self.rank,
                arrival.person_id,
                arrival.trip_id,
            )
            return None
        del self.held[arrival.person_id]
        return TripRecord(
            person_id=arrival.person_id,
            trip_id=arrival.trip_id,
            origin_place_id=pending.origin_place_id,
            destination_place_id=pending.destination_place_id,
            depart_s=pending.depart_s,
            scheduled_arrival_s=pending.scheduled_arrival_s,
            arrive_s=arrival.arrive_s,
            free_flow_s=arrival.free_flow_s,
            n_reroutes=arrival.n_reroutes,
            n_migrations=arrival.n_migrations,
        )
