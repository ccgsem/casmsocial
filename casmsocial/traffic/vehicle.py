"""Lightweight vehicle agent that carries one trip through the road network.

A casmsocial ``Person`` carries plans, cognition and message buffers, which
is too much to migrate at every rank boundary on a route. The mobsim spawns a
``Vehicle`` for each trip instead. The vehicle migrates the way BATSim's
``Individual`` does, while the person stays on its ``rank_place_id`` anchor.
"""

from __future__ import annotations

from repast4py import core

VEHICLE_AGENT_TYPE = 100
NO_EDGE = -1


class Vehicle(core.Agent):
    """One trip in progress.

    Invariant: a vehicle lives on the rank that owns ``node_id`` while it waits
    at a node, and on the rank that owns its current edge's start node while
    on that edge. Edge vehicle counts therefore stay rank-local, so the mobsim
    needs no global reduction of link loads.
    """

    TYPE = VEHICLE_AGENT_TYPE

    def __init__(
        self,
        local_id: int,
        rank: int,
        person_id: int,
        trip_id: int,
        node_id: int,
        dest_node_id: int,
        depart_s: float,
        path: list[int],
        *,
        path_pos: int = 0,
        cur_edge: int = NO_EDGE,
        edge_enter_s: float = 0.0,
        event_s: float | None = None,
        ff_s: float = 0.0,
        strategic: bool = False,
        n_reroutes: int = 0,
        n_migrations: int = 0,
    ) -> None:
        super().__init__(id=local_id, type=Vehicle.TYPE, rank=rank)
        self.person_id = int(person_id)
        self.trip_id = int(trip_id)
        self.node_id = int(node_id)
        self.dest_node_id = int(dest_node_id)
        self.depart_s = float(depart_s)
        self.path = list(path)
        self.path_pos = int(path_pos)
        self.cur_edge = int(cur_edge)
        self.edge_enter_s = float(edge_enter_s)
        self.event_s = float(depart_s if event_s is None else event_s)
        self.ff_s = float(ff_s)
        self.strategic = bool(strategic)
        self.n_reroutes = int(n_reroutes)
        self.n_migrations = int(n_migrations)

    @property
    def on_edge(self) -> bool:
        return self.cur_edge != NO_EDGE

    @property
    def remaining_path(self) -> list[int]:
        return self.path[self.path_pos :]

    def next_edge(self) -> int | None:
        if self.path_pos >= len(self.path):
            return None
        return self.path[self.path_pos]

    def replace_path(self, path: tuple[int, ...] | list[int]) -> None:
        self.path = list(path)
        self.path_pos = 0

    def save(self) -> tuple:
        return (
            self.uid,
            (
                self.person_id,
                self.trip_id,
                self.node_id,
                self.dest_node_id,
                self.depart_s,
                tuple(self.path[self.path_pos :]),
                self.cur_edge,
                self.edge_enter_s,
                self.event_s,
                self.ff_s,
                self.strategic,
                self.n_reroutes,
                self.n_migrations,
            ),
        )

    @staticmethod
    def restore(saved: tuple) -> Vehicle:
        uid, state = saved
        (
            person_id,
            trip_id,
            node_id,
            dest_node_id,
            depart_s,
            path,
            cur_edge,
            edge_enter_s,
            event_s,
            ff_s,
            strategic,
            n_reroutes,
            n_migrations,
        ) = state
        return Vehicle(
            uid[0],
            uid[2],
            person_id,
            trip_id,
            node_id,
            dest_node_id,
            depart_s,
            list(path),
            cur_edge=cur_edge,
            edge_enter_s=edge_enter_s,
            event_s=event_s,
            ff_s=ff_s,
            strategic=strategic,
            n_reroutes=n_reroutes,
            n_migrations=n_migrations,
        )
