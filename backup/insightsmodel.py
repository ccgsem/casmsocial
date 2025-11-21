from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from mpi4py import MPI
from repast4py import context, core, random, schedule, space
from repast4py.space import BoundingBox, ContinuousPoint, DiscretePoint, SharedGrid

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Place(core.Agent):
    TYPE = 1

    def __init__(self, agent_id, rank, x, y, lat, lon, is_remote=False):
        super().__init__(agent_id, rank)
        self.x, self.y = x, y
        self.latitude = lat
        self.longitude = lon
        self.is_remote = is_remote


class Person(core.Agent):
    TYPE = 0

    def __init__(self, agent_id, rank, place_id, behavior_id, social_links=None, environment=None):
        super().__init__(agent_id, rank)
        self.place_id = place_id
        self.behavior_id = behavior_id
        self.energy = 100
        self.social_links = social_links or []
        self.environment = (
            np.zeros(2, dtype=np.float32) if environment is None else np.array(environment, dtype=np.float32)
        )
        self.at_remote_place = False

    def move(self, new_place_id, context):
        self.place_id = new_place_id
        dest_place = context.get(new_place_id)
        self.at_remote_place = dest_place.is_remote

    def step(self, hour, day):
        if day in [5, 6]:
            self.energy = min(self.energy + 3, 100)
        elif 9 <= hour < 17:
            self.energy -= 2
        else:
            self.energy += 1


class PersonPlaceModel:
    def __init__(self, num_persons=1000, num_places=100, grid_resolution=1000):
        assert hasattr(self, "step") or callable(
            getattr(self, "step", None)
        ), "step() must be defined before scheduling."
        self.context = context.SharedContext(comm)
        self.runner = schedule.init_schedule_runner(comm)
        # self.schedule = schedule.Schedule()
        self.grid_resolution = grid_resolution

        ds = xr.open_dataset("environment.nc")
        self.env_var = "temperature"
        self.time_dim = ds.coords["time"].values

        self.min_x, self.min_y = float(ds.x.min()), float(ds.y.min())
        self.width = len(ds.x) * grid_resolution
        self.height = len(ds.y) * grid_resolution
        self.grid_bounds = BoundingBox(int(self.min_x), int(self.min_y), self.width, self.height)

        self.space = space.SharedCSpace(
            "space",
            self.grid_bounds,
            space.BorderType.Sticky,
            space.OccupancyType.Multiple,
            2,
            comm,
            tree_threshold=100,
        )

        self.grid = SharedGrid("grid", self.grid_bounds, space.BorderType.Sticky, space.OccupancyType.Multiple, 1, comm)

        self.context.add_projection(self.space)
        self.context.add_projection(self.grid)

        self.grid_width = len(ds.x)
        self.grid_height = len(ds.y)
        self.grid_env = np.zeros((self.grid_width, self.grid_height, 1), dtype=np.float32)
        self.env_data = ds

        self.places, self.place_ids, self.remote_places = [], [], []
        self._init_places(num_places)
        self.persons = []
        self._init_persons(num_persons)

        self.start_datetime = datetime(2025, 1, 1)
        self.current_tick = 0
        self.current_datetime = self.start_datetime

        # self.evt =
        self.runner.schedule.schedule_repeating_event(1, self.step, None)  # Ensure self.step is a valid method

    def _within_bounds(self, x, y):
        return self.min_x <= x <= self.min_x + self.width and self.min_y <= y <= self.min_y + self.height

    def _init_places(self, count):
        for i in range(count):
            x = random.default_rng.uniform(self.min_x - 10000, self.min_x + self.width + 10000)
            y = random.default_rng.uniform(self.min_y - 10000, self.min_y + self.height + 10000)
            lat = random.default_rng.uniform(50, 70)
            lon = random.default_rng.uniform(-170, -130)
            is_remote = not self._within_bounds(x, y)
            place = Place(i + 100000, rank, x, y, lat, lon, is_remote)
            self.context.add(place)
            if not is_remote:
                self.space.add(place)
                self.space.move(place, ContinuousPoint(x, y))
                gx, gy = int((x - self.min_x) / self.grid_resolution), int((y - self.min_y) / self.grid_resolution)
                self.grid.add(place)
                self.grid.move(place, DiscretePoint(gx, gy))
            elif rank == 0:
                self.remote_places.append(place)
            self.places.append(place)
            self.place_ids.append(place.id)

    def _init_persons(self, count):
        all_ids = list(range(count))
        for i in range(count):
            place = random.default_rng.choice(self.places)
            behavior_id = random.default_rng.integers(0, 2)
            links = random.default_rng.choice(all_ids, size=3, replace=False).tolist()
            if i in links:
                links.remove(i)
            person = Person(i, rank, place.id, behavior_id, links)
            self.context.add(person)
            self.space.add(person)
            self.space.move(person, ContinuousPoint(place.x, place.y))
            self.persons.append(person)

    def _get_env_slice(self):
        model_time = np.datetime64(self.current_datetime)
        idx = int(np.searchsorted(self.time_dim, model_time, side="left"))
        return self.env_data[self.env_var].isel(time=idx).values

    def step(self):
        self.current_tick += 1
        self.current_datetime = self.start_datetime + timedelta(hours=self.current_tick)
        hour, day = self.current_datetime.hour, self.current_datetime.weekday()
        self.grid_env[:, :, 0] = self._get_env_slice()

        for person in self.persons:
            dest_id = random.default_rng.choice(self.place_ids)
            person.move(dest_id, self.context)
            place = self.context.get(dest_id)
            if not place.is_remote:
                self.space.move(person, ContinuousPoint(place.x, place.y))
                gx = int((place.x - self.min_x) / self.grid_resolution)
                gy = int((place.y - self.min_y) / self.grid_resolution)
                gx = np.clip(gx, 0, self.grid_width - 1)
                gy = np.clip(gy, 0, self.grid_height - 1)
                self.grid.move(person, DiscretePoint(gx, gy))
                person.environment[0] = self.grid_env[gx, gy, 0]
            else:
                person.environment[0] = -9999

        self._place_updates(hour, day)

    def _place_updates(self, hour, day):
        place_idx = {p.id: i for i, p in enumerate(self.places)}
        id_to_idx, flat_ids, offsets = {}, [], [0]
        local_agents = [a for a in self.context.agents() if isinstance(a, Person)]

        for idx, agent in enumerate(local_agents):
            id_to_idx[agent.id] = idx
        groups = [[] for _ in range(len(self.places))]
        for person in local_agents:
            if person.place_id in place_idx:
                groups[place_idx[person.place_id]].append(person.id)
        for g in groups:
            flat_ids.extend(g)
            offsets.append(len(flat_ids))

        for i in range(len(offsets) - 1):
            for j in range(offsets[i], offsets[i + 1]):
                local_agents[id_to_idx[flat_ids[j]]].step(hour, day)


def run_model():
    model = PersonPlaceModel()
    if model.runner.schedule is None:
        print("Schedule is not initialized.")
        # raise ValueError("Schedule is not initialized.")
        return

    model.runner.schedule.execute()
    # for _ in range(10):
    #    model.schedule.execute()


if __name__ == "__main__":
    run_model()
