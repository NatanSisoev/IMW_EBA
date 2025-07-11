import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from numpy.linalg import inv

from visualizer import visualize_bus_stations


class BusSimulator:
    def __init__(self, stops_separation: np.ndarray, stops_demand: pd.DataFrame, max_speed: np.float64 = 50 / 3.6,
                 acc_time: np.float64 = 10,
                 brk_time: np.float64 = 5, dwell_t0: np.float64 = 5, dwell_alpha: np.float64 = 2,
                 traffic_light_dur: int = 30, gamma: np.float64 = 30 / 3600, granularity: np.float64 = 10) -> None:
        self.stops_separation = stops_separation.flatten()
        self.stops = np.cumsum(self.stops_separation)

        self.hour = None
        self.hour_f = None

        self.gamma = gamma
        self.traffic = True
        self.traffic_lights = True

        self.demands = stops_demand
        self.demands_per_hour = self.demands.sum(axis=0)
        self.demands_per_stop = self.demands.sum(axis=1)

        self.current_stop = 0
        self.t0 = 0

        self.dwell_t0 = dwell_t0
        self.dwell_alpha = dwell_alpha

        self.max_speed = max_speed
        self.traffic_light_dur = traffic_light_dur

        self.acc_time = acc_time
        self.brk_time = brk_time

        self.acc_coeffs = None
        self.brk_coeffs = None

        self.acc_len = None
        self.brk_len = None

        self.done = False
        self.granularity = granularity

        self.ts = []
        self.vs = []
        self.labels = []

        self.driving_time = []
        self.dwell_time = []

    @classmethod
    def read_csv(cls, filename: str = "data/data.csv"):
        df = pd.read_csv(filename)
        stops = df["Distància"].fillna(0).to_numpy()
        demand = df[[f"{h}:00" for h in range(6, 24)]].fillna(0)
        return cls(stops, demand)

    @property
    def trip_time(self):
        return self.ts[-1][-1]

    def car_speed(self):
        return self.max_speed - self.gamma * self.demands_per_hour[self.hour_f]

    @property
    def bus_speed(self):
        if not self.traffic:
            return self.max_speed

        p = (self.demands_per_hour[self.hour_f] / np.max(self.demands_per_hour.to_numpy())
             * self.demands[self.hour_f][self.current_stop] / np.max(self.demands[self.hour_f]))

        return p * self.car_speed() + (1 - p) * self.max_speed

    def initial_dwell(self):
        self.current_stop = -1
        ts_dwell = np.linspace(0, self.dwell, int(self.dwell * self.granularity))
        self.ts.append(ts_dwell)
        self.vs.append(np.zeros_like(ts_dwell))
        self.labels.append(np.array(["dwell"] * len(ts_dwell)))
        self.dwell_time.append(self.dwell)
        self.t0 = self.dwell
        self.current_stop = 0

    def calculate_acc_coeffs(self):
        acc_mat = np.array([
            [0, 0, 0, 1],
            [self.acc_time ** 3, self.acc_time ** 2, self.acc_time, 1],
            [0, 0, 1, 0],
            [3 * self.acc_time ** 2, 2 * self.acc_time, 1, 0],
        ])
        acc_ind = np.array([0, self.bus_speed, 0, 0])
        acc_coeffs = inv(acc_mat).dot(acc_ind)

        self.acc_coeffs = acc_coeffs

        return self.acc_coeffs

    def calculate_brk_coeffs(self):
        brk_mat = np.array([
            [0, 0, 0, 1],
            [self.brk_time ** 3, self.brk_time ** 2, self.brk_time, 1],
            [3 * 0, 2 * 0, 1, 0],
            [3 * self.brk_time ** 2, 2 * self.brk_time, 1, 0],
        ])
        brk_ind = np.array([self.bus_speed, 0, 0, 0])
        brk_coeffs = inv(brk_mat).dot(brk_ind)

        self.brk_coeffs = brk_coeffs

        return self.brk_coeffs

    def calculate_acc_len(self):
        self.acc_len = self.acc_coeffs[0] / 4 * self.acc_time ** 4 + self.acc_coeffs[1] / 3 * self.acc_time ** 3 + \
                       self.acc_coeffs[2] / 2 * self.acc_time ** 2 + self.acc_coeffs[3] * self.acc_time
        return self.acc_len

    def calculate_brk_len(self):
        self.brk_len = self.brk_coeffs[0] / 4 * self.brk_time ** 4 + self.brk_coeffs[1] / 3 * self.brk_time ** 3 + \
                       self.brk_coeffs[2] / 2 * self.brk_time ** 2 + self.brk_coeffs[3] * self.brk_time
        return self.brk_len

    @property
    def const_len(self):
        return self.stops[self.current_stop + 1] - self.stops[self.current_stop] - self.acc_len - self.brk_len

    @property
    def const_time(self):
        return self.const_len / self.bus_speed

    def accelerate(self, t: np.float64 | np.ndarray) -> np.float64:
        t0 = self.t0
        return self.acc_coeffs[0] * (t - t0) ** 3 + self.acc_coeffs[1] * (t - t0) ** 2 + self.acc_coeffs[2] * (t - t0) + \
            self.acc_coeffs[3]

    def brake(self, t: np.float64 | np.ndarray) -> np.float64:
        t0 = self.t0 + self.acc_time + self.const_time + (
            self.traffic_light_dur if self.traffic_lights and (self.current_stop % 2) else 0)
        return self.brk_coeffs[0] * (t - t0) ** 3 + self.brk_coeffs[1] * (t - t0) ** 2 + self.brk_coeffs[2] * (t - t0) + \
            self.brk_coeffs[3]

    def speed(self, t: np.float64 | np.ndarray) -> np.ndarray:
        y = np.where(
            (self.t0 <= t) & (t <= self.t0 + self.acc_time),
            self.accelerate(t),
            np.where(
                (self.t0 + self.acc_time + self.const_time + (
                    self.traffic_light_dur if self.traffic_lights and (self.current_stop % 2) else 0) <= t) & (
                        t <= self.t0 + self.acc_time + self.const_time + (
                    self.traffic_light_dur if self.traffic_lights and (self.current_stop % 2) else 0) + self.brk_time),
                self.brake(t),
                np.where(
                    (self.t0 + self.acc_time <= t) & (t <= self.t0 + self.acc_time + self.const_time + (
                        self.traffic_light_dur if self.traffic_lights and (self.current_stop % 2) else 0)),
                    self.bus_speed,
                    0
                )
            )
        )

        return y

    @property
    def dwell(self):
        return self.dwell_t0 + self.demands[self.hour_f][self.current_stop + 1] * self.dwell_alpha

    def advance(self):
        self.calculate_acc_coeffs()
        self.calculate_brk_coeffs()

        self.calculate_acc_len()
        self.calculate_brk_len()

        dur = self.acc_time + self.const_time + self.brk_time
        self.driving_time.append(dur)

        # BASE (+ TRAFFIC?) (+ TRAFFIC LIGHTS?)
        if self.traffic_lights and (self.current_stop % 2):
            ts = np.linspace(self.t0, self.t0 + dur + self.traffic_light_dur,
                             int((dur + self.traffic_light_dur) * self.granularity))
            vs = np.concatenate((
                self.speed(ts[:int((self.acc_time + self.const_time / 2) * self.granularity)]),
                np.zeros(self.traffic_light_dur * self.granularity - 1),
                self.speed(ts[int(self.traffic_light_dur * self.granularity) + int(
                    (self.acc_time + self.const_time / 2) * self.granularity) + 1:])
            ))
            self.labels.append(
                np.array(["moving"] * len(ts[:int((self.acc_time + self.const_time / 2) * self.granularity)])))
            self.labels.append(np.array(["traffic_light"] * (self.traffic_light_dur * self.granularity - 1)))
            self.labels.append(np.array(["moving"] * len(ts[int(self.traffic_light_dur * self.granularity) + int(
                (self.acc_time + self.const_time / 2) * self.granularity):])))
        else:
            ts = np.linspace(self.t0, self.t0 + dur, int(dur * self.granularity))
            vs = self.speed(ts)
            self.labels.append(np.array(["moving"] * len(ts[1:])))

        self.ts.append(ts[1:])
        self.vs.append(vs[1:])

        # DWELL
        ts_dwell = np.linspace(
            self.t0 + dur + (self.traffic_light_dur if self.traffic_lights and (self.current_stop % 2) else 0),
            self.t0 + dur + (
                self.traffic_light_dur if self.traffic_lights and (self.current_stop % 2) else 0) + self.dwell,
            int(self.dwell * self.granularity))
        self.ts.append(ts_dwell[1:])
        self.vs.append(np.zeros_like(ts_dwell[1:]))
        self.dwell_time.append(self.dwell)
        self.labels.append(np.array(["dwell"] * len(ts_dwell[1:])))

        # NEXT
        self.t0 = self.ts[-1][-1]
        if self.current_stop < len(self.stops) - 2:
            self.current_stop += 1
        else:
            self.done = True

    def flush(self):
        self.hour = None
        self.hour_f = None

        self.traffic = True

        self.current_stop = 0
        self.t0 = 0

        self.acc_coeffs = None
        self.brk_coeffs = None

        self.acc_len = None
        self.brk_len = None

        self.done = False

        self.ts = []
        self.vs = []
        self.labels = []

        self.driving_time = []
        self.dwell_time = []

    def run(self, hour: int = 9, traffic: bool = True, traffic_lights: bool = True):
        self.flush()

        self.hour = hour
        self.hour_f = f"{hour}:00"

        self.traffic = traffic
        self.traffic_lights = traffic_lights

        self.initial_dwell()

        while not self.done:
            self.advance()

    def plot(self):
        if len(self.ts) == 0:
            print(f"No simulation data found. First run the simulator.")
            return

        ts = np.concatenate(self.ts) / 60
        vs = np.concatenate(self.vs) * 3.6

        fig = go.Figure()

        # Main velocity trace
        fig.add_trace(go.Scatter(x=ts, y=vs, mode='lines', name='Velocity', line=dict(color='blue')))

        label_color_map = {
            "moving": "green",
            "dwell": "grey",
            "traffic_light": "red",
        }

        labels = np.concatenate(self.labels)

        def get_segments(mask, ts):
            segments = []
            in_segment = False
            for i, val in enumerate(mask):
                if val and not in_segment:
                    start = ts[i]
                    in_segment = True
                elif not val and in_segment:
                    end = ts[i]
                    segments.append((start, end))
                    in_segment = False
            if in_segment:
                segments.append((start, ts[-1]))
            return segments

        for label, color in label_color_map.items():
            mask = labels == label
            segments = get_segments(mask, ts)
            for x0, x1 in segments:
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=color,
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                )

        # Layout
        fig.update_layout(
            title=f"H12 Velocity Profile, {self.hour_f}",
            xaxis_title="Time (min)",
            yaxis_title="Velocity (km/h)",
            hovermode='x unified',
            template='plotly_white'
        )

        fig.show()
