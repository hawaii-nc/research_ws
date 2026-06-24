"""
Real f110_gym wrapper for RMA F1Tenth env.
Maps f110_gym's (params-dict, [steer,vel]-action, dict-obs) API
to the Gymnasium-style interface expected by f1tenth_env.py.

Also implements actuator delay (Zhang Section II-E "motor effectiveness" analog):
commanded actions are buffered in a FIFO queue and applied N steps later,
where N = round(delay_seconds / timestep). delay_steering and delay_drive
come from PhysicsRandomizer.sample() (typical F1Tenth servo/ESC latencies
~10-50ms, per literature).
"""
import numpy as np
import os
from collections import deque
from f110_gym.envs.f110_env import F110Env


class RealF110Wrapper:
    """
    Single-agent wrapper around F110Env.
    reset(physics_params) -> obs_vec
    step(action) -> obs_vec, raw_obs_dict, done, info
    """
    def __init__(self, map_name="/f1tenth_gym/examples/example_map", timestep=0.01, track="example_map"):
        self.map_name = map_name
        self.timestep = timestep
        self.track = track
        self.env = None
        self.last_steer = 0.0
        self.last_vel = 0.0
        self.steer_queue = deque([0.0])
        self.drive_queue = deque([0.0])

    def _build_env(self, physics_params):
        # f110_gym physics param overrides from Zhang-style et (grip_factor, mass/inertia scale)
        params = {
            'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562,
            'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74,
            'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189,
            'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319,
            'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
            'width': 0.31, 'length': 0.58
        }
        p = physics_params or {}
        if 'grip_factor' in p:
            params['mu'] = 1.0489 * float(p['grip_factor'])
        if 'mass_scale' in p:
            params['m'] = 3.74 * float(p['mass_scale'])
        if 'inertia_scale' in p:
            params['I'] = 0.04712 * float(p['inertia_scale'])

        self.env = F110Env(map=self.map_name, map_ext=".png",
                            params=params, num_agents=1, timestep=self.timestep)
        # Increase iTTC threshold for more realistic collision detection.
        # Default 0.005s triggers at 4cm at 8m/s -- too lenient, allows wall grazing.
        # 0.05s triggers at 40cm -- within car half-width, stops wall phasing.
        for agent in self.env.sim.agents:
            agent.ttc_thresh = 0.05

    def _setup_delay_queues(self, physics_params):
        """Initialize FIFO delay buffers. Queue length N means an action
        issued now is applied N steps from now (N=0 -> applied immediately)."""
        p = physics_params or {}
        delay_steer_s = float(p.get('delay_steering', 0.0))
        delay_drive_s = float(p.get('delay_drive', 0.0))
        n_steer = max(0, round(delay_steer_s / self.timestep))
        n_drive = max(0, round(delay_drive_s / self.timestep))
        # queue holds N pending zero-actions; popleft() on step gives delayed value
        self.steer_queue = deque([0.0] * n_steer, maxlen=max(n_steer, 1) if n_steer > 0 else None)
        self.drive_queue = deque([0.0] * n_drive, maxlen=max(n_drive, 1) if n_drive > 0 else None)
        self.n_steer = n_steer
        self.n_drive = n_drive

    def reset(self, physics_params=None):
        self._build_env(physics_params)
        self._setup_delay_queues(physics_params)
        # Spawn heading matches track's centerline direction at spawn point.
        # AUT centerline runs at ~0 deg at spawn; example_map runs at ~78 deg.
        # Read heading from config track setting so this works for any map.
        import math
        # Generic spawn heading: find nearest centerline point to [0.7,0.0]
        # and compute heading from neighbors. Works for any map.
        import os as _os
        _cl_path = f'/research_ws/maps/{self.track}_centerline.csv'
        if _os.path.exists(_cl_path):
            _cl = np.loadtxt(_cl_path, delimiter=',')
            _xy = _cl[:, 0:2]
            _dists = np.sqrt((_xy[:,0]-0.7)**2 + (_xy[:,1]-0.0)**2)
            _idx = int(np.argmin(_dists))
            _ip = max(0, _idx-2)
            _in = min(len(_cl)-1, _idx+2)
            _dx = _xy[_in,0] - _xy[_ip,0]
            _dy = _xy[_in,1] - _xy[_ip,1]
            _cl_heading = float(np.arctan2(_dy, _dx))
            # f110_gym: stheta=0 = +x direction, matches centerline heading.
            # No correction needed.
            _stheta = _cl_heading
        else:
            _stheta = 1.37079632679  # example_map default (pi/2 - 0.20)
        poses = np.array([[0.7, 0.0, _stheta]])
        obs, _, _, _ = self.env.reset(poses)
        self.last_steer = 0.0
        self.last_vel = float(obs['linear_vels_x'][0])
        return self._to_vec(obs), obs

    def _apply_delay(self, queue, n, new_value):
        """Push new_value, return the delayed value to actually apply."""
        if n == 0:
            return new_value
        queue.append(new_value)
        return queue.popleft()

    def step(self, action):
        # action: [steer_cmd, throttle_cmd] -> f110_gym wants [steer_angle, velocity]
        steer_cmd, throttle_cmd = float(action[0]), float(action[1])

        # Apply actuator delay (Zhang Section II-E analog: motor effectiveness/latency)
        delayed_steer = self._apply_delay(self.steer_queue, self.n_steer, steer_cmd)
        delayed_throttle = self._apply_delay(self.drive_queue, self.n_drive, throttle_cmd)

        ctrl = np.array([[delayed_steer, delayed_throttle]])
        obs, _, done, info = self.env.step(ctrl)
        self.last_steer = delayed_steer
        self.last_vel = float(obs['linear_vels_x'][0])
        done = bool(done) or bool(obs['collisions'][0])
        return self._to_vec(obs), obs, done, info

    def _to_vec(self, obs):
        # xt = [current_velocity, current_steering(proxy), yaw_rate]
        return np.array([
            obs['linear_vels_x'][0],
            self.last_steer,
            obs['ang_vels_z'][0],
        ], dtype=np.float32)
