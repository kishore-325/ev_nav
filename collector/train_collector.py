import os
import sys
import csv
import cv2
import numpy as np
import time

sys.path.append(os.path.join(os.environ['FLIGHTMARE_PATH']))
sys.path.append(os.path.join(os.environ['FLIGHTMARE_PATH'], 'flightlib'))
sys.path.append(os.path.join(os.environ['FLIGHTMARE_PATH'], 'flightpy', 'flightrl'))
sys.path.append(os.path.join(os.environ['PROJECT_PATH']))

from ruamel.yaml import YAML, dump, RoundTripDumper
from flightgym import VisionEnv_v1
from rpg_baselines.torch.envs.vec_env_wrapper import FlightEnvVec
from expert.expert_policy import ExpertPolicy
from expert.geometric_controller import GeometricVelocityController

class DataCollector:

    def __init__(self):

        self.SMALL_EPS = 1e-5
        self.pos_thresh, self.neg_thresh = 0.2, 0.2

        self.save_dir_raw = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Train/DEPTH_RAW/')
        self.save_dir_eve_raw = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Train/EVENTS_RAW/')
        self.save_dir_labels = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Train/LABELS/')

        log_dir = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Train/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"collection_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._csv_file = open(os.path.join(log_dir, log_filename), 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['timestamp', 'elapsed_time_s', 'message'])
        self._log_start = time.time()

    def log(self, message):
        print(message)
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        elapsed = time.time() - self._log_start
        self._csv_writer.writerow([ts, f'{elapsed:.3f}', message])
        self._csv_file.flush()

    def close_log(self):
        self._csv_file.close()

    def create_env(self, num_envs=10):

        yaml = YAML()
        config_path = os.path.join(
            os.environ['FLIGHTMARE_PATH'],
            'flightpy/configs/vision/config.yaml'
        )

        with open(config_path, 'r') as f:
            cfg = yaml.load(f)

        cfg['simulation']['num_envs']=num_envs

        cfg_str = dump(cfg, Dumper=RoundTripDumper)
        raw_env = VisionEnv_v1(cfg_str, False)
        env = FlightEnvVec(raw_env)

        return env

    def create_directories(self, num_envs):

        for i in range(num_envs):
            os.makedirs(os.path.join(self.save_dir_raw, f'environment_{i+1}'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir_eve_raw, f'environment_{i+1}'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir_labels, f'environment_{i+1}'), exist_ok=True)

        return None

    def stabilize_height(self, env, controller, target_z=3, height_tol=0.2,
                     kp=2.0, kd=0.5, max_vz=2.0, max_steps=500):
        """
        Fly the drone to target_z using a PD controller before data collection.
        Returns when drone is within height_tol of target_z or max_steps exceeded.
        """
        frame_id = 0
        flags = [False]*env.num_envs
        for step in range(max_steps):
            env.getObs()
            quad_state = env.getQuadState()

            actions = []
            for env_id in range(env.num_envs):
                current_z = quad_state[env_id][3]
                current_vz = quad_state[env_id][10]
                height_err = target_z-current_z
                vel_z_cmd = float(np.clip(kp * height_err - kd*current_vz, -max_vz, max_vz))
                if abs(height_err)<=height_tol:
                    if not flags[env_id]:
                        self.log(f"Height stabilized for environment {env_id} at z={current_z:.3f}m in {step} steps")
                        flags[env_id] = True
                    vel_cmd_world = np.array([0.0, 0.0, vel_z_cmd])
                    action = controller.compute_action(vel_cmd_world, quad_state[env_id])
                    actions.append(action)
                else:
                    vel_cmd_world = np.array([1.0, 0.0, vel_z_cmd])
                    action = controller.compute_action(vel_cmd_world, quad_state[env_id])
                    actions.append(action)

            env.step(np.array(actions))
            env.render(frame_id)
            frame_id += 1

            if all(flags):
                    return True
        return False


    def compute_events(self, prev_gray, gray):

        events_tot = []

        for i in range(len(gray)):

            difflog = np.log(gray[i]+self.SMALL_EPS) - np.log(prev_gray[i]+self.SMALL_EPS)  # 260 x 346
            events = np.zeros_like(difflog)

            if np.abs(difflog).max() >= max(self.pos_thresh, self.neg_thresh):
                pos_idx = np.where(difflog>0.0)
                neg_idx = np.where(difflog<0.0)
                events[pos_idx] = (difflog[pos_idx] // self.pos_thresh) * self.pos_thresh
                events[neg_idx] = (difflog[neg_idx] // -self.neg_thresh) * -self.neg_thresh

            events_tot.append(events)

        return events_tot


    def save_images(self, env, max_steps):

        expert = ExpertPolicy()
        controller = GeometricVelocityController()

        env.reset()
        env.move()
        env.render(0)
        while True:
            stable = self.stabilize_height(env, controller, 2.0)
            if stable:
                break
            else:
                env.reset()
        frame_id = 1
        capture_interval = 3
        capture_gap = 10
        pair_stride = capture_interval + capture_gap
        ref_gray = None    # shape = [num_envs, 260, 346]
        sample_id = 0
        config_num = 1
        start = time.time()

        for step in range(max_steps):

            env.getObs()
            raw_obs = env._observation
            quad_state = env.getQuadState()
            actions = []
            vel_body_cmds = []

            for env_id in range(len(raw_obs)):

                vel_cmd_body, _ = expert.compute_expert_velocity(raw_obs[env_id])
                R_wb = raw_obs[env_id, 3:12].reshape(3,3)
                vel_cmd_world = R_wb.T @ vel_cmd_body
                vel_cmd_world[2] = 0.0  # expert navigates horizontally; zero z so controller holds altitude
                action = controller.compute_action(vel_cmd_world, quad_state[env_id])
                actions.append(action)
                vel_body_cmds.append(vel_cmd_body)

            step_mod = step % pair_stride

            # Phase 1 — capture reference frame (frame N: steps 3, 16, 29, ...)
            if step > 0 and step_mod == capture_interval:
                rgb_flat = env.getImage(rgb=True)
                rgb = rgb_flat.reshape(env.num_envs, 260, 346, 3)
                ref_gray = np.zeros((env.num_envs, 260, 346), dtype=np.float32)
                for i in range(env.num_envs):
                    ref_gray[i] = cv2.cvtColor(rgb[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

            # Phase 2 — capture event frame (frame N+3: steps 6, 19, 32, ...) and save
            elif step > 0 and step_mod == 2 * capture_interval and ref_gray is not None:
                rgb_flat = env.getImage(rgb=True)
                rgb = rgb_flat.reshape(env.num_envs, 260, 346, 3)
                gray = np.zeros((env.num_envs, 260, 346), dtype=np.float32)
                for i in range(env.num_envs):
                    gray[i] = cv2.cvtColor(rgb[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

                eve_raw = self.compute_events(ref_gray, gray)

                depth_flat = env.getDepthImage()
                depth = depth_flat.reshape(env.num_envs, 260, 346)

                for i in range(env.num_envs):
                    np.save(os.path.join(self.save_dir_raw, f'environment_{i+1}', f'depth_raw_sample{sample_id:05d}.npy'), depth[i])
                    np.save(os.path.join(self.save_dir_eve_raw, f'environment_{i+1}', f'event_raw_sample{sample_id:05d}.npy'), eve_raw[i])
                    np.save(os.path.join(self.save_dir_labels, f'environment_{i+1}', f'vel_cmd_sample{sample_id:05d}.npy'), vel_body_cmds[i])

                sample_id += 1

            _, _, done, _ = env.step(np.array(actions))
            env.render(frame_id)
            frame_id += 1

            if done.any():
                self.log(f"Environment crashed at step {step}")
                config_num += 1
                self.log(f"Loading new forest configuration {config_num}")
                env.reset()
                env.move()
                env.render(frame_id)
                frame_id += 1
                ref_gray = None
                while True:
                    if step <= 300000:
                        stable = self.stabilize_height(env, controller, 2.0)
                    elif step <= 600000:
                        stable = self.stabilize_height(env, controller, 2.5)
                    else:
                        stable = self.stabilize_height(env, controller, 3.0)

                    if stable:
                        break
                    else:
                        env.reset()

        end = time.time()
        self.log(f"Training Data collection Over.")
        self.log(f"Time Taken = {end-start:.2f}s")

        return None


def main():

    data = DataCollector()
    env = data.create_env(1)
    data.create_directories(env.num_envs)

    data.log("\nConnecting to Unity...........")
    env.connectUnity()
    data.save_images(env, int(9e5))

    env.disconnectUnity()
    env.close()

    data.close_log()

if __name__ == '__main__':
    main()
