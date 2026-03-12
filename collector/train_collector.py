import os
import sys
import csv
import cv2
import numpy as np
import time
import subprocess

sys.path.append(os.path.join(os.environ['FLIGHTMARE_PATH']))
sys.path.append(os.path.join(os.environ['FLIGHTMARE_PATH'], 'flightlib'))
sys.path.append(os.path.join(os.environ['FLIGHTMARE_PATH'], 'flightpy', 'flightrl'))
sys.path.append(os.path.join(os.environ['PROJECT_PATH']))

from ruamel.yaml import YAML, dump, RoundTripDumper
from flightgym import VisionEnv_v1
from rpg_baselines.torch.envs.vec_env_wrapper import FlightEnvVec
from expert.expert_policy import ExpertPolicy
from expert.geometric_controller import GeometricVelocityController

class Train_DataCollector:

    def __init__(self):

        self.SMALL_EPS = 1e-5
        self.pos_thresh, self.neg_thresh = 0.2, 0.2

        self.save_dir_raw = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Train/DEPTH_RAW/')
        self.save_dir_eve_raw = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Train/EVENTS_RAW/')
        #self.save_dir_labels = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Train/LABELS/')

        log_dir = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Train/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"collection_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._csv_file = open(os.path.join(log_dir, log_filename), 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['timestamp', 'elapsed_time_s', 'message'])
        self._log_start = time.time()

        self.config_num = 1

    def log(self, message):
        print(message)
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        elapsed = time.time() - self._log_start
        self._csv_writer.writerow([ts, f'{elapsed:.3f}', message])
        self._csv_file.flush()

    def close_log(self):
        self._csv_file.close()

    def create_env(self, num_envs=1):

        yaml = YAML()
        config_path = os.path.join(
            os.environ['FLIGHTMARE_PATH'],
            'flightpy/configs/vision/config.yaml'
        )

        with open(config_path, 'r') as f:
            cfg = yaml.load(f)

        cfg['simulation']['num_envs'] = num_envs

        cfg_str = dump(cfg, Dumper=RoundTripDumper)
        raw_env = VisionEnv_v1(cfg_str, False)
        env = FlightEnvVec(raw_env)

        return env

    def create_directories(self, num_envs):

        for i in range(num_envs):
            os.makedirs(os.path.join(self.save_dir_raw, f'environment_{i+1}'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir_eve_raw, f'environment_{i+1}'), exist_ok=True)
            #os.makedirs(os.path.join(self.save_dir_labels, f'environment_{i+1}'), exist_ok=True)

        return None

    def stabilize_height(self, env, controller, target_z=3, height_tol=0.2,
                     kp=2.0, kd=0.5, max_vz=2.0, max_steps=500, frame_id=0):
        """
        Fly the drone to target_z using a PD controller before data collection.
        Returns when drone is within height_tol of target_z or max_steps exceeded.
        """
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
                    return True, frame_id
        return False, frame_id


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


    def save_images(self, env, samples_per_height, mode, sample_id):
        """
        Collect samples across 3 expert speeds (3, 5, 7 m/s) x 3 heights (2.0, 2.5, 3.0m).
        Stops each segment after samples_per_height samples.
        sample_id and config_num persist across all segments.
        """
        controller = GeometricVelocityController()

        capture_interval = 3
        capture_gap = 35
        pair_stride = capture_interval + capture_gap
        desired_vels = [3.0, 5.0, 7.0]
        heights = [2.0, 2.5, 3.0]

        A = 2.0
        f = 0.5
        fps = 100

        start = time.time()

        for desired_vel in desired_vels:

            self.log(f"\n****************MODE : {mode}*******************")
            if mode == "Expert":
                expert = ExpertPolicy(desired_vel=desired_vel)
            self.log(f"\n=== Starting collection at desired_vel={desired_vel}m/s ===")

            for target_z in heights:

                self.log(f"\n--- Collecting {samples_per_height} samples at height {target_z}m ---")

                env.reset()
                env.move()
                frame_id = 0
                env.render(frame_id)
                frame_id += 1

                while True:
                    stable, frame_id = self.stabilize_height(env, controller, target_z, frame_id=frame_id)
                    if stable:
                        break
                    else:
                        env.reset()

                ref_gray = None
                capture_gap = 35
                pair_stride = capture_interval + capture_gap
                height_sample_count = 0
                step = 0

                while height_sample_count < samples_per_height:

                    env.getObs()
                    raw_obs = env._observation
                    quad_state = env.getQuadState()
                    actions = []
                    #vel_body_cmds = []

                    for env_id in range(len(raw_obs)):

                        if mode == "Expert":
                            vel_cmd_body, _ = expert.compute_expert_velocity(raw_obs[env_id])
                            
                        elif mode == "Straight":
                            vel_cmd_body = np.array([desired_vel, 0.0, 0.0])
                        elif mode == "Sinusoidal":
                                vy = A * np.sin(2 * np.pi * f * step / fps)
                                vel_cmd_body = np.array([desired_vel, vy, 0.0])

                        
                        R_wb = raw_obs[env_id, 3:12].reshape(3,3)
                        vel_cmd_world = R_wb.T @ vel_cmd_body
                        vel_cmd_world[2] = 0.0
                        action = controller.compute_action(vel_cmd_world, quad_state[env_id])
                        actions.append(action)
                        #vel_body_cmds.append(vel_cmd_body)

                    step_mod = step % pair_stride

                    # Phase 1 — capture reference frame
                    if step > 0 and step_mod == capture_interval:
                        rgb_flat = env.getImage(rgb=True)
                        rgb = rgb_flat.reshape(env.num_envs, 260, 346, 3)
                        ref_gray = np.zeros((env.num_envs, 260, 346), dtype=np.float32)
                        for i in range(env.num_envs):
                            ref_gray[i] = cv2.cvtColor(rgb[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

                    # Phase 2 — capture event frame and save
                    elif step > 0 and step_mod == (2 * capture_interval) % pair_stride and ref_gray is not None:
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
                            #np.save(os.path.join(self.save_dir_labels, f'environment_{i+1}', f'vel_cmd_sample{sample_id:05d}.npy'), vel_body_cmds[i])

                        sample_id += 1
                        height_sample_count += 1

                    _, _, done, _ = env.step(np.array(actions))
                    env.render(frame_id)
                    frame_id += 1
                    step += 1

                    if done.any():
                        self.log(f"Crash at step {step} ({height_sample_count}/{samples_per_height} samples at {target_z}m, {desired_vel}m/s)")
                        self.config_num += 1
                        self.log(f"Loading new forest configuration {self.config_num}")
                        env.reset()
                        env.move()
                        env.render(frame_id)
                        frame_id += 1
                        ref_gray = None
                        while True:
                            stable, frame_id = self.stabilize_height(env, controller, target_z, frame_id=frame_id)
                            if stable:
                                break
                            else:
                                env.reset()

                self.log(f"Completed {samples_per_height} samples at {target_z}m, {desired_vel}m/s (total so far: {sample_id})")

        end = time.time()
        self.log(f"Training Data collection complete. Total samples: {sample_id}")
        self.log(f"Time Taken = {end-start:.2f}s")

        return sample_id


def main():

    subprocess.run(['python3', '/home/kishore/ev_nav/flightmare/flightpy/configs/vision/custom_sim_forest_environment/tree_deleter.py'], check=True)

    print("\n****************** TRAIN COLLECTOR ********************")
    
    data = Train_DataCollector()
    env = data.create_env(1)
    data.create_directories(env.num_envs)

    data.log("\nConnecting to Unity for Expert...........")
    env.connectUnity()
    sample_id = data.save_images(env, 3500, "Expert", 0)   

    env.disconnectUnity()
    env.close()

    data.close_log()

    time.sleep(20.0)
    subprocess.run(['python3', '/home/kishore/ev_nav/collector/close_range_collector.py', '--mode', 'train', '--sample_id', str(sample_id)], check=True)

if __name__ == '__main__':
    main()
