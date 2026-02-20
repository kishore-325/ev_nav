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
        self.save_dir_rgb = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Test/RGB_IMAGES/')
        self.save_dir_dep = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Test/DEPTH_IMAGES/')
        self.save_dir_raw = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Test/DEPTH_RAW/')
        self.save_dir_eve = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Test/EVENTS_VIS/')
        self.save_dir_eve_raw = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Test/EVENTS_RAW/')
        self.save_dir_labels = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Test/LABELS/')

        log_dir = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Test/logs/')
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
            os.makedirs(os.path.join(self.save_dir_rgb, f'environment_{i+1}'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir_dep, f'environment_{i+1}'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir_raw, f'environment_{i+1}'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir_eve, f'environment_{i+1}'), exist_ok=True)
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
        events_vis_tot = []

        for i in range(len(gray)):

            # raw difflog for network input
            difflog = np.log(gray[i]+self.SMALL_EPS) - np.log(prev_gray[i]+self.SMALL_EPS)  # 260 x 346
            events = np.zeros_like(difflog)

            # check atleast one change greater than threshold
            if np.abs(difflog).max() >= max(self.pos_thresh, self.neg_thresh):
                pos_idx = np.where(difflog>0.0)
                neg_idx = np.where(difflog<0.0)
                events[pos_idx] = (difflog[pos_idx] // self.pos_thresh) * self.pos_thresh
                events[neg_idx] = (difflog[neg_idx] // -self.neg_thresh) * -self.neg_thresh 

            # Red/Blue Visualization
            scale = np.percentile(np.abs(events), 80)
            if scale>0:
                events_sc = np.clip(events/scale, -1.0, 1.0)
            else:
                events_sc = events.copy()
            events_vis = np.zeros((260, 346,3), dtype=np.uint8)
            pos = events_sc>0
            neg = events_sc<0
            events_vis[pos, 2] = (255 * events_sc[pos]).astype(np.uint8)    # Red chanel
            events_vis[neg, 0] = (255 * -events_sc[neg]).astype(np.uint8)   # Blue chanel

            events_tot.append(events)
            events_vis_tot.append(events_vis)
        
        return events_tot, events_vis_tot
    
    
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
        prev_gray = None    # shape = [num_envs, 260, 346]
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

            if step>0 and step%capture_interval == 0:

                
                # Get RGB images
                rgb_flat = env.getImage(rgb=True)   # shape = [num_envs, W*H*3]
                rgb = rgb_flat.reshape(env.num_envs, 260, 346, 3)

                # convert to gray
                gray = np.zeros((env.num_envs, 260, 346), dtype=np.float32)         # shape = [num_envs, 260, 346]
                for i in range(env.num_envs):
                    gray[i] = cv2.cvtColor(rgb[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0        # divide 255.0 to get [0,1]

                # compute raw events and visualiztion
                if prev_gray is None:
                    prev_gray = gray.copy()
                else:
                    eve_raw, eve_vis = self.compute_events(prev_gray, gray)
                    prev_gray = gray.copy()

                    depth_flat = env.getDepthImage()    # shape = [num_envs, W*H]
                    depth = depth_flat.reshape(env.num_envs, 260, 346)

                    depth_vis = depth.copy()
                    depth_vis[depth_vis<0.0] = 0.0
                    depth_vis[depth_vis>20.0] = 20.0
                    depth_vis = depth_vis * 255

                    for i in range(env.num_envs):
                        cv2.imwrite(os.path.join(self.save_dir_rgb, f'environment_{i+1}', f'rgb_sample{sample_id:05d}.png'), rgb[i])
                        cv2.imwrite(os.path.join(self.save_dir_dep, f'environment_{i+1}', f'depth_sample{sample_id:05d}.png'), depth_vis[i].astype(np.uint8))
                        np.save(os.path.join(self.save_dir_raw, f'environment_{i+1}', f'depth_raw_sample{sample_id:05d}.npy'), depth[i])
                        cv2.imwrite(os.path.join(self.save_dir_eve, f'environment_{i+1}', f'event_vis_sample_{sample_id:05d}.png'), eve_vis[i])
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
                prev_gray = None
                while True:
                    if step <= 13333:
                        stable = self.stabilize_height(env, controller, 2.0)
                    elif step > 13333 and step <= 26666:
                        stable = self.stabilize_height(env, controller, 2.5)
                    else:
                        stable = self.stabilize_height(env, controller, 3.0)
                    
                    if stable:
                        break
                    else:
                        env.reset()

        end = time.time()
        self.log(f"Data collection Over.")
        self.log(f"Time Taken = {end-start:.2f}s")

        return None
                    

def main():

    data = DataCollector()
    env = data.create_env(1)
    data.create_directories(env.num_envs)

    data.log("\nConnecting to Unity...........")
    env.connectUnity()
    data.save_images(env, int(4e4))

    env.disconnectUnity()
    env.close()

    data.close_log()

if __name__ == '__main__':
    main()