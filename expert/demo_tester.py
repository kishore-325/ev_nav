import os
import numpy as np
import sys

sys.path.append(os.environ['FLIGHTMARE_PATH'])
sys.path.append(os.environ['PROJECT_PATH'])

from ruamel.yaml import YAML, dump, RoundTripDumper
from flightgym import VisionEnv_v1
from rpg_baselines.torch.envs.vec_env_wrapper import FlightEnvVec

from expert_policy import ExpertPolicy
from geometric_controller import GeometricVelocityController

def create_env(num_envs):

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

def stabilize_height(env, controller, target_z=3, height_tol=0.2,
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
                        print(f"Height stabilized for environment {env_id} at z={current_z:.3f}m in {step} steps")
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

def main():

    expert = ExpertPolicy()
    controller = GeometricVelocityController()

    env = create_env(1)
    env.connectUnity()

    env.reset()
    env.move()

    frame_id = 0
    env.render(frame_id)
    frame_id+=1

    stable, frame_id = stabilize_height(env, controller, frame_id=frame_id)

    for step in range(1000000):

        # Use raw (un-normalized) observations — expert needs actual obstacle positions
        env.getObs()
        raw_obs = env._observation[0]  # shape (55,), raw values

        quad_state = env.getQuadState()

        vel_cmd_body, _ = expert.compute_expert_velocity(raw_obs)

        # Convert vel_cmd from body frame to world frame
        # obs[3:12] is Eigen column-major flattened, numpy reshape gives R^T (world-to-body)
        # So R.T is body-to-world
        R_wb = raw_obs[3:12].reshape(3, 3)   # world-to-body (due to column-major → row-major)
        vel_cmd_world = R_wb.T @ vel_cmd_body  # body-to-world
        vel_cmd_world[2] = 0.0

        action = controller.compute_action(vel_cmd_world, quad_state[0])
        _, rew, done, info = env.step(action)
        env.render(frame_id)
        frame_id += 1

        if done[0]:
            obs = env.reset()
            env.move()
            env.render(frame_id)
            frame_id += 1
            stabilize_height(env, controller)

    env.disconnectUnity()
    env.close() 



if __name__ == '__main__':
    main()