import numpy as np

class GeometricVelocityController:
    """
    Geometric velocity controller for flightmare's body-rate control mode.

    Chain: velocity_cmd → acceleration → thrust + attitude → body rates
    Output: [collective_thrust, omega_x, omega_y, omega_z] normalized to [-1, 1]

    Flightmare's VisionEnv internally denormalizes via:
        pi_act = action * act_std + act_mean
    where act_mean = [max_acc/2, 0, 0, 0], act_std = [max_acc/2, omega_max...]

    Then flightmare's internal LowLevelControllerSimple converts body rates → motor thrusts.

    Requires full quad state from env.getQuadState() — a 25D vector:
      [0]     = id
      [1:4]   = position (px, py, pz)
      [4:8]   = quaternion (qw, qx, qy, qz)
      [8:11]  = velocity (vx, vy, vz) in world frame
      [11:14] = angular velocity (wx, wy, wz) in body frame

    Usage:
        controller = GeometricVelocityController()
        action = controller.compute_action(vel_cmd, quad_state)
    """

    def __init__(self,
                 mass=0.752,
                 motor_omega_max=2800.0,
                 thrust_coeff=1.562522e-6,
                 omega_max=np.array([6.0, 6.0, 2.0])):

        self.mass = mass
        self.g = np.array([0.0, 0.0, -9.81])

        # --- Outer loop gains (geometric controller, from geo.yaml) ---
        self.kd_acc = np.array([3.0, 3.0, 5.0])     # velocity D gains
        self.v_err_max = np.array([1.0, 5.0, 5.0])   # velocity error clamp

        # --- Attitude gains ---
        self.kp_att_xy = 7.0    # roll/pitch attitude gain (10.0 causes wobble at ~50 Hz vs evfly's 300 Hz)
        self.kp_att_z = 2.0     # yaw attitude gain

        # --- Normalization constants (must match flightmare's VisionEnv) ---
        max_force = 4.0 * thrust_coeff * motor_omega_max ** 2
        max_acc = max_force / mass  # max mass-normalized thrust (m/s²)
        self.act_mean = np.array([max_acc / 2.0, 0.0, 0.0, 0.0])
        self.act_std = np.array([max_acc / 2.0, omega_max[0], omega_max[1], omega_max[2]])

    @staticmethod
    def _quat_multiply(q1, q2):
        """Hamilton quaternion product q1 * q2. Format: [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _quat_inverse(q):
        """Inverse of unit quaternion [w, x, y, z]."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_to_rotation(q):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])

    def _tilt_prioritized_rates(self, q_current, q_desired):
        """
        Tilt-prioritized attitude control (Fohn 2020).
        Computes body rate command from quaternion error.

        Args:
            q_current:  [w, x, y, z] current attitude
            q_desired:  [w, x, y, z] desired attitude

        Returns:
            omega_cmd: [wx, wy, wz] body rate command
        """
        q_e = self._quat_multiply(self._quat_inverse(q_current), q_desired)

        # Attitude gain matrix
        T_att = np.diag([self.kp_att_xy, self.kp_att_xy, self.kp_att_z])

        # Quaternion error → body rate mapping
        tmp = np.array([
            q_e[0]*q_e[1] - q_e[2]*q_e[3],
            q_e[0]*q_e[2] + q_e[1]*q_e[3],
            q_e[3],
        ])

        if q_e[0] <= 0:
            tmp[2] *= -1.0

        denom = np.sqrt(q_e[0]**2 + q_e[3]**2)
        if denom < 1e-6:
            denom = 1e-6

        rate_cmd = (2.0 / denom) * T_att @ tmp
        return rate_cmd

    def compute_action(self, vel_cmd, quad_state):
        """
        Geometric controller: velocity command → [thrust, omega_x, omega_y, omega_z].

        Flightmare's VisionEnv uses body-rate control mode (THRUSTRATE).
        We output [collective_thrust, roll_rate, pitch_rate, yaw_rate]
        normalized to [-1, 1]. Flightmare's internal low-level controller
        then converts body rates to motor thrusts.

        Args:
            vel_cmd:      desired velocity [vx, vy, vz] in world frame
            quad_state:   25D array from env.getQuadState()

        Returns:
            action: np.array of shape (4,) normalized to [-1, 1]
                    [collective_thrust, omega_x, omega_y, omega_z]
        """
        # --- Parse quad state ---
        q = quad_state[4:8]        # [qw, qx, qy, qz]
        vel = quad_state[8:11]     # world frame velocity

        # ============================================================
        # Stage 1: Velocity error → Acceleration command
        # ============================================================
        vel_error = np.clip(vel_cmd - vel, -self.v_err_max, self.v_err_max)
        acc_cmd = self.kd_acc * vel_error - self.g  # -g = [0,0,+9.81]

        # ============================================================
        # Stage 2: Acceleration → Thrust + Desired attitude
        # ============================================================
        # Mass-normalized collective thrust (m/s²)
        collective_thrust = np.linalg.norm(acc_cmd)

        # Desired body z-axis = direction of acceleration
        if collective_thrust > 1e-6:
            z_B_des = acc_cmd / collective_thrust
        else:
            z_B_des = np.array([0.0, 0.0, 1.0])

        # Desired yaw rotation (maintain current yaw)
        yaw_current = np.arctan2(2.0*(q[0]*q[3] + q[1]*q[2]),
                                  1.0 - 2.0*(q[2]**2 + q[3]**2))
        cy, sy = np.cos(yaw_current), np.sin(yaw_current)
        y_C = np.array([-sy, cy, 0.0])

        # Build desired rotation matrix
        x_B_des = np.cross(y_C, z_B_des)
        x_B_norm = np.linalg.norm(x_B_des)
        if x_B_norm < 1e-6:
            x_B_des = np.array([1.0, 0.0, 0.0])
        else:
            x_B_des = x_B_des / x_B_norm
        y_B_des = np.cross(z_B_des, x_B_des)
        y_B_des = y_B_des / np.linalg.norm(y_B_des)

        R_des = np.column_stack([x_B_des, y_B_des, z_B_des])

        # Convert desired rotation to quaternion
        tr = np.trace(R_des)
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            q_des = np.array([
                0.25 / s,
                (R_des[2, 1] - R_des[1, 2]) * s,
                (R_des[0, 2] - R_des[2, 0]) * s,
                (R_des[1, 0] - R_des[0, 1]) * s,
            ])
        else:
            if R_des[0, 0] > R_des[1, 1] and R_des[0, 0] > R_des[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R_des[0, 0] - R_des[1, 1] - R_des[2, 2])
                q_des = np.array([
                    (R_des[2, 1] - R_des[1, 2]) / s,
                    0.25 * s,
                    (R_des[0, 1] + R_des[1, 0]) / s,
                    (R_des[0, 2] + R_des[2, 0]) / s,
                ])
            elif R_des[1, 1] > R_des[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R_des[1, 1] - R_des[0, 0] - R_des[2, 2])
                q_des = np.array([
                    (R_des[0, 2] - R_des[2, 0]) / s,
                    (R_des[0, 1] + R_des[1, 0]) / s,
                    0.25 * s,
                    (R_des[1, 2] + R_des[2, 1]) / s,
                ])
            else:
                s = 2.0 * np.sqrt(1.0 + R_des[2, 2] - R_des[0, 0] - R_des[1, 1])
                q_des = np.array([
                    (R_des[1, 0] - R_des[0, 1]) / s,
                    (R_des[0, 2] + R_des[2, 0]) / s,
                    (R_des[1, 2] + R_des[2, 1]) / s,
                    0.25 * s,
                ])
        q_des = q_des / np.linalg.norm(q_des)

        # ============================================================
        # Stage 3: Attitude error → Body rate command
        # ============================================================
        omega_cmd = self._tilt_prioritized_rates(q, q_des)

        # ============================================================
        # Output: Normalize [thrust, omega] to [-1, 1] for flightmare
        # ============================================================
        # Flightmare denormalizes: pi_act = action * act_std + act_mean
        # So: action = (pi_act - act_mean) / act_std
        raw_output = np.array([collective_thrust, omega_cmd[0], omega_cmd[1], omega_cmd[2]])
        action = (raw_output - self.act_mean) / self.act_std
        action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float64)
