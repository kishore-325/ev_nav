""""
Expert policy for Flightmare - Grid-Based Reactive Collision Avoidance

VisionEnv_v1 observation format:
  obs[0:3]   = goal_velocity (vx, vy, vz)
  obs[3:12]  = rotation_matrix (flattened 3x3, body-to-world)
  obs[12:15] = linear_velocity (vx, vy, vz) in body frame
  obs[15:55] = obstacle_states (10 obstacles × [rel_x, rel_y, rel_z, radius])

Obstacle are in **body frame** (relative to the quadcopter)
"""

import numpy as np

class ExpertPolicy:

    def __init__(self, desired_vel=5.0, max_detection_range=10.0, is_trees=True,x_displacement=8.0, grid_center_offset=8.0, grid_displacement=0.5,
                 obst_dist_threshold=10.0, obst_inflate_factor=1.0):
        
        self.desired_vel = desired_vel
        self.max_detection_range = max_detection_range
        self.is_trees = is_trees
        # Grid parameters
        self.x_displacement = x_displacement             # forward distance of candidate waypoints
        self.grid_center_offset = grid_center_offset     # lateral extent +_ 8m
        self.grid_displacement = grid_displacement       # grid spacing
        self.obst_dist_threshold = obst_dist_threshold   # ignore obstacles beyond this
        self.obst_inflate_factor = obst_inflate_factor   # safety margin added to obstacle raadius

        x_grid = np.arange(grid_center_offset, -grid_center_offset - grid_displacement, -grid_displacement)
        if is_trees :
            y_grid = np.array([0.0])
        else:
            y_grid = np.arrange(grid_center_offset, -grid_center_offset - grid_displacement, -grid_displacement)

        self.num_wpts_x = len(x_grid)
        self.num_wpts_y = len(y_grid)

        self.wpts_2d = np.zeros((self.num_wpts_y, self.num_wpts_x, 3))

        # Build waypoint grid and check collisions
        for xi, x in enumerate(x_grid):
            for yi, y in enumerate(y_grid):
                # Waypoint : x_displacement forward(8), x_lateral, y_vertical
                self.wpts_2d[yi, xi] = [x_displacement, x, y]
        


    def parse_obstacle_from_obs(self,obs, max_detection_range=10.0):
        """
        Extract obstacle list from a single env's observation.

        Args:
            obs: 1D array of shape (55,) — one environment's observation
            max_detection_range: obstacles beyond this are padding (10,10,10)

        Returns:
            list of dicts with keys: rel_x, rel_y, rel_z, radius, distance, detected
        """
        obstacles_data = obs[15:55].reshape(10,4)
        obstacles = []
        for obstacle in obstacles_data:

            rel_pos = obstacle[0:3]
            radius = obstacle[3]
            distance = np.linalg.norm(rel_pos)
            detected = distance < max_detection_range-0.1

            if detected:
                obstacles.append(
                    {
                        'rel_x' : rel_pos[0],
                        'rel_y' : rel_pos[1],
                        'rel_z' : rel_pos[2],
                        'radius' : radius,
                        'distance' : distance,
                    }
                )
    
        return obstacles
    
    def check_collision(self, line, obstacle):
        """
        Sphere-line intersection test.

        Line: two 3D points. Sphere: center + radius.
        Returns True if the line intersects the sphere.
        """
        (x1, y1, z1), (x2, y2, z2) = line
        (x3, y3, z3), r = obstacle

        if self.is_trees:
            z3 = 0.0

        # quadratic formula discriminant
        b = 2.0 * ((x2-x1)*(x1-x3) + (y2-y1)*(y1-y3) + (z2-z1)*(z1-z3))
        a = (x2 - x1)**2 + (y2-y1)**2 + (z2-z1)**2
        c = (
            x3**2 + y3**2 + z3**2
            + x1**2 + y1**2 + z1**2
            - 2.0 * (x3*x1 + y3*y1 + z3*z1)
            - r ** 2
        )

        return b ** 2 - 4.0 * a * c >= 0 
        

    def find_closest_zero_index(self, arr):
        """Find the zero-valued cell closest to the array center."""

        center = np.array(arr.shape) // 2
        dist_to_center = np.abs(
            np.indices(arr.shape) - center.reshape(-1, 1, 1)
        ).sum(0)
        
        zero_indices = np.argwhere(arr == 0)
        if len(zero_indices) == 0:
            return None
        dist_to_zeros = dist_to_center[tuple(zero_indices.T)]
        min_dist_indices = np.argwhere(dist_to_zeros == dist_to_zeros.min()).flatten()
        chosen_index = np.random.choice(min_dist_indices)
        
        return tuple(zero_indices[chosen_index])


    def compute_expert_velocity(self,obs):
        """
        Run collision avoidance on a single env's observation.

        Args:
            obs: observation array (1D shape (55,) or 2D shape (1,55))

        Returns:
            vel_cmd: np.array([vx, vy, vz]) desired velocity in body frame
            extras:  dict with collision grid and chosen waypoint index
        """
        
        
        obstacles = self.parse_obstacle_from_obs(obs, self.max_detection_range)

        collisions = np.zeros((self.num_wpts_y,self.num_wpts_x))

        for xi in range(self.num_wpts_x):
            for yi in range(self.num_wpts_y):
                for obst in obstacles:
                    ox, oy, oz = obst['rel_x'], obst['rel_y'], obst['rel_z']
                    r = obst['radius'] + self.obst_inflate_factor

                    if ox+r < 0 or ox-r > self.obst_dist_threshold:
                        continue
                    
                    if self.check_collision(
                        ((0, 0, 0), tuple(self.wpts_2d[yi, xi])),
                        ((ox,oy,oz), r)
                    ):
                        collisions[yi,xi] = 1
                        break
        
        # select best waypoint
        if collisions.sum() == collisions.size:
            vel_cmd = np.array([self.desired_vel, 0.0, 0.0])
            wpt_idx = None
        
        else:
            wpt_idx = self.find_closest_zero_index(collisions)
            wpt = self.wpts_2d[wpt_idx[0], wpt_idx[1]]
            # Normalizing to unit vector and scaling to desired velocity
            vel_cmd = wpt / np.linalg.norm(wpt) * self.desired_vel      


        extras = {
            'collisions' : collisions,
            'wpt_idx' : wpt_idx,
            'num_obstacles_detected' : len(obstacles),
        }

        return vel_cmd, extras
    

    
        