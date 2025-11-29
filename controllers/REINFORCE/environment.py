import sys
import numpy as np
from controller import Supervisor

class Environment(Supervisor):
    """The robot's environment in Webots."""
    
    def __init__(self):
        super().__init__()
                
        # General environment parameters
        self.max_speed = 1.5 # Maximum Angular speed in rad/s
        self.destination_coordinate = np.array([2.5, 2.5]) # Target (Goal) position
        self.reach_threshold = 0.06 # Distance threshold for considering the destination reached.
        obstacle_threshold = 0.1 # Threshold for considering proximity to obstacles.
        self.obstacle_threshold = 1 - obstacle_threshold
        self.floor_size = np.linalg.norm([8, 8])
        
        
        # Activate Devices
        #~~ 1) Wheel Sensors
        self.left_motor = self.getDevice('left wheel')
        self.right_motor = self.getDevice('right wheel')

        # Set the motors to rotate for ever
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Zero out starting velocity
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        #~~ 2) GPS Sensor
        sampling_period = 1 # in ms
        self.gps = self.getDevice("gps")
        self.gps.enable(sampling_period)
        
        #~~ 3) Enable Touch Sensor
        self.touch = self.getDevice("touch sensor")
        self.touch.enable(sampling_period)
              
        # List of all available sensors
        available_devices = list(self.devices.keys())
        # Filter sensors name that contain 'so'
        filtered_list = [item for item in available_devices if 'so' in item and any(char.isdigit() for char in item)]
        filtered_list = sorted(filtered_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        self.step_simulation(200) # take some dummy steps in environment for initialization
        
        # Create dictionary from all available distance sensors and keep min and max of from total values
        self.max_sensor = 0
        self.min_sensor = 0
        self.dist_sensors = {}
        for i in filtered_list:    
            self.dist_sensors[i] = self.getDevice(i)
            self.dist_sensors[i].enable(sampling_period)
            self.max_sensor = max(self.dist_sensors[i].max_value, self.max_sensor)    
            self.min_sensor = min(self.dist_sensors[i].min_value, self.min_sensor)
           
    def step_simulation(self, ms):
        self.step(ms)
            
    def normalizer(self, value, min_value, max_value):
        """
        Performs min-max normalization on the given value.

        Returns:
        - float: Normalized value.
        """
        normalized_value = (value - min_value) / (max_value - min_value)        
        return normalized_value
        

    def get_distance_to_goal(self):
        """
        Calculates and returns the normalized distance from the robot's current position to the goal.
        
        Returns:
        - numpy.ndarray: Normalized distance vector.
        """
        
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        distance_to_goal = np.linalg.norm(self.destination_coordinate - current_coordinate)
        normalizied_coordinate_vector = self.normalizer(distance_to_goal, min_value=0, max_value=self.floor_size)
        
        return normalizied_coordinate_vector
        
    
    def get_sensor_data(self):
        """
        Retrieves and normalizes data from distance sensors.
        
        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """
        
        # Gather values of distance sensors.
        sensor_data = []
        for z in self.dist_sensors:
            sensor_data.append(self.dist_sensors[z].value)  
            
        sensor_data = np.array(sensor_data)
        normalized_sensor_data = self.normalizer(sensor_data, self.min_sensor, self.max_sensor)
        
        return normalized_sensor_data
        
    
    def get_observations(self):
        """
        Obtains and returns the normalized sensor data and current distance to the goal.
        
        Returns:
        - numpy.ndarray: State vector representing distance to goal and distance sensors value.
        """
        
        normalized_sensor_data = np.array(self.get_sensor_data(), dtype=np.float32)
        normalizied_current_coordinate = np.array([self.get_distance_to_goal()], dtype=np.float32)
        
        state_vector = np.concatenate([normalizied_current_coordinate, normalized_sensor_data], dtype=np.float32)
        
        return state_vector
    
    
    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observations.
        
        Returns:
        - numpy.ndarray: Initial state vector.
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        self.prev_distance_to_goal = self.get_distance_to_goal()
        return self.get_observations()


    def step_env(self, action, max_steps):    
        """
        Takes a step in the environment based on the given action.
        
        Returns:
        - state       = float numpy.ndarray with shape of (3,)
        - step_reward = float
        - done        = bool
        """
        
        self.apply_action(action)
        step_reward, done = self.get_reward()
        
        state = self.get_observations() # New state
        
        # Time-based termination condition
        if (int(self.getTime()) + 1) % max_steps == 0:
            done = True
                
        self.prev_distance_to_goal = self.get_distance_to_goal()
        
        return state, step_reward, done
        

    def get_reward(self):
        """
        Calculates and returns the reward based on the current state.
        
        Returns:
        - The reward and done flag.
        """
        
        done = False
        reward = 0
        
        normalized_sensor_data = self.get_sensor_data()
        current_distance = self.get_distance_to_goal()
        
        # Reward based on distance improvement (Shaping Reward)
        # If we get closer, reward is positive. If we move away, reward is negative.
        distance_improvement = self.prev_distance_to_goal - current_distance
        reward += distance_improvement * 50 # Scale up the improvement
        
        # Time penalty to encourage speed
        reward -= 0.05

        # (2) Reward or punishment based on failure or completion of task
        check_collision = self.touch.value
        # reach_threshold is 0.06 normalized, so check directly
        if current_distance < self.reach_threshold:
            # Reward for finishing the task
            done = True
            reward += 25
            print('+++ SOlVED +++')
        elif check_collision:
            # Punish if Collision
            done = True
            reward -= 5
            
        # (3) Punish if close to obstacles
        elif np.any(normalized_sensor_data[normalized_sensor_data > self.obstacle_threshold]):
            reward -= 0.01

        return reward, done


    def apply_action(self, action):
        """
        Applies the specified action to the robot's motors.
        
        Returns:
        - None
        """
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        if action == 0: # move forward
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        elif action == 1: # turn right
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(-self.max_speed)
        elif action == 2: # turn left
            self.left_motor.setVelocity(-self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        
        self.step_simulation(500)
        
        self.left_motor.setPosition(0)
        self.right_motor.setPosition(0)
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)           
