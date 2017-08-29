# Rover class
# Contains the state of the rover and a good amount of
# the logic implemented in the instance methods

import matplotlib.image as mpimg
import numpy as np

# Adapted from the former RoverState class

class MoonRover():
    def __init__(self):
        self.start_time = None # To record the start time of navigation
        self.total_time = None # To record total duration of naviagation
        self.img = None # Current camera image
        self.pos = None # Current position (x, y)
        self.yaw = None # Current yaw angle
        self.pitch = None # Current pitch angle
        self.roll = None # Current roll angle
        self.vel = None # Current velocity
        self.steer = 0 # Current steering angle
        self.throttle = 0 # Current throttle value
        self.brake = 0 # Current brake value
        self.nav_angles = None # Angles of navigable terrain pixels
        self.nav_dists = None # Distances of navigable terrain pixels

        # Angles and distances of sample pixels
        self.sample_distances, self.sample_angles = None, None

        # Read in ground truth map and create 3-channel green version for overplotting
        # NOTE: images are read in by default with the origin (0, 0) in the upper left
        # and y-axis increasing downward.
        ground_truth = mpimg.imread('../calibration_images/map_bw.png')
        # This next line creates arrays of zeros in the red and blue channels
        # and puts the map into the green channel.  This is why the underlying 
        # map output looks green in the display image
        ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float)
        self.ground_truth = ground_truth_3d # Ground truth worldmap

        # Current mode
        # Can be one of (forward, stop)
        self.mode = 'forward'

        self.throttle_set = 0.2 # Throttle setting when accelerating
        self.brake_set = 10 # Brake setting when braking
        # The stop_forward and go_forward fields below represent total count
        # of navigable terrain pixels.  This is a very crude form of knowing
        # when you can keep going and when you should stop.  Feel free to
        # get creative in adding new fields or modifying these!
        self.stop_forward = 50 # Threshold to initiate stopping
        self.go_forward = 500 # Threshold to go forward again
        self.max_vel = 2 # Maximum velocity (meters/second)
        # Image output from perception step
        # Update this image to display your intermediate analysis steps
        # on screen in autonomous mode
        self.vision_image = np.zeros((160, 320, 3), dtype=np.float) 
        # Worldmap
        # Update this image with the positions of navigable terrain
        # obstacles and rock samples
        self.worldmap = np.zeros((200, 200, 3), dtype=np.float) 
        self.samples_pos = None # To store the actual sample positions
        self.samples_to_find = 0 # To store the initial count of samples
        self.samples_located = 0 # To store number of samples located on map
        self.samples_collected = 0 # To count the number of samples collected
        self.near_sample = 0 # Will be set to telemetry value data["near_sample"]
        self.picking_up = 0 # Will be set to telemetry value data["picking_up"]
        self.send_pickup = False # Set to True to trigger rock pickup

        self.approaching_sample = None

    def update_steer(self):
        # We compute the percentiles instead of the mean.
        # This makes the rover driving slightly on the left side
        # (it is well-known that on the moon you should drive on the left side of the street)
        # More seriously, the rover is turning to the right in a dead end, so this 
        # makes the stearing less than if we would drive on the right side.
        # Otherwise no difference between driving left or right side.
        self.steer = np.clip(np.percentile(self.nav_angles * 180/np.pi, 70), -15, 15)

    def approach_sample(self):

        if self.picking_up:
            return

        # Check if we are close to the sample and not moving
        if self.near_sample and self.vel == 0:
            self.send_pickup = True
            self.approaching_sample = None
            return

        if self.near_sample and self.vel > 0:
            self.mode = 'stop'
            self.brake = 1
            self.throttle = 0
            self.approaching_sample = 'prepare to pickup'
            return

        # Compute the sample angle
        if (len(self.sample_angles) > 0):
            self.last_sample_angles = self.sample_angles
        mean_angle = np.mean(self.last_sample_angles*180/np.pi)
        print(mean_angle)


        if len(self.sample_angles) == 0:
            self.mode = 'forward'
            self.throttle = 0.2
            self.brake = 0
            self.steer = np.clip(mean_angle, -15, 15)
        elif self.approaching_sample != 'approaching' and (self.mode == 'forward' or (self.mode == 'stop' and self.vel > 0)):
            # stop the rover
            self.approaching_sample = 'stopping'
            self.throttle = 0
            self.brake = self.brake_set
            self.mode = 'stop'

            print(self.mode, self.approaching_sample, self.vel)
            return

        # In a second step, turn the rover towards the sample,
        # set steering to 0, and go ahead
        if self.mode == 'stop' and self.vel < 0.1:
            if (mean_angle < 0.1 and mean_angle > -0.1):
                self.approaching_sample = 'approaching'
                self.mode = 'forward'
                self.throttle = 0.2
                self.steer = 0
                return
            self.approaching_sample = 'turning'
            self.throttle = 0
            self.brake = 0
            self.steer = np.clip(mean_angle / 4, -5, 5)
            print('Mode:',self.mode, 'Approaching mode:', self.approaching_sample, 'Vel:', self.vel, 'Steer:', self.steer)

        






