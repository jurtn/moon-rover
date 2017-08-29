import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_min_thresh=(160, 160, 160), rgb_max_thresh=(255,255,255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    within_range = (img[:,:,0] > rgb_min_thresh[0]) \
                & (img[:,:,1]  > rgb_min_thresh[1]) \
                & (img[:,:,2]  > rgb_min_thresh[2]) \
                & (img[:,:,0] <= rgb_max_thresh[0]) \
                & (img[:,:,1] <= rgb_max_thresh[1]) \
                & (img[:,:,2] <= rgb_max_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[within_range] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    return warped


def rotate_img(image, angle):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D( (w/2, h/2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img

    # 1) Define source and destination points for perspective transform

    # Resolution in pixels for one grid square (1 square meter)
    res = 10
    # Bottom offset: bottom of image is not position of the rover;
    # make a guess to correct it
    bottom_offset = 6
    # Image width and height
    w, h = Rover.img.shape[1], Rover.img.shape[0]
    # Source calibration points
    src = np.array([
            [14, 140], 
            [301 ,140],
            [200, 96], 
            [118, 96]
        ], dtype=np.float32)
    # Destination calibration points
    dst = np.array([
            [w/2 - res/2, h - bottom_offset],
            [w/2 + res/2, h - bottom_offset],
            [w/2 + res/2, h - res - bottom_offset],
            [w/2 - res/2, h - res - bottom_offset]
        ], dtype=np.float32)

    # 2) Apply perspective transform
    # Rotate first the input image to correct for the roll of the roboter
    #roll_corrected = rotate_img(Rover.img, Rover.roll)
    warped = perspect_transform(Rover.img, src, dst)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh(warped)
    obstacles = -1 * navigable + 1
    samples = color_thresh(warped, rgb_min_thresh=(140,110,0), rgb_max_thresh=(240,200,40))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    # It makes more sense to display obstacles in red and navigable in blue, 
    # to be consistent with the worldmap at the bottom right
    Rover.vision_image[:,:,0] = obstacles * 255
    Rover.vision_image[:,:,1] = samples * 255
    Rover.vision_image[:,:,2] = navigable * 255

    # 5) Convert map image pixel values to rover-centric coords
    xpix_navigable, ypix_navigable = rover_coords(navigable)
    xpix_obstacles, ypix_obstacles = rover_coords(obstacles)
    xpix_samples, ypix_samples = rover_coords(samples)

    # 6) Convert rover-centric pixel values to world coordinates
    navigable_x_world, navigable_y_world = pix_to_world(
        xpix_navigable, ypix_navigable,
        Rover.pos[0], Rover.pos[1],
        Rover.yaw,
        Rover.worldmap.shape[0],
        res
    )
    obstacles_x_world, obstacles_y_world = pix_to_world(
        xpix_obstacles, ypix_obstacles,
        Rover.pos[0], Rover.pos[1],
        Rover.yaw,
        Rover.worldmap.shape[0],
        res
    )
    samples_x_world, samples_y_world = pix_to_world(
        xpix_samples, ypix_samples,
        Rover.pos[0], Rover.pos[1],
        Rover.yaw,
        Rover.worldmap.shape[0],
        res
    )

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # Handle non-zero roll values.
    # Our calculation of worldmap relies on zero roll values. We don't want to
    # make any updates to the worldmap with non-zero roll values.
    if (Rover.roll < 0.5 or Rover.roll > 359.5) and (Rover.pitch < 1. or Rover.pitch > 359.):
        Rover.worldmap[obstacles_y_world, obstacles_x_world, 0] = 255
        Rover.worldmap[samples_y_world, samples_x_world, 1] = 255
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix_navigable, ypix_navigable)
    Rover.sample_dists, Rover.sample_angles = to_polar_coords(xpix_samples, ypix_samples)
    
    return Rover