#!/usr/bin/env python3

#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

# ------------------------
#   IMPORTS
# ------------------------
import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

# Pygame Init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off screen alpha
screen.set_alpha(None)

# Showing sensors and redrawing screens
show_sensors = True
draw_screen = True


# ------------------------
#   GameState Class
# ------------------------
class GameState:
    """
        Constructor for GameState class
    """
    def __init__(self):
        self.crashed = False
        self.car_body = None
        self.car_shape = None
        self.c_body = None
        self.c_shape = None
        # physics stuff
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        # create car
        self.create_car(100, 100, 0.5)
        # record steps
        self.num_steps = 0
        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width - 1, height), (width - 1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles.append(self.create_obstacle(200, 350, 100))
        self.obstacles.append(self.create_obstacle(700, 200, 125))
        self.obstacles.append(self.create_obstacle(600, 600, 35))
        # Create a car obstacle
        self.create_car_obstacle()

    """
        Create Car Method
    """
    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    """
        Create Obstacle Method
    """
    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    """
        Create Obstacle Car Method
    """
    def create_car_obstacle(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.c_body = pymunk.Body(1, inertia)
        self.c_body.position = 50, height - 100
        self.c_shape = pymunk.Circle(self.c_body, 30)
        self.c_shape.color = THECOLORS["orange"]
        self.c_shape.elasticity = 1.0
        self.c_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.c_body.angle)
        self.space.add(self.c_body, self.c_shape)

    """
        Frame Step Method
    """
    def frame_step(self, action):
        if action == 0:  # Turn left.
            self.car_body.angle -= .2
        elif action == 1:  # Turn right.
            self.car_body.angle += .2
        # Move obstacles.
        if self.num_steps % 100 == 0:
            self.move_obstacles()
        # Move vehicle obstacle
        if self.num_steps % 5 == 0:
            self.move_car_obstacle()
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * driving_direction
        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        draw(screen, self.space)
        self.space.step(1. / 10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()
        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sensor_readings(x, y, self.car_body.angle)
        normalized_readings = [(x - 20.0) / 20.0 for x in readings]
        state = np.array([normalized_readings])
        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            reward = -500
            self.recover_from_crash(driving_direction)
        else:
            # Higher readings are better, so return the sum.
            reward = -5 + int(self.sum_readings(readings) / 10)
        self.num_steps += 1
        return reward, state

    """
        Move Obstacles Method
    """
    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction
    """
        Move Car Obstacle Method
    """
    def move_car_obstacle(self):
        speed = random.randint(20, 200)
        self.c_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.c_body.angle)
        self.c_body.velocity = speed * direction

    """
        Check Car Crash Method
    """
    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False

    """
        Recover From Car Crash Method
    """
    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -100 * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["grey7"])
                draw(screen, self.space)
                self.space.step(1. / 10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()

    """
        Read Sums Readings Method
    """
    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    """
        Get Sensor Readings Method
    """
    def get_sensor_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, the sensor readings
        simply return N "distance" readings, one for each sensor
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sensor "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self.make_sensor_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left
        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))
        if show_sensors:
            pygame.display.update()
        return readings

    """
        Get Arm Distance Method
    """
    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0
        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1
            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )
            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i
            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), rotated_p, 2)
        # Return the distance for the arm.
        return i

    """
        Make Sensor Arm Method
    """
    def make_sensor_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    """
        Get Rotated Point Method
    """
    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
                   (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
                   (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    """
        Check Tracking Method
    """
    def get_track_or_not(self, reading):
        if reading == THECOLORS['black']:
            return 0
        else:
            return 1


if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))