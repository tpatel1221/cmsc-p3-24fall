""" Particle filtering """

import random
import numpy as np
import bisect
import copy
from utils import add_noise as utils_add_noise

class Particle:
    """
    Represents a particle for particle filtering
    Each particle has a position, orientation, and weight
    """
    def __init__(self, pos, orient, weight=1.0):
        """
        Initializes a particle
        pos (numpy array of shape (2,)): position of the particle
        orient (numpy array of shape (2,)): a unit vector representing the orientation, pointing in the direction the particle is heading
        weight (float): weight of the particle
        """
        self.pos = pos
        self.orient = orient
        self.weight = weight
    
    def add_noise(self, std_pos=1.0, std_orient=1.0):
        """
        Adds noise to pos and orient
            this is useful when sampling from a distribution with mean at
            the given pos and orient
        std_pos: standard deviation for noise in position
        std_orient: standard deviation for noise in orientation

        Note: orient must have unit norm
        """
        self.pos[0] = utils_add_noise(x=self.pos[0], std=std_pos)
        self.pos[1] = utils_add_noise(x=self.pos[1], std=std_pos)
        while True:
            self.orient[0] = utils_add_noise(x=self.orient[0], std=std_orient)
            self.orient[1] = utils_add_noise(x=self.orient[1], std=std_orient)
            if np.linalg.norm(self.orient) >= 1e-8:
                break
        self.orient = self.orient / np.linalg.norm(self.orient)

class ParticleFilter:
    """
    Particle filter for estimating position and orientation (pose) in a rectangular map, from sensor readings
    """

    def __init__(self, num_particles, minx, maxx, miny, maxy):
        """
        Initialize the particle filter
        num_particles: number of particles for this particle filter
        minx: lower bound on x-coordinate of position
        maxx: upper bound on x coordinate of position
        miny: lower bound on y coordinate of position
        maxy: uppoer bound on y coordinate of position
        """
        self.num_particles = num_particles
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.particles = self.initialize_particles()
        
    def initialize_particles(self):
        """
        Initialize the particles uniformly randomly within the bounds of the rectangular region
        returns a list of Particle objects.
        """
        particles = []

        # BEGIN_YOUR_CODE ######################################################
        for _ in range(self.num_particles):
            x = np.random.uniform(self.minx, self.maxx)
            y = np.random.uniform(self.miny, self.maxy)
            angle = np.random.uniform(0, 2 * np.pi)
            orient = np.array([np.cos(angle), np.sin(angle)])
            particle = Particle(np.array([x, y]), orient)
            particles.append(particle)
        # END_YOUR_CODE ########################################################

        return particles

    
    def filtering_and_estimation(self, sensor, max_sensor_range, sensor_std, evidence, delta_angle, speed):
        """
        Performs particle filtering and estimation of position and orientation
        sensor: function that returns the sensor readings for an arbitrary pose in the map (up,down,left,right), i.e. read_distances in racetrack.py
        sensor_std: std of car's sensor noise
        evidence: sensor readings from car, with the same form as outputs from sensor argument. numpy array of shape (4,)
        delta_angle: clockwise rotation of the car from the previous timestep, in radians
        speed: current speed of the car (distance traveled over 1 unit of time)
        returns x_est (estimated x-component of position), y_est (estimated y-component of position), orient_est (estimated orientation)
        """

        # run filtering step to update particles
        self.particles = self.filtering(sensor, max_sensor_range, sensor_std, evidence, delta_angle, speed)

        # fix the particles in case some are outside the bounds of the region
        for p in self.particles:
            self.fix_particle(p)

        # compute estimated position, angle
        x_est, y_est, orient_est = estimate_pose(self.particles)

        return x_est, y_est, orient_est
    
    def filtering(self, sensor, max_sensor_range, sensor_std, evidence, delta_angle, speed):
        """
        Performs one step of particle filtering according to particle-filtering pseudocode in AIMA.
        """
        new_particles = []

        # BEGIN_YOUR_CODE ######################################################
        #step 1
        for p in self.particles:
            new_particle = self.transition_sample(p, delta_angle, speed)
            new_particles.append(new_particle)

            new_particle.weight = self.compute_prenorm_weight(new_particle, sensor, max_sensor_range, sensor_std, evidence) #step2
        
        #step 2 part 2
        normalize_weights(new_particles)
        
        #step 3
        self.particles = self.weighted_sample_w_replacement(new_particles)
        # END_YOUR_CODE ########################################################

        return new_particles
    
    def compute_prenorm_weight(self, particle, sensor, max_sensor_range, sensor_std, evidence):
        """
        Computes the pre-normalization weight of a particle given evidence.
        """
        weight = None
        # BEGIN_YOUR_CODE ######################################################
        predicted_readings = sensor(particle.pos)
        #Hint: use the weight_gaussian_kernel method
        weight = 1.0
        for i in range(len(predicted_readings)):  
            weight = weight * weight_gaussian_kernel(predicted_readings[i], evidence[i], sensor_std)
        
        # END_YOUR_CODE ########################################################
        return weight

    def transition_sample(self, particle, delta_angle, speed):
        """
        Samples a next pose for this particle according to the car's transition model.
        """
        new_particle = None
        # BEGIN_YOUR_CODE ###################################################### 
        #Hint: rotate the orientation by delta_angle, and then move in that
        # direction at the given speed over 1 unit of time. You will need to add
        # noise at the end to simulate stochasticity in dynamics
        current = np.arctan2(particle.orient[1], particle.orient[0])
        new_angle = current + delta_angle

        new_orientation = np.array([np.cos(new_angle), np.sin(new_angle)])
        new_position = particle.pos + (speed * new_orientation)

        new_particle = Particle(new_position, new_orientation)
        new_particle.add_noise()
        return new_particle
    
    def fix_particle(self, particle):
        """
        Fixes a particle so that it becomes a valid particle, in case it is invalid.
        i.e. this method clips the position of the particle so that it lies within the bounds of the rectangular region.
            this is useful if you sampled a point randomly and it happend to be just outside the bounds
        particle: the particle to be fixed
        """
        x = particle.pos[0]
        y = particle.pos[1]
        particle.pos[0] = max(min(x,self.maxx),self.minx)
        particle.pos[1] = max(min(y,self.maxy),self.miny)
        return particle
    
    def weighted_sample_w_replacement(self, particles):
        """ Performs weighted sampling with replacement """
        new_particles = []

        distribution = WeightedDistribution(particles=particles)

        for _ in range(len(particles)):
            particle = distribution.random_select()
            if particle is None:
                pos = np.array([np.random.uniform(self.minx, self.maxx), np.random.uniform(self.miny, self.maxy)])
                orient = np.array([random.random() - 0.5, random.random() - 0.5])
                orient = orient / np.linalg.norm(orient)
                new_particles.append(Particle(pos, orient))
            else:
                p = Particle(copy.deepcopy(particle.pos), copy.deepcopy(particle.orient))
                new_particles.append(p)
        
        return new_particles

def weight_gaussian_kernel(x1, x2, std = 500):
    """
    Returns the gaussian kernel of the distance between vectors x1 and x2
    std: controls the shape of the gaussian, i.e. controls how much you penalize
    very distant vectors compared with very close vectors
        try plotting exp(-(x^2) / (2*std) using WolframAlpha for different values
        of std to see how this works
    NOTE: std is NOT the same as the std of the car's sensor noise
    """
    distance = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
    return np.exp(-distance ** 2 / (2 * std))

def normalize_weights(particles):
    """
    Normalizes the weights of all the particles, so sum of weights is 1
    """
    weight_total = 0
    for p in particles:
        weight_total += p.weight

    if weight_total == 0:
        weight_total = 1e-8
    
    for p in particles:
        p.weight /= weight_total

class WeightedDistribution(object):

    def __init__(self, particles):
        
        accum = 0.0
        self.particles = particles
        self.distribution = list()
        for particle in self.particles:
            accum += particle.weight
            self.distribution.append(accum)

    def random_select(self):

        try:
            return self.particles[bisect.bisect_left(self.distribution, np.random.uniform(0, 1))]
        except IndexError:
            # When all particles have weights zero
            return None

def estimate_pose(particles):
    """ Estimates the position and orientation based on the given set of particles """
    pos_accum = np.array([0,0])
    orient_accum = np.array([0,0])
    weight_accum = 0.0
    for p in particles:
        weight_accum += p.weight
        pos_accum = pos_accum + p.pos * p.weight
        orient_accum = orient_accum + p.orient * p.weight
    if weight_accum != 0:
        x_est = pos_accum[0] / weight_accum
        y_est = pos_accum[1] / weight_accum
        orient_est = orient_accum / weight_accum
        return x_est, y_est, orient_est
    else:
        raise ValueError
