
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt



class LaneFilterHistogramKF():
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'range_min',
            'range_est',
            'range_max',
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])

<<<<<<< HEAD
        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        
        print("MEAN_D_0", self.mean_d_0)
        print("MEAN_PHI_0", self.mean_phi_0)
        print("SIGMA_D_0", self.sigma_d_0)
        print("SIGMA_PHI_0", self.sigma_phi_0)
=======
>>>>>>> 8d8da7d386a0a62c1a3c2f7c36cf71a611b3e5a7


        self.encoder_resolution = 0
        self.wheel_radius = 0.0
        self.baseline = 0.0
        self.initialized = False
        self.reset()

    def reset(self):
        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}

        #Init matrices
        self.A = np.identity(2)
        self.B = np.identity(2)
        self.Q = np.array([[0.6,0],[0,0.6]])
        self.H = np.identity(2)
        self.R = np.array([[0.15,0],[0,0.05]])
        
        # self.A = np.identity(2)
        # self.B = np.identity(2)
        # self.Q = np.array([[0.3,0],[0,0.3]])
        # self.H = np.identity(2)
        # self.R = np.array([[0.75,0],[0,0.6]])       

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        #TODO update self.belief based on right and left encoder data + kinematics
        if not self.initialized:
            return
        #Compute right and left velocities
        d_left = (2 * np.pi * self.wheel_radius * left_encoder_delta) / self.encoder_resolution
        v_left = d_left / dt
        
        d_right = (2 * np.pi * self.wheel_radius * right_encoder_delta) / self.encoder_resolution
        v_right = d_right / dt

        #Compute angular displacement
        theta_delta = (d_right - d_left) / self.baseline
        theta_dot = (v_right - v_left) / self.baseline
        
        #Compute robot velocities 
        v = 0.5 * (v_right + v_left)
        omega = theta_delta / dt

        #Update belief
        self.belief['mean'][0] = self.belief['mean'][0] + dt*v*np.sin(theta_dot) 
        self.belief['mean'][1] = self.belief['mean'][1] + dt*omega 
        #Update covariance
        self.belief['covariance'] = self.A @ np.array(self.belief['covariance']) @ self.A.T + self.Q
        
        # d_left = (2 * np.pi * self.wheel_radius * left_encoder_delta) / self.encoder_resolution
        # v_left = d_left / dt
        # d_right = (2 * np.pi * self.wheel_radius * right_encoder_delta) / self.encoder_resolution
        # v_right = d_right / dt
        # d = 0.5 * (d_right + d_left)
        # v = 0.5 * (v_right + v_left)
        # theta_delta = (d_right - d_left) / self.baseline   
        # d_t = d*v*np.sin(self.belief['mean'][1] + 0.5*theta_delta)
        # u_t = np.array([d_t, theta_delta])
        # self.belief['mean'] = self.A @ self.belief['mean'] + self.B @ u_t
        # self.belief['covariance'] = self.A @ np.array(self.belief['covariance']) @ self.A.T + self.Q

    def update(self, segments):
        # prepare the segments for each belief array
        segmentsArray = self.prepareSegments(segments)
        # generate all belief arrays
        measurement_likelihood = self.generate_measurement_likelihood(
            segmentsArray)
        
        # TODO: Parameterize the measurement likelihood as a Gaussian
        # TODO: Apply the update equations for the Kalman Filter to self.belief
        if(measurement_likelihood is not None):
            d_new = self.d_min + np.unravel_index(np.argmax(measurement_likelihood, axis=None), measurement_likelihood.shape)[0] * self.delta_d + self.delta_d*0.5 + np.random.normal(loc=0.0, scale=self.R[0,0])
            phi_new = self.phi_min + np.unravel_index(np.argmax(measurement_likelihood, axis=None), measurement_likelihood.shape)[1] * self.delta_phi + self.delta_phi*0.5 + np.random.normal(loc=0.0, scale=self.R[1,1])
            z = np.array([d_new, phi_new])
            mu_residual = z - self.H @ np.array(self.belief['mean'])
        else:
            return
        
        covariance_residual = self.H @ np.array(self.belief['covariance']) @ self.H.T + self.R
        K = np.array(self.belief['covariance']) @ self.H.T @ np.linalg.inv(covariance_residual)
        self.belief['mean'] = np.array(self.belief['mean']) + K @ mu_residual
        self.belief['covariance'] = np.array(self.belief['covariance']) - K @ self.H @ np.array(self.belief['covariance'])


    def getEstimate(self):
        return self.belief

    def generate_measurement_likelihood(self, segments):

        if len(segments) == 0:
            return None

        grid = np.mgrid[self.d_min:self.d_max:self.delta_d,
                                    self.phi_min:self.phi_max:self.delta_phi]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape)

        for segment in segments:
            d_i, phi_i, l_i, weight = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return None

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / \
            np.sum(measurement_likelihood)
        return measurement_likelihood





    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < 3*self.delta_d and abs(phi_s - phi_max) < 3*self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray