import tensorflow as tf
import tensorflow.contrib.distributions as ds
import matplotlib.pyplot as plt
import math



# Define a class 
class Gaussian_mixture(object):
    
    def __init__(self, theta, phi, r, initial_cov, num_gaussians):
        """
        The first Gaussian is specified by N(mu_0, Sigma_0), where mu_0 = [rcos(theta), rsin(theta)].
        Each we rotate it for phi to get a distribution. num_gaussians is the number of rotations.
        """
        self.theta = theta
        self.phi = phi
        self.r = r
        self.scale_0 = tf.cholesky(initial_cov)
        self.num_gaussians = num_gaussians
        self.mixture_model = self.get_mixture_model()
    
    def get_mixture_model(self):
        """
        ds.Mixture in TensorFlow requires a Categorical dist. to determine which individual dist. is 
        used for generating a sample, 'components' is a list of different classes defined from 
        tf.contrib.distributions
        """
        prob = 1. / self.num_gaussians
        probs = [prob for i in range(self.num_gaussians)]
        
        mus = self.get_mus()
        scales = self.get_scale_matrices()
        gaussians = [ds.MultivariateNormalTriL(loc=mus[i], scale_tril=scales[i]) for i in range(self.num_gaussians)]
        
        mixture = ds.Mixture(cat=ds.Categorical(probs=probs), components=gaussians)
        
        return mixture
    
    def _get_scale(self, phi, B):
        A = [[tf.cos(phi), -tf.sin(phi)],
         [tf.sin(phi), tf.cos(phi)]]
        AB = tf.matmul(A, B)
        cov = tf.matmul(AB, tf.transpose(AB))
        scale = tf.cholesky(cov)
        return scale
    
    def get_mus(self):
        mus = []
        for i in range(self.num_gaussians):
            mu = [self.r * tf.cos(self.theta + i * self.phi), self.r * tf.sin(self.theta + i * self.phi)]
            mus.append(mu)
        return mus
    
    def get_scale_matrices(self):
        scale_matrices = []
        for i in range(self.num_gaussians):
            scale_matrices.append(self._get_scale(i * self.phi, self.scale_0))
        return scale_matrices
    
    def get_samples(self, number):
        samples = self.mixture_model.sample(number)
        return samples
    
    def plot(self, num_samples, color):
        samples = self.get_samples(num_samples)
        figure = plt.scatter(samples[:, 0], samples[:, 1], s=1., color=color)
        return figure