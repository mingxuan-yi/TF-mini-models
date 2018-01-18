import tensorflow as tf
import numpy as np
uniform = tf.contrib.distributions.Uniform

class metropolis_hastings_sampler(object):
  
    def __init__(self, proposal_fn, initial_sample, log_density):
        self.proposal_fn = proposal_fn
        self.current_state = initial_sample
        self.log_density = log_density
        self.markov_chain = [initial_sample]
        self.dtype = initial_sample.dtype
        
        
    def _iterate(self, i):
        theta = self.current_state
        num_chains = self.current_state.get_shape()[0]
        
        theta_star = self.proposal_fn(theta)
            
        # Log density ignoring the constant
        log_hastings_ratio = log_density(theta_star) - log_density(theta)
            
        # acceptance_rate = min[1, density(theta_star) / density(theta)]
        log_acceptance_rate = tf.where(tf.greater(log_hastings_ratio, 0.0),
                                           tf.zeros_like(log_hastings_ratio, dtype=self.dtype),
                                           log_hastings_ratio)
            
        uniform_sampler = uniform(low=tf.constant(0.0, dtype=self.dtype), high=tf.constant(1.0, dtype=self.dtype))
        log_u = tf.log(uniform_sampler.sample(num_chains))
            
        # Accept new sample if acceptance_rate > u
        theta = tf.where(tf.greater(log_acceptance_rate - log_u, 0.0), theta_star, theta)
        assign_op = tf.assign(self.current_state, theta)
        
        with tf.control_dependencies([assign_op]):
             return i + 1
    
    def run(self, num_iter=1):
        i = tf.constant(0)
        stop_cond = lambda i: tf.less(i, num_iter)
        body = self._iterate
        # while_loop is built in Tensorflow, which is much faster than naive python loop methods.
        r = tf.while_loop(stop_cond, body, [i])
        return r
