import tensorflow as tf
import numpy as np

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  s, a, s_next = x

  mean, logvar = model.encode((s, a, s_next))
  z = model.reparameterize(mean, logvar)

  x_logit = model.decode((s,a,z))
  
  cross_ent_img = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit[0], labels=s_next[0])
  cross_ent_imu = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit[1], labels=s_next[1])
  cross_ent_vec = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit[2], labels=s_next[2])

  logpx_z_img = -tf.reduce_sum(cross_ent_img, axis=[1,2,3])
  logpx_z_imu = -tf.reduce_sum(cross_ent_imu, axis=[1,2])
  logpx_z_vec = -tf.reduce_sum(cross_ent_vec, axis=[1,2])

  logpx_z = logpx_z_img + logpx_z_imu + logpx_z_vec

  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

