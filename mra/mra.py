from logging import exception
import numpy as np
import tensorflow as tf
from collections.abc import Iterable

class mra:

    """
    Initialize an instance for split learning training and model recovery attack.
    Note that we do not take in any hyperparameters here in the initializer,
    as taking them in the corresponding train/attack functions will make it easier
    for tuning.
    """
    def __init__(self, train_ds) -> None:
        self.train_ds = train_ds
        self.input_shape = train_ds.element_spec[0].shape
        # The following code is very inefficient
        self.train_size = len(list(train_ds))

    """
    Train the split learning networks as the protocol defines.
    Note that for an honest-but-curious server-side attacks, the server does not
    tamper with the training process in any way.
    """
    def train(self, make_f, make_g, loss_fn, batch_size=32, epoch=1, lr=0.001, alpha1=0.0, alpha2=1.0, verbose=True, log_every=10):
        self.batch_size = batch_size
        self.iterations = epoch * self.train_size // batch_size
        train_batches = self.train_ds.batch(batch_size=batch_size, drop_remainder=True).repeat(-1).take(self.iterations)

        self.f = make_f(self.input_shape)
        self.int_shape = self.f.layers[-1].output_shape[1:]
        self.g = make_g(self.int_shape)

        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # log all the input data and intermediate representations for futher reference when attacking
        self.x_ref = []
        self.z_ref = []
        
        # log the losses of each batch while training
        log = []
        iter_count = 1
        sum_loss = 0.0

        for (x_batch, y_batch) in train_batches:
            with tf.GradientTape(persistent=True) as tape:
                z = self.f(x_batch, training=True)
                y_pred = self.g(z, training = True)
                loss = loss_fn(y_true=y_batch, y_pred=y_pred)+1e-10
                if alpha1 != 0.0:
                    dcor = dist_corr(x_batch, z)
                    loss = alpha1*dcor + loss*alpha2

            var = self.f.trainable_variables + self.g.trainable_variables
            grad = tape.gradient(loss, var)
            opt.apply_gradients(zip(grad, var))

            log.append(loss)
            
            self.x_ref.append(x_batch)
            self.z_ref.append(z)
            sum_loss += loss
            
            if verbose and iter_count % log_every == 0:
                print("Iteration %04d: Training loss: %0.4f" % (iter_count, sum_loss/log_every))
                sum_loss = 0.0
            iter_count += 1
        return np.array(log)

    """
    make_generator: None if we don't use a generator network to attack, otherwise a function
        to initialize a generator model.
    input_noise: only applicable when make_generator is not None, set to True if we feed random
        noise to the generator.
    input_z: only applicable when make_generator is not None, set to True if we feed intermediate
        representations to the generator.
    model_leak: set to True if the surrogate client model is initialized by leaked parameters of
        the last batch. Otherwise, the surrogate client model is initialized randomly.
    lr_x: learning rate to update generator model or to optimize x if generator is not used.
    lr_f: learning rate to update surrogate client model. When model_leak is true, iter_f should
        be small, like 1. Otherwise, iter_f should be large.
    """
    def attack(self, attack_iter, make_generator, input_noise, input_z, model_leak, lr_x, lr_f, epoch, iter_x, iter_f, verbose=True, log_every=10):
        if make_generator is None:
            use_generator = False
        else:
            use_generator = True
            if input_noise and input_z:
                self.generator = make_generator(tuple(map(sum, zip(self.int_shape, self.int_shape))))
            else:
                self.generator = make_generator(self.int_shape)
        
        f_temp = tf.keras.models.clone_model(self.f)
        if model_leak:
            f_temp.set_weights(self.f.get_weights())

        iter_count = 0
        log = []

        if attack_iter > self.iterations:
            attack_iter = self.iterations

        for i in range(attack_iter):
            x_opt = tf.keras.optimizers.Adam(learning_rate=lr_x)
            f_opt = tf.keras.optimizers.Adam(learning_rate=lr_f)
            
            x = self.x_ref[self.iterations - i - 1]
            z = self.z_ref[self.iterations - i - 1]

            if not use_generator:
                # we randomly initialize x_temp to search the best x
                x_temp = np.random.normal(0.5, 0.25, x.shape)
                x_temp = tf.Variable(x_temp)
            
            for _ in range(epoch):

                for _ in range(iter_x):

                    if use_generator:
                        # if we use generator, we update the generator
                        with tf.GradientTape() as tape:
                            if input_noise == "normal" and input_z:
                                r = tf.random.normal(shape=z.shape, mean=tf.math.reduce_mean(z), stddev=tf.math.reduce_std(z))
                                input = tf.concat([z,r],axis=1)
                            elif input_noise == "uniform" and input_z:
                                r = tf.random.uniform(shape=z.shape, minval=tf.math.reduce_min(z), maxval=tf.math.reduce_max(z))
                                input = tf.concat([z,r],axis=1)
                            elif input_noise == "normal" and (not input_z):
                                input = tf.random.normal(shape=z.shape, mean=0.5, stddev=0.25)
                            elif input_noise == "uniform" and (not input_z):
                                input = tf.random.uniform(shape=z.shape, minval=0.0, maxval=1.0)
                            elif (input_noise == False) and input_z:
                                input = z
                            else:
                                raise ValueError("Must define the input when generator is present.")
                            
                            x_temp = self.generator(input, training=True)
                            loss_x = tf.keras.losses.MeanSquaredError()(f_temp(x_temp, training=False), z)
                        
                        vars = self.generator.trainable_variables
                        grad = tape.gradient(loss_x, vars)
                        x_opt.apply_gradients(zip(grad, vars))
                    
                    else:
                        # if we do not use generator, we find the minimizing x_temp
                        loss = lambda: tf.keras.losses.MeanSquaredError()(f_temp(x_temp, training=False), z)
                        x_opt.minimize(loss, var_list=[x_temp])

                for _ in range(iter_f):
                    with tf.GradientTape() as tape:
                        loss_f = tf.keras.losses.MeanSquaredError()(f_temp(x_temp, training=True), z)
                    vars = f_temp.trainable_variables
                    grad = tape.gradient(loss_f, vars)
                    f_opt.apply_gradients(zip(grad, vars))

            attack_mse = tf.losses.MeanSquaredError()(x_temp, x)
            rg_uniform = tf.losses.MeanSquaredError()(x, tf.random.uniform(shape=x.shape, minval=0.0, maxval=1.0))
            rg_normal = tf.losses.MeanSquaredError()(x, tf.random.normal(shape=x.shape, mean=0.5, stddev=0.25))
            log.append([rg_uniform, rg_normal, attack_mse])
            iter_count += 1
#             print(tf.keras.losses.MeanSquaredError()(f_temp(x_temp, training=False), z))
            if verbose and (iter_count - 1) % log_every == 0:
                print("Iteration %04d: RG-uniform: %0.4f, RG-normal: %0.4f, reconstruction validation: %0.4f" % (iter_count, rg_uniform, rg_normal, attack_mse))
        
        return np.array(log)

def pairwise_dist(A):
    r = tf.reduce_sum(A*A, 1)
    r = tf.reshape(r, [-1, 1])
    D = tf.maximum(r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r), 1e-7)
    D = tf.sqrt(D+1e-10)
    return D

def dist_corr(X, Y):
    X = tf.keras.layers.Flatten()(X)
    Y = tf.keras.layers.Flatten()(Y)
    n = tf.cast(tf.shape(X)[0], tf.float32)
    a = pairwise_dist(X)
    b = pairwise_dist(Y)
    A = a - tf.reduce_mean(a, axis=1) -\
        tf.expand_dims(tf.reduce_mean(a,axis=0),axis=1)+\
        tf.reduce_mean(a)
    B = b - tf.reduce_mean(b, axis=1) -\
        tf.expand_dims(tf.reduce_mean(b,axis=0),axis=1)+\
        tf.reduce_mean(b)
    dCovXY = tf.sqrt(1e-10+tf.reduce_sum(A*B) / (n ** 2))
    dVarXX = tf.sqrt(1e-10+tf.reduce_sum(A*A) / (n ** 2))
    dVarYY = tf.sqrt(1e-10+tf.reduce_sum(B*B) / (n ** 2))

    dCorXY = dCovXY / tf.sqrt(1e-10 + dVarXX * dVarYY)
    return dCorXY