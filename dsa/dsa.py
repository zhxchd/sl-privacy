import tensorflow as tf
import numpy as np
import math

class dsa:
    """
    target_ds is the private training examples stored on the clients.
    aux_ds is the public auxiliary dataset stored on the semi-honest server.
    """
    def __init__(self, target_ds, aux_ds) -> None:
        self.target_ds = target_ds
        self.aux_ds = aux_ds
        self.input_shape = target_ds.element_spec[0].shape
    
    def dsa_attack(self, make_f, make_g, lr, batch_size, iterations, make_e, make_d, make_c, lr_e, lr_d, lr_c, iter_d=50, w=500., verbose=True, log_freq=1):
        client_dataset = self.target_ds.batch(batch_size, drop_remainder=True).repeat(-1)
        attacker_dataset = self.aux_ds.batch(batch_size, drop_remainder=True).repeat(-1)
        
        self.f = make_f(self.input_shape)
        self.intermidiate_shape = self.f.layers[-1].output_shape[1:]
        self.g = make_g(input_shape=self.intermidiate_shape)
        self.e = make_e(input_shape=self.input_shape)
        # note that the input of the decoder is first flattened
        self.flattened_inter_dim = math.prod(self.intermidiate_shape)
        self.d = make_d(input_shape=(self.flattened_inter_dim, ))
        self.c = make_c(self.intermidiate_shape)
        
        self.f_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.e_opt = tf.keras.optimizers.Adam(learning_rate=lr_e)
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=lr_d)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=lr_c)
        
        iterator = zip(client_dataset.take(iterations), attacker_dataset.take(iterations))
        
        iter = 1
        log = []
        acc_loss = 0.0
        
        for (x_private, label_private), (x_public, _) in iterator:

            with tf.GradientTape(persistent=True) as tape:
                z_private = self.f(x_private, training=True)
                y_pred = self.g(z_private, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label_private, y_pred=y_pred)

                z_public = self.e(x_public, training=True)
                
                c_privite_logits = self.c(z_private, training=True)
                c_public_logits = self.c(z_public, training=True)
                
                e_loss = -tf.reduce_mean(c_public_logits)
                
                c_loss = tf.reduce_mean( c_public_logits ) - tf.reduce_mean( c_privite_logits )
                
                gp = self.get_gradient_penalty(z_private, z_public)
                c_loss += gp * float(w)
            
            # update f and g:
            var = self.f.trainable_variables + self.g.trainable_variables
            gradients = tape.gradient(loss, var)
            self.f_opt.apply_gradients(zip(gradients, var))
            
            # encoder is updated by e_loss, to simulate
            var = self.e.trainable_variables
            gradients = tape.gradient(e_loss, var)
            self.e_opt.apply_gradients(zip(gradients, var))
            
            # update critic (discriminator) c:
            var = self.c.trainable_variables
            gradients = tape.gradient(c_loss, var)
            self.c_opt.apply_gradients(zip(gradients, var))

            # Now let's do something with the generative decoder:
            flat_z_pub = self.f(x_public, training=False).numpy().reshape((batch_size, self.flattened_inter_dim))
            for _ in range(iter_d):
                with tf.GradientTape() as tape:
                    x_temp = self.d(flat_z_pub)
                    d_loss = tf.losses.MeanSquaredError()(x_public, x_temp)
                    var = self.d.trainable_variables
                gradients = tape.gradient(d_loss, var)
                self.d_opt.apply_gradients(zip(gradients, var))
    
            # Now we have the generative decoder trained, let's attack original image
            flat_z_priv = z_private.numpy().reshape((batch_size, math.prod(self.intermidiate_shape)))
            rec_x_private = self.d(flat_z_priv, training=False)

            loss_verification = tf.losses.MeanSquaredError()(x_private, rec_x_private)
            log.append([sum(loss)/len(loss), loss_verification])

            if verbose and iter % log_freq == 0:
                print("Iteration {}, average attack MSE: {}".format(iter, acc_loss/log_freq))
                acc_loss = 0.0
            else:
                acc_loss += loss_verification.numpy()

            iter += 1
        
        return np.array(log)
    
    def get_gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.c(x_hat, training=True)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer
    
    def attack_examples(self, input_examples):
        flattened_z = self.f(input_examples, training=False).numpy().reshape((len(input_examples), self.flattened_inter_dim))
        rec_res = self.d(flattened_z, training=False)
        mse = tf.losses.MeanSquaredError()(input_examples, rec_res)
        return mse, rec_res