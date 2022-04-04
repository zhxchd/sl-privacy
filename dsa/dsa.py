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
    
    def dsa_attack(self, make_f, make_g, loss_fn, acc_fn, lr, batch_size, iterations, make_e, make_d, make_c, lr_e, lr_d, lr_c, iter_d=1, w=None, flatten=False, verbose=True, log_freq=1):
        client_dataset = self.target_ds.batch(batch_size, drop_remainder=True).repeat(-1)
        attacker_dataset = self.aux_ds.repeat(-1).batch(batch_size, drop_remainder=True)
        
        self.f = make_f(self.input_shape)
        self.intermidiate_shape = self.f.layers[-1].output_shape[1:]
        self.g = make_g(input_shape=self.intermidiate_shape)
        self.e = make_e(input_shape=self.input_shape)
        self.loss_fn = loss_fn
        self.acc_fn = acc_fn
        self.flatten = flatten
        
        if self.flatten:
            # note that the input of the decoder is first flattened
            self.flattened_inter_dim = math.prod(self.intermidiate_shape)
            self.d = make_d(input_shape=(self.flattened_inter_dim, ))
        else:
            self.d = make_d(input_shape=self.intermidiate_shape)
        self.c = make_c(self.intermidiate_shape)
        
        self.f_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.e_opt = tf.keras.optimizers.Adam(learning_rate=lr_e)
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=lr_d)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=lr_c)
        
        iterator = zip(client_dataset.take(iterations), attacker_dataset.take(iterations))
        
        iter = 1
        log = []
        acc_loss = 0.0
        acc_acc = 0.0
        
        for (x_private, label_private), (x_public, _) in iterator:

            with tf.GradientTape(persistent=True) as tape:
                z_private = self.f(x_private, training=True)
                y_pred = self.g(z_private, training=True)
                
                loss = self.loss_fn(y_true=label_private, y_pred=y_pred)
                acc = self.acc_fn(y_true=label_private, y_pred=y_pred)

                z_public = self.e(x_public, training=True)
                
                c_private_logits = self.c(z_private, training=True)
                c_public_logits = self.c(z_public, training=True)
                
                if w is not None:
                    e_loss = -tf.reduce_mean(c_public_logits)
                    c_loss = tf.reduce_mean( c_public_logits ) - tf.reduce_mean( c_private_logits )
                    gp = self.get_gradient_penalty(z_private, z_public)
                    c_loss += gp * float(w)
                else:
                    e_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(c_public_logits), c_public_logits)
                    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(c_private_logits), c_private_logits)
                    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(c_public_logits), c_public_logits)
                    c_loss = real_loss + fake_loss

                if self.flatten:
                    flat_z_pub = z_public.numpy().reshape((batch_size, self.flattened_inter_dim))
                    x_temp = self.d(flat_z_pub, training=True)
                else:
                    x_temp = self.d(z_public, training=True)
                d_loss = tf.losses.MeanSquaredError()(x_public, x_temp)
            
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

            # update decoder d:
            var = self.d.trainable_variables
            gradients = tape.gradient(d_loss, var)
            self.d_opt.apply_gradients(zip(gradients, var))

            # Now let's do something with the generative decoder:
            # self.d_opt = tf.keras.optimizers.Adam(learning_rate=lr_d)
            
#             # decoder decode the updated encoder
#             if self.flatten:
#                 flat_z_pub = self.e(x_public, training=False).numpy().reshape((batch_size, self.flattened_inter_dim))
#             else:
#                 z_public = self.e(x_public, training=False)
            # for _ in range(iter_d):
            #     with tf.GradientTape() as tape:
            #         if self.flatten:
            #             flat_z_pub = z_public.numpy().reshape((batch_size, self.flattened_inter_dim))
            #             x_temp = self.d(flat_z_pub, training=True)
            #         else:
            #             x_temp = self.d(z_public, training=True)
            #         d_loss = tf.losses.MeanSquaredError()(x_public, x_temp)
            #         var = self.d.trainable_variables
            #     gradients = tape.gradient(d_loss, var)
            #     self.d_opt.apply_gradients(zip(gradients, var))
    
            # Now we have the generative decoder trained, let's attack original image
            if self.flatten:
                flat_z_priv = z_private.numpy().reshape((batch_size, math.prod(self.intermidiate_shape)))
                rec_x_private = self.d(flat_z_priv, training=False) 
            else:
                rec_x_private = self.d(z_private, training=False)

            loss_verification = tf.losses.MeanSquaredError()(x_private, rec_x_private)
            log.append([loss, acc, loss_verification])
            acc_acc += acc.numpy()
            acc_loss += loss_verification.numpy()

            if verbose and iter % log_freq == 0:
                print("Iteration {}, train accuracy: {}, average attack MSE: {}".format(iter, acc_acc/log_freq, acc_loss/log_freq))
                acc_loss = 0.0
                acc_acc = 0.0
            iter += 1
        
        return np.array(log)
    
    def get_gradient_penalty(self, x, x_gen):
        e_shape = [1 for i in range(len(x.shape))]
        e_shape[0] = x.shape[0]
        epsilon = tf.random.uniform(e_shape, 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.c(x_hat, training=True)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[i + 1 for i in range(len(x.shape)-2)]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer
    
    def attack_examples(self, input_examples):
        if self.flatten:
            flattened_z = self.f(input_examples, training=False).numpy().reshape((len(input_examples), self.flattened_inter_dim))
            rec_res = self.d(flattened_z, training=False)
        else:
            rec_res = self.d(self.f(input_examples, training=False), training=False)
        mse = tf.losses.MeanSquaredError()(input_examples, rec_res)
        return mse, rec_res
