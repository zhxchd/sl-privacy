import tensorflow as tf
import numpy as np
import tqdm
import defense
# import datasets, architectures

def distance_data_loss(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

def distance_data(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

class sl_with_attack:
    
    def loadBiasNetwork(self, make_decoder, z_shape, channels):
        return make_decoder(z_shape, channels=channels)
        
    def __init__(self, models, xpriv, xpub, id_setup, batch_size, hparams, class_num, server_attack=None, sorted=False):
            input_shape = xpriv.element_spec[0].shape
            self.hparams = hparams
            self.server_attack = server_attack
            self.class_num = class_num
            self.sorted = sorted

            # setup dataset
            self.client_dataset = xpriv.batch(batch_size, drop_remainder=True).repeat(-1)
            self.attacker_dataset = xpub.batch(batch_size, drop_remainder=True).repeat(-1)
            
            self.batch_size = batch_size

            ## setup models
            make_f, make_g, make_tilde_f, make_decoder, make_D = models

            # f and g are always present, even with an honest server
            self.f = make_f(input_shape)
            self.g = make_g(input_shape=self.f.layers[-1].output_shape[1:], class_num=class_num)

            # when there is some sort of attack
            if self.server_attack is not None:
                self.tilde_f = make_tilde_f(input_shape)

                assert self.f.output.shape.as_list()[1:] == self.tilde_f.output.shape.as_list()[1:]
                z_shape = self.tilde_f.output.shape.as_list()[1:]

                self.D = make_D(z_shape)
                self.decoder = self.loadBiasNetwork(make_decoder, z_shape, channels=input_shape[-1])

            # setup optimizers
            # the optimizer to update f and g are always present (in active attack, it only updates g since update of f has been hijacked)
            self.optimizer3 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_classify'])

            if self.server_attack is not None:
                self.optimizer0 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_f'])
                self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_tilde'])
                self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_D'])
            
            self.alpha1 = 0.0
            self.alpha2 = 1.0
            if ("alpha" in hparams) and (hparams["alpha"] is not None):
                self.alpha1, self.alpha2 = hparams["alpha"]
                print("Minimize distance correlation with alpha1=" + str(self.alpha1) + " and alpha2=" + str(self.alpha2))
                if self.attack == "active":
                    print("Actice attack scales alpha2 to 25.0")

    @staticmethod
    def addNoise(x, alpha):
        return x + tf.random.normal(x.shape) * alpha

    @tf.function
    def train_step(self, x_private, x_public, label_private, label_public):
        with tf.GradientTape(persistent=True) as tape:

            #### Virtually, ON THE CLIENT SIDE:
            # clients' smashed data, z_private = f(x_private)
            z_private = self.f(x_private, training=True)
            ####################################

            #### SERVER-SIDE:

            # classification output and loss
            server_output = self.g(z_private, training=True)
            if self.alpha1 != 0.0:
                # nopeek defense
                dcor = defense.dist_corr(x_private, z_private) * self.alpha1

            if self.class_num == 2:
                c_loss = tf.keras.losses.binary_crossentropy(y_true=label_private, y_pred=server_output) * self.alpha2
                c_train_accuracy = tf.keras.metrics.binary_accuracy(label_private, server_output)
            else:
                c_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label_private, y_pred=server_output) * self.alpha2
                c_train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(label_private, server_output)

            if self.server_attack is not None:
                # map to data space (for evaluation and style loss)
                if self.sorted:
                    z_private_sorted = tf.sort(z_private,axis=-1)
                    rec_x_private = self.decoder(z_private_sorted, training=True) # reconstructed x_private
                else:
                    rec_x_private = self.decoder(z_private, training=True) # reconstructed x_private

                # in the meantime, this is f'(xpub)
                z_public = self.tilde_f(x_public, training=True)
                # adversarial loss (f's output must similar be to \tilde{f}'s output):
                # discriminator classification output of f(xpriv), we want this to be all ones
                adv_private_logits = self.D(z_private, training=True)
                # f is trained to be classified by D as "generated", i.e. looks like the output of autoencoder's encoder f_tilda
                # discriminator classification output of f'(xpub), we want this to be all zeros
                adv_public_logits = self.D(z_public, training=True)
                if self.server_attack == "active":
                    if self.hparams['WGAN']:
                        # print("Use WGAN loss")
                        f_loss = tf.reduce_mean(adv_private_logits)
                    else:
                        f_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(adv_private_logits), adv_private_logits, from_logits=True))
                    if self.alpha1 != 0.0:
                        f_loss = f_loss * 25.0
                else:
                    if self.hparams['WGAN']:
                        f_loss = -tf.reduce_mean(adv_public_logits)
                    else:
                        f_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(adv_public_logits), adv_public_logits, from_logits=True))
                

                # invertibility loss: to train the decoder/autoencoder
                if self.sorted:
                    z_public_sorted = tf.sort(z_public, axis=-1)
                    rec_x_public = self.decoder(z_public_sorted, training=True)
                else:
                    rec_x_public = self.decoder(z_public, training=True)
                # print("decoder(tilde_f(pub_x)) shape")
                # print(rec_x_public.shape)

                # print(x_public.shape)
                tilde_f_loss = distance_data_loss(x_public, rec_x_public) # decoder loss, based on xpub (of course)

                # Discriminator loss
                if self.hparams['WGAN']:
                    loss_discr_true = tf.reduce_mean( adv_public_logits )
                    loss_discr_fake = -tf.reduce_mean( adv_private_logits )
                    # discriminator's loss
                    D_loss = loss_discr_true + loss_discr_fake
                else:
                    loss_discr_true = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(adv_public_logits), adv_public_logits, from_logits=True))
                    loss_discr_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(adv_private_logits), adv_private_logits, from_logits=True))
                    # discriminator's loss
                    D_loss = (loss_discr_true + loss_discr_fake) / 2

                if 'gradient_penalty' in self.hparams:
                    # print("Use GP")
                    w = float(self.hparams['gradient_penalty'])
                    D_gradient_penalty = self.gradient_penalty(z_private, z_public)
                    D_loss += D_gradient_penalty * w

                ##################################################################
                ## attack validation #####################
                loss_c_verification = distance_data(x_private, rec_x_private)
                ############################################
                ##################################################################


        # train all the models:
        if self.server_attack is None:
            # No attacks, directly update f and g
            var = self.f.trainable_variables + self.g.trainable_variables
            gradients = tape.gradient(c_loss, var)
            self.optimizer3.apply_gradients(zip(gradients, var))
        else:
            var = self.D.trainable_variables
            gradients = tape.gradient(D_loss, var)
            self.optimizer2.apply_gradients(zip(gradients, var))
            if self.server_attack == "active":
                # g is updated as usual
                var = self.g.trainable_variables
                gradients = tape.gradient(c_loss, var)
                self.optimizer3.apply_gradients(zip(gradients, var))
                # f is updated by f_loss
                var = self.f.trainable_variables
                gradients = tape.gradient(f_loss, var)
                self.optimizer0.apply_gradients(zip(gradients, var))
                if self.alpha1 != 0.0:
                    # f is also updated by distance decorrelation
                    var = self.f.trainable_variables
                    gradients = tape.gradient(dcor, var)
                    self.optimizer0.apply_gradients(zip(gradients, var))
                # encoder and decoder are updated together
                var = self.tilde_f.trainable_variables + self.decoder.trainable_variables
                gradients = tape.gradient(tilde_f_loss, var)
                self.optimizer1.apply_gradients(zip(gradients, var))
            elif self.server_attack == "passive":
                # f and g are updated together
                var = self.f.trainable_variables + self.g.trainable_variables
                gradients = tape.gradient(c_loss, var)
                self.optimizer3.apply_gradients(zip(gradients, var))
                if self.alpha1 != 0.0:
                    # f and g are also updated by distance decorrelation
                    var = self.f.trainable_variables
                    gradients = tape.gradient(dcor, var)
                    self.optimizer3.apply_gradients(zip(gradients, var))
                # encoder is updated by f_loss
                var = self.tilde_f.trainable_variables
                gradients = tape.gradient(f_loss, var)
                self.optimizer1.apply_gradients(zip(gradients, var))
                # decoder is updated separately
                var = self.decoder.trainable_variables
                gradients = tape.gradient(tilde_f_loss, var)
                self.optimizer1.apply_gradients(zip(gradients, var))

        if self.server_attack is None:
            return c_loss, c_train_accuracy
        else:
            return f_loss, tilde_f_loss, D_loss, loss_c_verification, c_loss, c_train_accuracy


    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.D(x_hat, training=True)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer
    
    
    @tf.function
    def score(self, x_private, label_private):
        z_private = self.f(x_private, training=False)
        tilde_x_private = self.decoder(z_private, training=False)
        
        err = tf.reduce_mean( distance_data(x_private, tilde_x_private) )
        
        return err
    
    def scoreAttack(self, dataset):
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        scorelog = 0
        i = 0
        for x_private, label_private in tqdm.tqdm(dataset):
            scorelog += self.score(x_private, label_private).numpy()
            i += 1
             
        return scorelog / i

    def attack(self, x_private):
        # smashed data sent from the client:
        z_private = self.f(x_private, training=False)
        # recover private data from smashed data
        tilde_x_private = self.decoder(z_private, training=False)

        z_private_control = self.tilde_f(x_private, training=False)
        control = self.decoder(z_private_control, training=False)
        return tilde_x_private.numpy(), control.numpy()


    def __call__(self, iterations, log_frequency=500, verbose=False, progress_bar=True):
        n = int(iterations / log_frequency)
        if self.server_attack is None:
            LOG = np.zeros((n,2))
        else:
            LOG = np.zeros((n,6))
        iterator = zip(self.client_dataset.take(iterations), self.attacker_dataset.take(iterations))
        if progress_bar:
            iterator = tqdm.tqdm(iterator , total=iterations)

        i, j = 0, 0
        print("RUNNING...")
        for (x_private, label_private), (x_public, label_public) in iterator:
            log = self.train_step(x_private, x_public, label_private, label_public)

            if self.server_attack is None:
                if i == 0:
                    train_loss = sum(log[0])/len(log[0])
                    train_accuracy = sum(log[1])/len(log[1])
                else:
                    train_loss += sum(log[0])/len(log[0])
                    train_accuracy += sum(log[1])/len(log[1])
                if i % log_frequency == log_frequency - 1:
                    train_loss = train_loss / log_frequency
                    train_accuracy = train_accuracy / log_frequency
                    LOG[j] = (sum(log[0])/len(log[0]), sum(log[1])/len(log[1]))
                    if verbose:
                        print("[log--%02d%%-%07d] train loss: %0.4f train accuracy: %0.4f" % ( int(i/iterations*100) ,i, train_loss, train_accuracy) )
                    j += 1
                    train_loss = 0
                    train_accuracy = 0
                i += 1
            else:
                # f_loss, tilde_f_loss, D_loss, loss_c_verification, c_loss, c_train_accuracy
                if i == 0:
                    attack_validation = log[3]
                    train_loss = sum(log[4])/len(log[4])
                    train_accuracy = sum(log[5])/len(log[5])
                else:
                    attack_validation += log[3]
                    train_loss += sum(log[4])/len(log[4])
                    train_accuracy += sum(log[5])/len(log[5])

                if  i % log_frequency == log_frequency - 1:
                    attack_validation = attack_validation / log_frequency
                    train_loss = train_loss / log_frequency
                    train_accuracy = train_accuracy / log_frequency
                    LOG[j] = log[0:4] + (sum(log[4])/len(log[4]), sum(log[5])/len(log[5]))

                    if verbose:
                        print("[log--%02d%%-%07d] reconstruction validation: %0.4f" % ( int(i/iterations*100) ,i, attack_validation) )
                        print("Original task: train loss: %0.4f train accuracy: %0.4f" % (train_loss, train_accuracy))

                    attack_validation = 0
                    train_loss = 0
                    train_accuracy = 0
                    j += 1

                i += 1

        return LOG