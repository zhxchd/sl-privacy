import tensorflow as tf
import numpy as np
import tqdm
import defense

class property_inference_attack:
    
    def loadBiasNetwork(self, make_decoder, z_shape, channels):
        return make_decoder(z_shape, channels=channels)
        
    def __init__(self, models, xpriv, xpub, batch_size, hparams, class_num, sorted=False):
            input_shape = xpriv.element_spec[0].shape
            self.hparams = hparams
            self.class_num = class_num
            self.sorted = sorted

            # setup dataset
            self.client_dataset = xpriv.batch(batch_size, drop_remainder=True).repeat(-1)
            self.attacker_dataset = xpub.batch(batch_size, drop_remainder=True).repeat(-1)

            # print([x[0]for x in xpriv])
            # self.property_priv = xpriv.map(lambda x: num_to_cat[x[0][property_id]]).batch(batch_size, drop_remainder=True).repeat(-1)
            
            self.batch_size = batch_size

            ## setup models
            make_f, make_g, make_tilde_f, make_classifier, make_D, make_get_property = models

            # f and g are always present, even with an honest server
            self.f = make_f(input_shape)
            self.g = make_g(input_shape=self.f.layers[-1].output_shape[1:], class_num=class_num)
            self.get_property = make_get_property(input_shape)

            # when there is some sort of attack
            if True:
                self.tilde_f = make_tilde_f(input_shape)

                assert self.f.output.shape.as_list()[1:] == self.tilde_f.output.shape.as_list()[1:]
                z_shape = self.tilde_f.output.shape.as_list()[1:]

                self.D = make_D(z_shape)
                self.classifier = self.loadBiasNetwork(make_classifier, z_shape, channels=input_shape[-1])

            # setup optimizers
            # the optimizer to update f and g are always present (in active attack, it only updates g since update of f has been hijacked)
            self.optimizer3 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_classify'])

            if True:
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
    def train_step(self, x_private, x_public, label_private, label_public, property_priv, property_pub):

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

            if True:
                # map to data space (for evaluation and style loss)
                if self.sorted:
                    z_private_sorted = tf.sort(z_private,axis=-1)
                    inferred_property_priv = self.classifier(z_private_sorted, training=True) # inferred property of x_private
                else:
                    inferred_property_priv = self.classifier(z_private, training=True) # inferred property of x_private

                # in the meantime, this is f'(xpub)
                z_public = self.tilde_f(x_public, training=True)
                # adversarial loss (f's output must similar be to \tilde{f}'s output):
                # discriminator classification output of f(xpriv), we want this to be all ones
                adv_private_logits = self.D(z_private, training=True)
                # f is trained to be classified by D as "generated", i.e. looks like the output of autoencoder's encoder f_tilda
                # discriminator classification output of f'(xpub), we want this to be all zeros
                adv_public_logits = self.D(z_public, training=True)

                if self.hparams['WGAN']:
                    f_loss = -tf.reduce_mean(adv_public_logits)
                else:
                    f_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(adv_public_logits), adv_public_logits, from_logits=True))

                # invertibility loss: to train the decoder/classifier
                if self.sorted:
                    z_public_sorted = tf.sort(z_public, axis=-1)
                    inferred_property_pub = self.classifier(z_public_sorted, training=True)
                else:
                    inferred_property_pub = self.classifier(z_public, training=True)

                tilde_f_loss = tf.keras.losses.binary_crossentropy(property_pub, inferred_property_pub)
                tilde_f_acc = tf.keras.metrics.binary_accuracy(property_pub, inferred_property_pub)

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
                # loss_c_verification = distance_data(x_private, rec_x_private)
                loss_c_verification = tf.keras.metrics.binary_crossentropy(property_priv, inferred_property_priv)
                acc_c_verification = tf.keras.metrics.binary_accuracy(property_priv, inferred_property_priv)
                ############################################
                ##################################################################

        # train all the models:
        if True:
            var = self.D.trainable_variables
            gradients = tape.gradient(D_loss, var)
            self.optimizer2.apply_gradients(zip(gradients, var))
            if True:
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
                # classifier is updated separately
                var = self.classifier.trainable_variables
                gradients = tape.gradient(tilde_f_loss, var)
                self.optimizer1.apply_gradients(zip(gradients, var))
        
        return tilde_f_loss, tilde_f_acc, loss_c_verification, acc_c_verification


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

    def attack(self, x_private):
        # smashed data sent from the client:
        z_private = self.f(x_private, training=False)
        # recover private data from smashed data
        inferred_property = self.classifier(z_private, training=False)

        # z_private_control = self.tilde_f(x_private, training=False)
        # control = self.decoder(z_private_control, training=False)
        return inferred_property.numpy()


    def __call__(self, iterations, log_frequency=500, verbose=False, progress_bar=True):
        n = int(iterations / log_frequency)
        LOG = np.zeros((iterations,4))
        iterator = zip(self.client_dataset.take(iterations), self.attacker_dataset.take(iterations))
        if progress_bar:
            iterator = tqdm.tqdm(iterator , total=iterations)

        i, j = 0, 0
        print("RUNNING...")
        for (x_private, label_private, property_private), (x_public, label_public, property_public) in iterator:
            log = self.train_step(x_private, x_public, label_private, label_public, property_private, property_public)
            LOG[j] = (sum(log[0]/len(log[0])), sum(log[1])/len(log[1]), sum(log[2])/len(log[2]), sum(log[3])/len(log[3]))
            j += 1
            # log = [0]tilde_f_loss, [1]tilde_f_acc, [2]loss_c_verification, [3]acc_c_verification
            if i == 0:
                classify_loss = sum(log[0]/len(log[0]))
                classify_acc = sum(log[1]/len(log[1]))
                attack_loss = sum(log[2]/len(log[2]))
                attack_validation = sum(log[3]/len(log[3]))
            else:
                classify_loss += sum(log[0]/len(log[0]))
                classify_acc += sum(log[1]/len(log[1]))
                attack_loss += sum(log[2]/len(log[2]))
                attack_validation += sum(log[3]/len(log[3]))

            if  i % log_frequency == log_frequency - 1:
                classify_loss = classify_loss / log_frequency
                classify_acc = classify_acc / log_frequency
                attack_loss = attack_loss / log_frequency
                attack_validation = attack_validation / log_frequency
                # LOG[j] = (sum(log[0]/len(log[0])), sum(log[1])/len(log[1]), sum(log[2])/len(log[2]), sum(log[3])/len(log[3]))

                if verbose:
                    print("[log--%02d%%-%07d] property inference loss: %0.4f accuracy: %0.4f" % ( int(i/iterations*100) ,i, attack_loss,attack_validation) )
                    print("Server-side classifier: loss: %0.4f accuracy: %0.4f" % (classify_loss, classify_acc))
                
                attack_loss = 0
                attack_validation = 0
                classify_loss = 0
                classify_acc = 0

            i += 1

        return LOG