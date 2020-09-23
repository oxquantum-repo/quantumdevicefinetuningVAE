import tensorflow as tf
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from .VAE_helper import VAE, load_json

class FactorVAE(VAE):
    def __init__(self, options_path = Path('./VAE/VAE_options.json')):

        options = load_json(options_path) # loading VAE_options.json
        for key, value in options.items(): # creating class attributes based on the VAE_options dict
            self.__setattr__(key, value)

        # Directories
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # model_name = 'factorvae_bias_bs%d_gam_%d_a_%d' %(self.bs,self.gamma,self.alpha)
        model_name = 'variance_factorvae_translated_bias_bs{}_gam_{}_a_{}'.format(self.bs, self.gamma, self.alpha)
        self.data_path = Path(self.data_path) # converting to operating system agnothic path

        self.model_path = Path(self.model_folder) / '{}_{}'.format(model_name, date)
        self.checkpoint_path = self.model_path / 'checkpoints/model'

        self.gif_path = self.model_path / 'traversal_gif'
        self.sorted_img_path = self.model_path / 'sorted_img_n={}'.format(self.n_neighbors)
        self.tb_path = self.model_path / 'tb_summaries/'

        # """
        # Data centered
        self.data_file = self.data_path / 'quantum_centered.npy'
        self.pref_array_path = self.data_path / 'pref_val_img_centered.npy'
        self.best_path = self.data_path / 'augmented_best_centered.npy'

        """       
  
        # Data translated
        self.data_file = self.data_path + 'quantum_translation.npy'
        self.pref_array_path= self.data_path + 'pref_val_img_translation.npy' 
        self.best_path=self.data_path + 'augmented_best_translation.npy'
        
        #"""
        self.all_img, self.data_train, self.data_test, self.pref_imgs, self.best_imgs = self._data_init()

        self.iterator, self.handle, self.train_img_ph, self.iterator_train, self.test_img_ph, self.iterator_test = \
            self._iterator_init()

        # Model setup
        self.input_vae, self.enc_mean, self.enc_logvar, self.z_sample, self.dec_logit, self.dec_sigm, \
        self.dec_mean_logit, self.dec_mean_sigm = self._vae_init(inputs=self.iterator.get_next())
        self.vae_loss, self.recon_loss, self.tc_estimate, self.tc_term, self.kl_divergence, self.disc_loss, self.probs_real_avg, self.probs_perm_avg = self._loss_init()
        self.vae_train_step, self.disc_train_step = self._optimizer_init()

        # Savers
        self.sess = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.train_writer = tf.summary.FileWriter(self.tb_path + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tb_path + 'test')
        tf.summary.scalar('vae_loss', self.vae_loss)
        tf.summary.scalar('recon_loss', self.recon_loss)
        tf.summary.scalar('disc_loss', self.disc_loss)
        tf.summary.scalar('tc_estimate', self.tc_estimate)

        # Initialization of variables
        self.sess.run(tf.global_variables_initializer())

        # Initialization of iterators
        self.sess.run([self.iterator_train.initializer, self.iterator_test.initializer],
                      feed_dict={self.train_img_ph: self.data_train, self.test_img_ph: self.data_test})

        # Initialization of handles
        self.train_handle = self.sess.run(self.iterator_train.string_handle())
        self.test_handle = self.sess.run(self.iterator_test.string_handle())

    def train(self):
        start_time = time.time()
        merged = tf.summary.merge_all()
        print("Beginning training")
        print("Beginning training", file=open(self.model_path + 'train.log', 'w'))
        it = 0

        self.sub_avg = []
        self.rec_loss_train = []
        self.kl_loss_train = []
        self.tc_loss_train = []
        self.disc_loss_train = []

        self.rec_loss_test = []
        self.tc_loss_test = []
        self.disc_loss_test = []

        while it < self.max_it:
            it += 1
            self.sess.run([self.vae_train_step, self.disc_train_step], feed_dict={self.handle: self.train_handle})

            if it % self.stat_every == 0:
                acc_sub = self.compute_pref_cost()
                self.sub_avg.append(acc_sub)
                # Train evaluation
                vae_loss, recon_loss, tc_est, tc_term, kl_div, disc_loss, probs_real, probs_perm, summ = self.sess.run(
                    [self.vae_loss, self.recon_loss, self.tc_estimate, self.tc_term, self.kl_divergence, self.disc_loss,
                     self.probs_real_avg, self.probs_perm_avg, merged],
                    feed_dict={self.handle: self.train_handle})
                print(
                    "Iteration %i (train):\n VAE loss %f - Rec loss %f - TC est %f - TC %f - KL %f- Disc loss %f probs_real %f probs_perm %f max acc_sub %f" % (
                        it, vae_loss, recon_loss, tc_est, tc_term, kl_div, disc_loss, probs_real, probs_perm,
                        np.max(acc_sub)), flush=True)
                print(
                    "Iteration %i (train):\n VAE loss %f - Rec loss %f - TC est %f - TC %f - KL %f- Disc loss %f probs_real %f probs_perm %f max acc_sub %f" % (
                        it, vae_loss, recon_loss, tc_est, tc_term, kl_div, disc_loss, probs_real, probs_perm,
                        np.max(acc_sub)), flush=True,
                    file=open(self.model_path + 'train.log', 'a'))
                self.train_writer.add_summary(summ, it)

                self.rec_loss_train.append(recon_loss)
                self.tc_loss_train.append(tc_est)
                self.kl_loss_train.append(kl_div)
                self.disc_loss_train.append(disc_loss)

                # Test evaluation
                vae_loss, recon_loss, tc_est, disc_loss, summ = self.sess.run(
                    [self.vae_loss, self.recon_loss, self.tc_estimate, self.disc_loss, merged],
                    feed_dict={self.handle: self.test_handle})
                print("Iteration %i (test):\n VAE loss %f - Rec loss %f - TC est %f - Disc loss %f" % (
                    it, vae_loss, recon_loss, tc_est, disc_loss), flush=True)
                print("Iteration %i (test):\n VAE loss %f - Rec loss %f - TC est %f - Disc loss %f" % (
                    it, vae_loss, recon_loss, tc_est, disc_loss), flush=True,
                      file=open(self.model_path + 'train.log', 'a'))
                self.test_writer.add_summary(summ, it)

                self.rec_loss_test.append(recon_loss)
                self.tc_loss_test.append(tc_est)
                self.disc_loss_test.append(disc_loss)

                time_usage = str(datetime.timedelta(seconds=int(round(time.time() - start_time))))
                print("Time usage: " + time_usage)
                print("Time usage: " + time_usage, file=open(self.model_path + 'train.log', 'a'))

            if it % self.saving_every == 0:
                save_path = self.saver.save(self.sess, self.checkpoint_path, global_step=it)
                print("Model saved to: %s" % save_path)
                print("Model saved to: %s" % save_path, file=open(self.model_path + 'train.log', 'a'))
                # self.get_traversals()
                # self.get_recontructions()

        save_path = self.saver.save(self.sess, self.checkpoint_path, global_step=it)
        print("Model saved to: %s" % save_path)
        print("Model saved to: %s" % save_path, file=open(self.model_path + 'train.log', 'a'))

        # Closing savers
        self.train_writer.close()
        self.test_writer.close()
        # Total time
        time_usage = str(datetime.timedelta(seconds=int(round(time.time() - start_time))))
        print("Total training time: " + time_usage)
        print("Total training time: " + time_usage, file=open(self.model_path + 'train.log', 'a'))

    def _discriminator_init(self, inputs, reuse=False):
        with tf.variable_scope("discriminator"):
            n_units = 1000
            disc_1 = tf.layers.dense(inputs=inputs, units=n_units, activation=tf.nn.leaky_relu, name="disc_1",
                                     reuse=reuse)
            disc_2 = tf.layers.dense(inputs=disc_1, units=n_units, activation=tf.nn.leaky_relu, name="disc_2",
                                     reuse=reuse)
            disc_3 = tf.layers.dense(inputs=disc_2, units=n_units, activation=tf.nn.leaky_relu, name="disc_3",
                                     reuse=reuse)
            disc_4 = tf.layers.dense(inputs=disc_3, units=n_units, activation=tf.nn.leaky_relu, name="disc_4",
                                     reuse=reuse)
            # disc_5 = tf.layers.dense(inputs=disc_4, units=n_units, activation=tf.nn.leaky_relu, name="disc_5", reuse=reuse)
            # disc_6 = tf.layers.dense(inputs=disc_5, units=n_units, activation=tf.nn.leaky_relu, name="disc_6", reuse=reuse)
            logits = tf.layers.dense(inputs=disc_4, units=2, name="disc_logits", reuse=reuse)
            probabilities = tf.nn.softmax(logits, name="disc_prob")

        return logits, probabilities

    def _loss_init(self):

        with tf.name_scope("disc"):
            # Get samples from the second batch
            input_aux = self.iterator.get_next()
            aux_mean, aux_logvar = self._encoder_init(inputs=input_aux, reuse=True)
            with tf.name_scope("sampling"):
                # Reparameterisation trick
                with tf.name_scope("noise"):
                    noise = tf.random_normal(shape=tf.shape(aux_mean))
                with tf.name_scope("variance"):
                    variance = tf.exp(aux_logvar / 2)
                with tf.name_scope("reparam_trick"):
                    aux_samples = tf.add(aux_mean, (variance * noise))
            real_samples = self.z_sample

            # Discrimination of joint samples
            logits_real, probs_real = self._discriminator_init(real_samples)
            with tf.name_scope("permute_dims"):
                # Use second batch to produce independent samples
                permuted_rows = []
                for i in range(aux_samples.get_shape()[1]):
                    permuted_rows.append(tf.random_shuffle(aux_samples[:, i]))
                permuted_samples = tf.stack(permuted_rows, axis=1)
            # Discrimination of independent samples
            logits_permuted, probs_permuted = self._discriminator_init(permuted_samples, reuse=True)

        with tf.name_scope("loss"):
            with tf.name_scope("reconstruction_loss"):
                # Reconstruction loss is bernoulli in each pixel
                im_flat = tf.reshape(self.input_vae, shape=[-1, 32 * 32 * 1])
                # logits_flat = tf.reshape(self.dec_logit, shape=[-1, 32 * 32 * 1])
                sigm_flat = tf.reshape(self.dec_sigm, shape=[-1, 32 * 32 * 1])

                # by_pixel_recon = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_flat, labels=im_flat)
                # by_example_recon = tf.reduce_sum(by_pixel_recon, axis=1)
                # recon_loss = tf.reduce_mean(by_example_recon)

                rec = tf.reduce_sum(tf.squared_difference(sigm_flat, im_flat), 1)
                recon_loss = tf.reduce_mean(rec, name="recon_loss")

                # recon_loss3= tf.reduce_mean(tf.reduce_sum(tf.squared_difference(logits_flat, im_flat),1))

            with tf.name_scope("kl_loss"):
                kl_divergence = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + self.enc_logvar
                                                                    - tf.square(self.enc_mean)
                                                                    - tf.exp(self.enc_logvar), 1))

            with tf.name_scope("tc_loss"):
                # FactorVAE paper has gamma * log(D(z) / (1- D(z))) in Algo 2, where D(z) is probability of being real
                # Let PT be probability of being true, PF be probability of being false. Then we want log(PT/PF)
                # Since PT = exp(logit_T) / [exp(logit_T) + exp(logit_F)]
                # and  PT = exp(logit_F) / [exp(logit_T) + exp(logit_F)], we have that
                # log(PT/PF) = logit_T - logit_F
                logit_t = logits_real[:, 0]
                logit_f = logits_real[:, 1]
                tc_estimate = tf.reduce_mean(logit_t - logit_f, axis=0)
                tc_term = tc_estimate

            with tf.name_scope("total_vae_loss"):
                vae_loss = self.alpha * recon_loss + kl_divergence + self.gamma * tc_term

            with tf.name_scope("disc_loss"):
                probs_real_avg = tf.reduce_mean(probs_real[:, 0])
                probs_perm_avg = tf.reduce_mean(probs_permuted[:, 1])

                real_samples_loss = tf.reduce_mean(tf.log(probs_real[:, 0]))
                permuted_samples_loss = tf.reduce_mean(tf.log(probs_permuted[:, 1]))
                disc_loss = - tf.add(0.5 * real_samples_loss,
                                     0.5 * permuted_samples_loss,
                                     name="disc_loss")

        return vae_loss, recon_loss, tc_estimate, tc_term, kl_divergence, disc_loss, probs_real_avg, probs_perm_avg

    def _optimizer_init(self):
        with tf.name_scope("optimizer"):
            with tf.name_scope("vae_optimizer"):
                enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
                dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
                vae_vars = enc_vars + dec_vars
                vae_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,
                                                       beta1=0.9,
                                                       beta2=0.999)
                vae_train_step = vae_optimizer.minimize(self.vae_loss, var_list=vae_vars)
            with tf.name_scope("disc_optimizer"):
                disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                disc_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,
                                                        beta1=0.5,
                                                        beta2=0.9)
                disc_train_step = disc_optimizer.minimize(self.disc_loss, var_list=disc_vars)
        return vae_train_step, disc_train_step


if __name__ == "__main__":
    tf.reset_default_graph()
    train = False
    if train == True:
        vae = FactorVAE(gamma=0, alpha=34, bs=96)
        vae.train()

        plt.figure()
        plt.title('Reconstruction loss')
        plt.plot(vae.rec_loss_test, label='test')
        plt.plot(vae.rec_loss_train, label='train')
        plt.legend()
        plt.savefig('%srec_loss' % (vae.model_path))
        plt.show()

        plt.figure()
        plt.title('Pref loss')
        count = 1
        for i in np.asarray(vae.sub_avg).T:
            plt.plot(i, label='k_n:%d' % count)
            plt.legend()
            count += 1
        plt.savefig('%spref_loss' % (vae.model_path))
        plt.show()

        plt.figure()
        plt.title('KL loss')
        plt.plot(vae.kl_loss_train)
        plt.savefig('%skl_loss' % (vae.model_path))
        plt.show()

        plt.figure()
        plt.title('TC loss')
        plt.plot(vae.tc_loss_test, label='test')
        plt.plot(vae.tc_loss_train, label='train')
        plt.legend()
        plt.savefig('%stc_loss' % (vae.model_path))
        plt.show()

        plt.figure()
        plt.title('Disc loss')
        plt.plot(vae.disc_loss_test, label='test')
        plt.plot(vae.disc_loss_train, label='train')
        plt.legend()
        plt.savefig('%sdisc_loss' % (vae.model_path))
        plt.show()

    else:
        vae = FactorVAE(gamma=0, alpha=34, bs=96)
        vae.load_latest_checkpoint(
            path='/media/oxml/DA12F77512F754CB/VAE_results/sub_test/variance_translated_factorvae_bias_bs96_gam_0_a_34_20190819-204147/checkpoints')

    make_plots = False
    if make_plots == True:
        print('get reconstructions')
        vae.get_recontructions()

        print('get traversals')
        idx_trav = [0, 600, 4350, 6550]
        for i in idx_trav:
            vae.get_traversals(example_index=i, show_figure=True)
        print("Evaluating final model")
        recon_loss_test = vae.evaluate_test_recon_loss()
        print("Test Reconstruction Loss: ", recon_loss_test)
        acc_sub = vae.compute_pref_cost()
        print('Preference loss: ', acc_sub)

        vae.plot_sample_traversal()

    plot_lat = False
    if plot_lat == True:
        print('plot latent space')
        vae.plot_latent_space()

        # print('create gifs')
        # vae.plot_sample_traversal()

    rank_test = False
    if rank_test == True:
        vae.rank_test()

    plot_slide = False
    if plot_slide == True:

        plt.figure()

        img = np.loadtxt(fname="/home/oxml/Dropbox/deep_rl_quantum/Basel_Triton training-testing/test_img_triton.txt")[
              :100, :100]  # 15x15 /10x10
        # img=np.loadtxt(fname = "/home/oxml/Dropbox/deep_rl_quantum/Basel_Triton training-testing/test_img_triton_2.txt")[:200,:200] #35x35 /40x40
        # img=np.loadtxt(fname = "/home/oxml/Dropbox/deep_rl_quantum/vu_drl/data/I1_174.txt") #20x20 / 30x30

        # file = pickle.load( open( "/home/oxml/Dropbox/deep_rl_quantum/Basel_Triton training-testing/measurement_8.pkl", "rb" ) ) #30x30/35x35/40x40
        # img=file['chan0']['data']*-1

        plt.imshow(img)
        plt.show()

        w_size = 15

        range_data = [np.min(img), np.max(img)]
        norm = (img - range_data[0]) / (range_data[1] - range_data[0])  # normalize before subsampling

        result_matrix = np.zeros((norm.shape[0] - w_size, norm.shape[1] - w_size))
        for ii in range(0, norm.shape[0] - w_size):
            for jj in range(0, norm.shape[1] - w_size):
                sub_arr = norm[ii:ii + w_size, jj:jj + w_size]
                sub_arr = cv2.resize(sub_arr, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
                sub_arr = b = sub_arr[np.newaxis, :, :, np.newaxis]
                score = vae.score(sub_arr, n_neighbor=5)
                result_matrix[ii, jj] = score
                result_matrix_norm = (np.max(result_matrix) - result_matrix) / (
                            np.max(result_matrix) - np.min(result_matrix))

        plt.figure()
        plt.imshow(result_matrix_norm)
        plt.colorbar()
