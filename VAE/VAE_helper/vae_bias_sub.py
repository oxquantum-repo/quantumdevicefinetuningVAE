import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit
import os
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import imageio
import cv2
from matplotlib.colors import Normalize


class VAE(object):
    """
    def __init__(self):
        # VAE parameters
        self.z_dim = 10

        # Iterations parameters
        self.max_it = 300000
        self.stat_every = 1000
        self.saving_every = 10000

        # Directories
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = 'vae_dsprites_bias'
        self.data_path = 'database/'
        self.model_path = 'results/' + model_name + '_' + date + '/'
        self.checkpoint_path = self.model_path + 'checkpoints/model'
        self.tb_path = self.model_path + 'tb_summaries/'
        
        # Data
        self.quantum_path = self.data_path + 'data_quantum.npy'
        self.pref_array_path= self.data_path + 'pref_val.npy'
        self.best_path=self.data_path + 'augmented_best.npy'
        
        self.data_train, self.data_test,  self.pref_imgs, self.best_imgs = self._data_init()
        self.iterator, self.handle, self.train_img_ph, self.iterator_train, self.test_img_ph, self.iterator_test =\
            self._iterator_init(batch_size=64)

        # Model setup
        self.input_vae, self.enc_mean, self.enc_logvar, self.z_sample, self.dec_logit, self.dec_sigm, \
            self.dec_mean_logit, self.dec_mean_sigm = self._vae_init(inputs=self.iterator.get_next())
        self.vae_loss, self.recon_loss = self._loss_init()
        self.vae_train_step = self._optimizer_init()

        # Savers
        self.sess = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.train_writer = tf.summary.FileWriter(self.tb_path + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tb_path + 'test')
        tf.summary.scalar('vae_loss', self.vae_loss)
        tf.summary.scalar('recon_loss', self.recon_loss)

        # Initialization of variables
        self.sess.run(tf.global_variables_initializer())

        # Initialization of iterators
        self.sess.run([self.iterator_train.initializer, self.iterator_test.initializer],
                      feed_dict={self.train_img_ph: self.data_train, self.test_img_ph: self.data_test})

        # Initialization of handles
        self.train_handle = self.sess.run(self.iterator_train.string_handle())
        self.test_handle = self.sess.run(self.iterator_test.string_handle())

    def train(self, final_evaluation=False):
        start_time = time.time()
        merged = tf.summary.merge_all()
        print("Beginning training")
        print("Beginning training", file=open(self.model_path + 'train.log', 'w'))
        it = 0

        while it < self.max_it:
            it += 1
            self.sess.run(self.vae_train_step, feed_dict={self.handle: self.train_handle})

            if it % self.stat_every == 0:

                # Train evaluation
                vae_loss, recon_loss, summ = self.sess.run([self.vae_loss, self.recon_loss, merged],
                                                           feed_dict={self.handle: self.train_handle})
                print("Iteration %i (train):\n VAE loss %f - Recon loss %f" % (it, vae_loss, recon_loss), flush=True)
                print("Iteration %i (train):\n VAE loss %f - Recon loss %f" % (it, vae_loss, recon_loss),
                      flush=True, file=open(self.model_path + 'train.log', 'a'))
                self.train_writer.add_summary(summ, it)

                # Test evaluation
                vae_loss, recon_loss, summ = self.sess.run([self.vae_loss, self.recon_loss, merged],
                                                           feed_dict={self.handle: self.test_handle})
                print("Iteration %i (test):\n VAE loss %f - Recon loss %f" % (it, vae_loss, recon_loss), flush=True)
                print("Iteration %i (test):\n VAE loss %f - Recon loss %f" % (it, vae_loss, recon_loss),
                      flush=True, file=open(self.model_path + 'train.log', 'a'))
                self.test_writer.add_summary(summ, it)

                time_usage = str(datetime.timedelta(seconds=int(round(time.time() - start_time))))
                print("Time usage: " + time_usage)
                print("Time usage: " + time_usage, file=open(self.model_path + 'train.log', 'a'))

            if it % self.saving_every == 0:
                save_path = self.saver.save(self.sess, self.checkpoint_path, global_step=it)
                print("Model saved to: %s" % save_path)
                print("Model saved to: %s" % save_path, file=open(self.model_path + 'train.log', 'a'))

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

        if final_evaluation:
            print("Evaluating final model...")
            mean_dis_metric = self.evaluate_mean_disentanglement()
            recon_loss_test = self.evaluate_test_recon_loss()
            print("Mean Disentanglement Metric: " + str(mean_dis_metric),
                  file=open(self.model_path + 'train.log', 'a'))
            print("Test Reconstruction Loss: " + str(recon_loss_test),
                  file=open(self.model_path + 'train.log', 'a'))
            """

    def load_latest_checkpoint(self, path):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))

    def _data_init(self):
        data = np.load(self.data_file).squeeze()
        print(data.shape)
        pref_array = np.load(self.pref_array_path).squeeze()
        print(pref_array.shape)
        best_imgs = np.load(self.best_path).squeeze()
        print(best_imgs.shape)
        data_list = []
        for i in data:
            img = i.reshape(32, 32)
            data_list.append(img)

        pref_list = []
        for i in pref_array:
            img = i.reshape(32, 32)
            pref_list.append(img)

        quantum_array = np.asarray(data_list)
        pref_array = np.asarray(pref_list)

        all_imgs = quantum_array[:, :, :, None]  # make into 4d tensor
        pref_imgs = pref_array[:, :, :, None]  # make into 4d tensor
        best_imgs = best_imgs[:, :, :, None]  # make into 4d tensor

        # 90% random test/train split
        n_data = all_imgs.shape[0]
        idx_random = np.random.permutation(n_data)
        data_train = all_imgs[idx_random[0: (9 * n_data) // 10]]
        data_test = all_imgs[idx_random[(9 * n_data) // 10:]]

        return all_imgs, data_train, data_test, pref_imgs, best_imgs

    def _iterator_init(self):
        with tf.name_scope("iterators"):
            # Generate TF Dataset objects for each split
            train_img_ph = tf.placeholder(dtype=tf.float32, shape=self.data_train.shape)
            test_img_ph = tf.placeholder(dtype=tf.float32, shape=self.data_test.shape)
            dataset_train = tf.data.Dataset.from_tensor_slices(train_img_ph)
            dataset_test = tf.data.Dataset.from_tensor_slices(test_img_ph)
            dataset_train = dataset_train.repeat()
            dataset_test = dataset_test.repeat()

            # Random batching
            dataset_train = dataset_train.shuffle(buffer_size=5000)
            dataset_train = dataset_train.batch(batch_size=self.bs)
            dataset_test = dataset_test.shuffle(buffer_size=1000)
            dataset_test = dataset_test.batch(batch_size=self.bs)

            # Prefetch
            dataset_train = dataset_train.prefetch(buffer_size=4)
            dataset_test = dataset_test.prefetch(buffer_size=4)

            # Iterator for each split
            iterator_train = dataset_train.make_initializable_iterator()
            iterator_test = dataset_test.make_initializable_iterator()

            # Global iterator
            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                handle, dataset_train.output_types, dataset_train.output_shapes)

        return iterator, handle, train_img_ph, iterator_train, test_img_ph, iterator_test

    def _vae_init(self, inputs):
        with tf.name_scope("vae"):
            # Input
            input_vae = inputs
            # Encoder network
            enc_mean, enc_logvar = self._encoder_init(inputs=input_vae)
            with tf.name_scope("sampling"):
                # Reparameterisation trick
                with tf.name_scope("noise"):
                    noise = tf.random_normal(shape=tf.shape(enc_mean))
                with tf.name_scope("variance"):
                    variance = tf.exp(enc_logvar / 2)
                with tf.name_scope("reparam_trick"):
                    z_sample = tf.add(enc_mean, (variance * noise))
            # Decoder network
            dec_logit, dec_sigm = self._decoder_init(inputs=z_sample)
            # Non-random decoder
            dec_mean_logit, dec_mean_sigm = self._decoder_init(inputs=enc_mean, reuse=True)

        return input_vae, enc_mean, enc_logvar, z_sample, dec_logit, dec_sigm, dec_mean_logit, dec_mean_sigm

    def _encoder_init(self, inputs, reuse=False):
        with tf.variable_scope("encoder"):
            e_1 = tf.layers.conv2d(inputs=inputs,
                                   filters=32,
                                   kernel_size=4,
                                   strides=2,
                                   activation=tf.nn.relu,
                                   padding="same",
                                   name="enc_conv_1",
                                   reuse=reuse)
            print(e_1)

            e_2 = tf.layers.conv2d(inputs=e_1,
                                   filters=64,
                                   kernel_size=4,
                                   strides=2,
                                   activation=tf.nn.relu,
                                   padding="same",
                                   name="enc_conv_2",
                                   reuse=reuse)
            print(e_2)
            e_3 = tf.layers.conv2d(inputs=e_2,
                                   filters=64,
                                   kernel_size=4,
                                   strides=2,
                                   activation=tf.nn.relu,
                                   padding="same",
                                   name="enc_conv_3",
                                   reuse=reuse)

            print(e_3)
            with tf.name_scope("enc_flatten"):
                dim = np.prod(e_3.get_shape().as_list()[1:])
                print(dim)
                e_4_flat = tf.reshape(e_3, shape=(-1, dim))
                print(e_4_flat)
            e_4 = tf.layers.dense(inputs=e_4_flat,
                                  units=128,
                                  activation=None,
                                  name="enc_fc_1",
                                  reuse=reuse)
            print(e_4)
            e_5 = tf.layers.dense(inputs=e_4,
                                  units=self.z_dim * 2,
                                  activation=None,
                                  name="enc_fc_2",
                                  reuse=reuse)
            print(e_5)
            enc_mean = e_5[:, :self.z_dim]
            print(enc_mean)
            enc_logvar = e_5[:, self.z_dim:]
            print(enc_logvar)
        return enc_mean, enc_logvar

    def _decoder_init(self, inputs, reuse=False):
        with tf.variable_scope("decoder"):
            d_1 = tf.layers.dense(inputs=inputs,
                                  units=128,
                                  activation=tf.nn.relu,
                                  name="dec_fc_1",
                                  reuse=reuse)
            print(d_1)
            d_2 = tf.layers.dense(inputs=d_1,
                                  units=4 * 4 * 64,
                                  activation=tf.nn.relu,
                                  name="dec_fc_2",
                                  reuse=reuse)
            print(d_2)
            with tf.name_scope("dec_reshape"):
                d_2_reshape = tf.reshape(d_2, shape=[-1, 4, 4, 64])
            print(d_2_reshape)

            d_3 = tf.layers.conv2d_transpose(inputs=d_2_reshape,
                                             filters=64,
                                             kernel_size=4,
                                             strides=2,
                                             activation=tf.nn.relu,
                                             padding="same",
                                             name="dec_upconv_1",
                                             reuse=reuse)
            print(d_3)
            d_4 = tf.layers.conv2d_transpose(inputs=d_3,
                                             filters=32,
                                             kernel_size=4,
                                             strides=2,
                                             activation=tf.nn.relu,
                                             padding="same",
                                             name="dec_upconv_2",
                                             reuse=reuse)
            print(d_4)
            dec_logit = tf.layers.conv2d_transpose(inputs=d_4,
                                                   filters=1,
                                                   kernel_size=4,
                                                   strides=2,
                                                   activation=None,
                                                   padding="same",
                                                   name="dec_upconv_3",
                                                   reuse=reuse)
            print(dec_logit)
            dec_sigm = tf.sigmoid(dec_logit, name="dec_sigmoid_out")

        return dec_logit, dec_sigm

    def evaluate_test_recon_loss(self):
        print("Evaluating reconstruction loss in test set.")
        recon_loss = 0
        batch_size = self.bs
        n_data_test = self.data_test.shape[0]
        n_batches = int(n_data_test / batch_size)
        print("Total batches:", n_batches)
        for i in range(n_batches):
            start_img = i * batch_size
            end_img = (i + 1) * batch_size
            batch_imgs = self.data_test[start_img:end_img, :, :, :]
            # Reconstruction without random sampling
            dec_mean_sigm = self.sess.run(self.dec_mean_sigm, feed_dict={self.input_vae: batch_imgs})
            # Error according to non-random reconstruction
            this_recon_loss = self.sess.run(self.recon_loss,
                                            feed_dict={self.dec_sigm: dec_mean_sigm, self.input_vae: batch_imgs})
            recon_loss = recon_loss + this_recon_loss
            if (i + 1) % 100 == 0:
                print(str(i + 1) + "/" + str(n_batches) + " evaluated.")
        recon_loss = recon_loss / n_batches
        print("Reconstruction loss: " + str(recon_loss))
        return recon_loss

    def compute_mean_kl_dim_wise(self, batch_mu, batch_logvar):
        # Shape of batch_mu is [batch, z_dim], same for batch_logvar
        # KL against N(0,1) is 0.5 * ( var_j - logvar_j + mean^2_j - 1 )
        variance = np.exp(batch_logvar)
        squared_mean = np.square(batch_mu)
        batch_kl = 0.5 * (variance - batch_logvar + squared_mean - 1)
        mean_kl = np.mean(batch_kl, axis=0)
        return mean_kl

    def get_traversals(self, example_index=None, show_figure=False):
        # Return a list of arrays (n_travers, 32, 32), one per dimension.
        # Dimensions are sorted in descending order of KL divergence

        if example_index == None:
            len_data_set = len(self.data_test)
            example_index = np.random.choice(range(len_data_set), 1, replace=False)
            originals = self.data_test[example_index, :, :, :]
            print(originals.shape)

        else:
            originals = self.all_img[np.array([example_index]), :, :, :]
            print(originals.shape)

        z_base, logvar_base = self.sess.run([self.enc_mean, self.enc_logvar], feed_dict={self.input_vae: originals})
        # Sort by KL (in descending order)
        mean_kl = self.compute_mean_kl_dim_wise(z_base, logvar_base)
        sorted_dims = np.argsort(-mean_kl)
        trav_values = np.arange(-2, 2.1, 0.5)
        n_trav = len(trav_values)
        traversals = []

        z_base_batch = np.concatenate([np.copy(z_base) for _ in range(n_trav)], axis=0)

        for j in sorted_dims:
            z_sample = np.copy(z_base_batch)
            z_sample[:, j] = trav_values
            generated_images = self.sess.run(self.dec_sigm, feed_dict={self.z_sample: z_sample})
            # generated_images = self.sess.run(self.dec_mean_logit, feed_dict={self.enc_mean: z_sample})
            traversals.append(generated_images[:, :, :, 0])  # shape (n_trav, 32, 32)

        if show_figure:
            plt.figure()
            plt.imshow(originals.squeeze(), vmin=0, vmax=1, cmap='bwr')
            plt.title('original')
            plt.show()

            plt.figure(figsize=(1.5 * n_trav, 1.5 * self.z_dim))
            gs1 = gridspec.GridSpec(self.z_dim, n_trav)
            gs1.update(wspace=0.02, hspace=0.02)
            for i in range(self.z_dim):
                for j in range(n_trav):
                    # Plot traversals for this z_j
                    ax1 = plt.subplot(gs1[i, j])
                    ax1.set_aspect('equal')
                    plt.axis('off')
                    ax1.imshow(traversals[i][j, :, :], vmin=0, vmax=1, cmap='bwr')
            plt.savefig('%strav_%d.pdf' % (self.model_path, example_index))
            plt.show()
        return

    def get_recontructions(self, example_index=None, show_figure=True):
        # Originals in first row, reconstructions in second row

        if example_index == None:
            len_data_set = len(self.data_test)
            example_index = np.random.choice(range(len_data_set), 5, replace=False)

        originals = self.data_test[example_index, :, :, :]
        # Non-random reconstructions
        reconstructions1 = self.sess.run(self.dec_mean_sigm, feed_dict={self.input_vae: originals})

        originals = originals[:, :, :, 0]
        # reconstructions = reconstructions[:, :, :, 0]
        reconstructions1 = reconstructions1[:, :, :, 0]

        if show_figure:
            # Prepare plot
            n_examples = len(example_index)
            plt.figure(figsize=(2 * n_examples, 2 * 2))
            gs1 = gridspec.GridSpec(2, n_examples)
            gs1.update(wspace=0.02, hspace=0.02)
            print('originals')
            # Plot originals
            for i in range(n_examples):
                ax1 = plt.subplot(gs1[0, i])
                ax1.set_aspect('equal')
                plt.axis('off')
                ax1.imshow(originals[i, :, :], vmin=0, vmax=1, cmap='bwr')
                print(np.min(originals[i, :, :]), np.max(originals[i, :, :]))

            print('reconstructions')
            # Plot reconstructions
            for i in range(n_examples):
                ax1 = plt.subplot(gs1[1, i])
                ax1.set_aspect('equal')
                plt.axis('off')
                ax1.imshow(reconstructions1[i, :, :], vmin=0, vmax=1, cmap='bwr')
                print(np.min(reconstructions1[i, :, :]), np.max(reconstructions1[i, :, :]))
            plt.savefig('%srec.pdf' % (self.model_path))
            plt.show()

        return

    def reconstruct(self, data):

        if len(data.shape) == 2:
            data_res = cv2.resize(data, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
            data_exp = data_res[np.newaxis, :, :, np.newaxis]
        else:
            data_exp = data

        reconstruction = self.sess.run(self.dec_mean_sigm, feed_dict={self.input_vae: data_exp})

        plt.figure(figsize=(8, 10))
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.02, hspace=0.02)
        print('original')
        # Plot original
        ax1 = plt.subplot(gs1[0, 0])
        ax1.set_aspect('equal')
        plt.axis('off')
        ax1.imshow(data_exp.squeeze(), vmin=0, vmax=1, cmap='bwr')

        print('reconstruction')
        # Plot reconstruction
        ax1 = plt.subplot(gs1[0, 1])
        ax1.set_aspect('equal')
        plt.axis('off')
        ax1.imshow(reconstruction.squeeze(), vmin=0, vmax=1, cmap='bwr')
        plt.show()

        return

    def rank_test(self):
        os.mkdir(self.sorted_img_path)
        z_best, z_best_logvar = self.sess.run([self.enc_mean, self.enc_logvar],
                                              feed_dict={self.input_vae: self.best_imgs})
        z_test, z_test_logvar = self.sess.run([self.enc_mean, self.enc_logvar],
                                              feed_dict={self.input_vae: self.data_test})

        z_test_var = np.exp(z_test_logvar)
        z_test_var_mean = np.mean(z_test_var, axis=1)

        dis_avg = dis_array_substract(z_best, z_test, z_best_logvar, z_test_logvar, n_neighbor=self.n_neighbors)
        order = np.argsort(dis_avg)
        test_sort = self.data_test[order]
        var_order = z_test_var_mean[order]
        dis_order = dis_avg[order]

        count = 0
        save_gif = True
        if save_gif == True:
            sort_rank = []
            for img in test_sort:
                x_gif = img.squeeze()
                sort_rank.append(x_gif)
                plt.figure()
                plt.imshow(x_gif)
                plt.axis('off')
                # plt.title('ranked:%d_var_%f,dis_%f,'  %(count,var_order[count],dis_order[count]))
                plt.savefig('%s/%d.pdf' % (self.sorted_img_path, count))
                plt.show()
                count += 1

    def compute_pref_cost(self):
        z_best, z_best_logvar = self.sess.run([self.enc_mean, self.enc_logvar],
                                              feed_dict={self.input_vae: self.best_imgs})
        z_pref, z_pref_logvar = self.sess.run([self.enc_mean, self.enc_logvar],
                                              feed_dict={self.input_vae: self.pref_imgs})

        acc_sub_list = []
        for neighbor in range(1, 7):
            dis_avg_sub = dis_array_substract(z_best, z_pref, z_best_logvar, z_pref_logvar, n_neighbor=neighbor)
            score_sub = dis_avg_sub[:int(len(dis_avg_sub) / 2)] - dis_avg_sub[int(len(dis_avg_sub) / 2):]
            score_sig_sub = expit(score_sub)
            acc_sub = len(score_sig_sub[score_sig_sub < 0.5]) / (len(score_sig_sub))
            acc_sub_list.append(acc_sub)
        return acc_sub_list

    def plot_latent_space(self):
        z_test = self.sess.run(self.enc_mean, feed_dict={self.input_vae: self.data_test})
        print('TSNE: perplexity 20')
        tsne = manifold.TSNE(n_components=2, perplexity=20)
        tsne_result = tsne.fit_transform(z_test)
        tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
        self.visualize_scatter(data_2d=tsne_result_scaled, tsne=True, com=20)
        self.visualize_scatter_with_images(X_2d_data=tsne_result_scaled, tsne=True, images=self.data_test.squeeze(),
                                           image_zoom=0.7, com=20)

        print('TSNE: perplexity 30')
        tsne = manifold.TSNE(n_components=2, perplexity=30)
        tsne_result = tsne.fit_transform(z_test)
        tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
        self.visualize_scatter(data_2d=tsne_result_scaled, tsne=True, com=30)
        self.visualize_scatter_with_images(X_2d_data=tsne_result_scaled, tsne=True, images=self.data_test.squeeze(),
                                           image_zoom=0.7, com=30)

        print('TSNE: perplexity 15')
        tsne = manifold.TSNE(n_components=2, perplexity=15)
        tsne_result = tsne.fit_transform(z_test)
        tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
        self.visualize_scatter(data_2d=tsne_result_scaled, tsne=True, com=15)
        self.visualize_scatter_with_images(X_2d_data=tsne_result_scaled, tsne=True, images=self.data_test.squeeze(),
                                           image_zoom=0.7, com=15)

        print('TSNE: perplexity 35')
        tsne = manifold.TSNE(n_components=2, perplexity=35)
        tsne_result = tsne.fit_transform(z_test)
        tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
        self.visualize_scatter(data_2d=tsne_result_scaled, tsne=True, com=35)
        self.visualize_scatter_with_images(X_2d_data=tsne_result_scaled, tsne=True, images=self.data_test.squeeze(),
                                           image_zoom=0.7, com=35)

        """
        print('Spectral: neighbors 5')
        se = manifold.SpectralEmbedding(n_components=2, n_neighbors=5)
        Y = se.fit_transform(z_test)
        self.visualize_scatter(data_2d=Y,tsne=False,com=5)
        self.visualize_scatter_with_images(X_2d_data=Y,tsne=False, images = self.data_test.squeeze(), image_zoom=0.7,com=5)
        
        print('Spectral: neighbors 10')
        se = manifold.SpectralEmbedding(n_components=2, n_neighbors=10)
        Y = se.fit_transform(z_test)
        self.visualize_scatter(data_2d=Y,tsne=False,com=10)
        self.visualize_scatter_with_images(X_2d_data=Y,tsne=False, images = self.data_test.squeeze(), image_zoom=0.7,com=10)
        """

    def plot_sample_traversal(self):
        # Creating 10 random gifs of length n_plot:
        n_plot = 40
        n_gif = 10
        os.mkdir(self.gif_path)
        for j in range(n_gif):
            # os.mkdir('%s/gif_%d' %(self.gif_path,j))
            os.mkdir('%s/gif_%d' % (self.gif_path, j))
            index = np.random.choice(range(len(self.data_test)), 2, replace=False)
            z_mu1 = self.sess.run(self.enc_mean,
                                  feed_dict={self.input_vae: np.expand_dims(self.data_test[index[0]], axis=0)})
            z_mu2 = self.sess.run(self.enc_mean,
                                  feed_dict={self.input_vae: np.expand_dims(self.data_test[index[1]], axis=0)})
            transition_array = np.linspace(np.expand_dims(z_mu1, axis=0), np.expand_dims(z_mu2, axis=0), n_plot,
                                           axis=1).squeeze()
            x_mean = self.sess.run(self.dec_sigm, feed_dict={self.z_sample: transition_array})

            for m in range(n_plot):
                plt.imshow(x_mean[m].squeeze(), vmin=0, vmax=1, cmap='bwr')
                plt.axis('off')
                plt.savefig('%s/gif_%d/img_%d.png' % (self.gif_path, j, m))
                plt.show()

        for i in range(n_gif):
            directory = ('%s/gif_%d' % (self.gif_path, i))
            gif_list = []
            for j in range(n_plot):
                filename = ('img_%d.png' % j)
                img_path = ('%s/%s' % (directory, filename))
                gif_list.append(imageio.imread(img_path))
            imageio.mimsave('%s/gif_%d.gif' % (self.gif_path, i), gif_list)

    def visualize_scatter_with_images(self, X_2d_data, images, tsne, com, figsize=(15, 15), image_zoom=1):
        fig, ax = plt.subplots(figsize=figsize)
        artists = []
        for xy, i in zip(X_2d_data, images):
            x0, y0 = xy
            img = OffsetImage(i, zoom=image_zoom, cmap='bwr', norm=Normalize(vmin=0, vmax=1, clip=False))
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(X_2d_data)
        ax.autoscale()
        ax.grid(True)
        if tsne == True:
            plt.savefig('%simg_tsne_%d.svg' % (self.model_path, com))
            plt.savefig('%simg_tsne_%d.pdf' % (self.model_path, com))
        else:
            plt.savefig('%simg_spectral_%d.svg' % (self.model_path, com))

        plt.show()

    def visualize_scatter(self, data_2d, tsne, com, figsize=(12, 12)):

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(data_2d[:, 0],
                   data_2d[:, 1],
                   marker='o',
                   linewidth='1',
                   alpha=0.8,
                   )

        ax.tick_params(labeltop=True, labelright=True)

        ax.legend(loc='best')
        ax.grid(True)
        if tsne == True:
            plt.savefig('%stsne_%d.svg' % (self.model_path, com))
        else:
            plt.savefig('%sspectral_%d.svg' % (self.model_path, com))
        plt.show()

    def score(self, input_img, n_neighbor=3):

        z_best, z_best_logvar = self.sess.run([self.enc_mean, self.enc_logvar],
                                              feed_dict={self.input_vae: self.best_imgs})
        z_test, z_test_logvar = self.sess.run([self.enc_mean, self.enc_logvar], feed_dict={self.input_vae: input_img})
        dis_avg = dis_array_substract(z_best, z_test, z_best_logvar, z_test_logvar, n_neighbor=n_neighbor)

        return dis_avg


def dis_array_substract(z1, z2, z1_logvar, z2_logvar, n_neighbor):
    z1_var = np.exp(z1_logvar)
    z2_var = np.exp(z2_logvar)

    z2_new = z2[np.newaxis, :]
    z1_new = z1[:, np.newaxis, :]

    z2_var_new = z2_var[np.newaxis, :]
    z1_var_new = z1_var[:, np.newaxis, :]

    z_diff = (z2_new - z1_new) ** 2
    z_var_sum = (z2_var_new + z1_var_new)  # **2
    tot_dis = z_diff + z_var_sum
    tot_dis_sum = np.sum(tot_dis, axis=2)

    dis_sort = np.sort(tot_dis_sum, axis=0)
    dis_avg = np.mean(dis_sort[:n_neighbor, :], axis=0)

    return dis_avg


if __name__ == "__main__":
    vae = VAE()
    # vae.train()
