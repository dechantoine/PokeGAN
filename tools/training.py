import datetime
import tensorflow as tf
from tensorflow.keras import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pokeplot import plot_image, plot_multiple_images_with_scores

def scores_latent_space(gan, codings_size, sample_size=1000, output_size=10):
    generator, discriminator = gan.layers
    noise = tf.random.normal(shape=[sample_size, codings_size])
    gen_noise = tf.concat([generator(tf.reshape(n, shape=[1,200]), training=True) for n in noise], axis=0)
    scores = discriminator(gen_noise).numpy().reshape(sample_size)
    best = np.sort(scores)[np.linspace(0,sample_size-1,output_size).astype(int)]
    indices = np.argsort(scores)[np.linspace(0,sample_size-1,output_size).astype(int)]
    best_fakes = tf.gather(gen_noise,indices)
    return best_fakes, best


def save_metrics(df_metrics, epoch, r_loss_g, r_loss_d_fake, r_loss_d_real, r_accuracy_g, r_accuracy_d_fake,
                 r_accuracy_d_real):
    stats = pd.Series(r_loss_g).describe().iloc[[1, 3, 4, 5, 6, 7]]
    for i in stats.index:
        df_metrics.loc[epoch, ("running_loss_g", i)] = stats.loc[i]

    stats = pd.Series(r_loss_d_fake).describe().iloc[[1, 3, 4, 5, 6, 7]]
    for i in stats.index:
        df_metrics.loc[epoch, ("running_loss_d_fake", i)] = stats.loc[i]

    stats = pd.Series(r_loss_d_real).describe().iloc[[1, 3, 4, 5, 6, 7]]
    for i in stats.index:
        df_metrics.loc[epoch, ("running_loss_d_real", i)] = stats.loc[i]

    stats = pd.Series(r_accuracy_g).describe().iloc[[1, 3, 4, 5, 6, 7]]
    for i in stats.index:
        df_metrics.loc[epoch, ("running_accuracy_g", i)] = stats.loc[i]

    stats = pd.Series(r_accuracy_d_fake).describe().iloc[[1, 3, 4, 5, 6, 7]]
    for i in stats.index:
        df_metrics.loc[epoch, ("running_accuracy_d_fake", i)] = stats.loc[i]

    stats = pd.Series(r_accuracy_d_real).describe().iloc[[1, 3, 4, 5, 6, 7]]
    for i in stats.index:
        df_metrics.loc[epoch, ("running_accuracy_d_real", i)] = stats.loc[i]

    return df_metrics


def train_gan(gan, preprocessing, gan_number, images, batch_size, codings_size,
              learning_ratio=5, n_epochs=50, resume_training_at_epoch=1,
              images_saving_ratio=5, model_saving_ratio=5):
    generator, discriminator = gan.layers
    m = metrics.BinaryAccuracy(threshold=0.5)

    resume_training_at_epoch = max(resume_training_at_epoch, 1)

    if resume_training_at_epoch == 1:
        samples = tf.random.normal(shape=[16, codings_size])
        np.savetxt('./DCGAN_{}/params/samples.csv'.format(gan_number), samples.numpy(), delimiter=',')

        logs = pd.DataFrame(columns=pd.MultiIndex.from_arrays([np.repeat(["running_loss_g",
                                                                          "running_loss_d_fake",
                                                                          "running_loss_d_real",
                                                                          "running_accuracy_g",
                                                                          "running_accuracy_d_fake",
                                                                          "running_accuracy_d_real"], 6),
                                                               np.tile(["mean",
                                                                        "min",
                                                                        "25%",
                                                                        "50%",
                                                                        "75%",
                                                                        "max"], 6)]))
    else:
        samples = tf.convert_to_tensor(np.loadtxt('./DCGAN_{}/params/samples.csv'.format(gan_number), delimiter=','))
        logs = pd.read_excel("./DCGAN_{}/logs.xlsx".format(gan_number), index_col=0, header=[0, 1])

    for epoch in np.arange(resume_training_at_epoch, n_epochs + 1):
        n = 0
        print("------------------------------------ Epoch {}/{} ------------------------------------".format(epoch,
                                                                                                             n_epochs))

        start_epoch = datetime.datetime.now()

        dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(1000).batch(batch_size,
                                                                                 drop_remainder=True).prefetch(1)

        running_loss_g = []
        running_loss_d_fake = []
        running_loss_d_real = []
        running_accuracy_g = []
        running_accuracy_d_fake = []
        running_accuracy_d_real = []

        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            # X_fake = generator(noise)
            X_fake = generator(noise, training=True)
            y_fake = tf.constant([[(np.random.random() * 0.2)] for i in range(batch_size)])
            y_batch = tf.constant([[(np.random.random() * 0.2) + 0.8] for i in range(batch_size)])

            discriminator.trainable = True
            predicted_real = discriminator(preprocessing(X_batch))
            m.update_state(tf.constant([[1.0]] * batch_size), predicted_real)
            d_accuracy_real = m.result().numpy()
            running_accuracy_d_real += [d_accuracy_real]
            m.reset_states()
            d_loss_real = discriminator.train_on_batch(preprocessing(X_batch), y_batch)
            running_loss_d_real += [d_loss_real]

            predicted_fake = discriminator(X_fake)
            m.update_state(tf.constant([[0.0]] * batch_size), predicted_fake)
            d_accuracy_fake = m.result().numpy()
            running_accuracy_d_fake += [d_accuracy_fake]
            m.reset_states()
            d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
            running_loss_d_fake += [d_loss_fake]

            n += 1

            if n % learning_ratio == 0:
                # phase 2 - training the generator
                noise = tf.random.normal(shape=[batch_size, codings_size])
                y2 = tf.constant([[1.0]] * batch_size)
                discriminator.trainable = False
                # predicted = discriminator(generator(noise))
                predicted = discriminator(generator(noise, training=True))
                m.update_state(tf.constant([[1.0]] * batch_size), predicted)
                g_accuracy = m.result().numpy()
                running_accuracy_g += [g_accuracy]
                m.reset_states()
                g_loss = gan.train_on_batch(noise, y2)
                running_loss_g += [g_loss]

        logs = save_metrics(logs, epoch, running_loss_g, running_loss_d_fake, running_loss_d_real,
                            running_accuracy_g, running_accuracy_d_fake, running_accuracy_d_real)
        logs.to_excel("./DCGAN_{}/logs.xlsx".format(gan_number))
        print(logs.iloc[epoch - 1, [18, 24, 30]])

        duration_epoch = datetime.datetime.now() - start_epoch
        print("Time to train GAN on this epoch: {} seconds".format(duration_epoch.seconds))

        start_samples = datetime.datetime.now()
        X_samples = tf.concat([generator(tf.reshape(s, shape=[1, codings_size]), training=True) for s in samples],
                              axis=0)
        for k in np.arange(16):
            plot_image(np.squeeze(X_samples[k], axis=-1))
            plt.savefig('./DCGAN_{}/samples/{}/epoch_{:04d}.png'.format(gan_number, k, epoch),
                        bbox_inches='tight', pad_inches=0)
            plt.close()
        duration_samples = datetime.datetime.now() - start_samples
        print("Time to generate and save samples: {} seconds".format(duration_samples.seconds))

        if epoch % images_saving_ratio == 0:
            start_images = datetime.datetime.now()
            predicted_samples = discriminator(X_samples).numpy().reshape(16)
            best_fakes, best = scores_latent_space(gan, codings_size, sample_size=1000, output_size=16)
            plot_multiple_images_with_scores(tf.concat([X_samples, best_fakes], 0),
                                             np.concatenate([predicted_samples, best]), 8)
            plt.savefig('./DCGAN_{}/epoch_{:04d}.png'.format(gan_number, epoch), dpi=1200)
            plt.close()
            duration_images = datetime.datetime.now() - start_images
            print("Time to generate and save images: {} seconds".format(duration_images.seconds))

        if epoch % model_saving_ratio == 0:
            start_checkpoint = datetime.datetime.now()
            gan.save('./DCGAN_{}/models/model_at_epoch_{}.h5'.format(gan_number, epoch))
            duration_checkpoint = datetime.datetime.now() - start_checkpoint
            print("Time to save model checkpoint: {} seconds".format(duration_checkpoint.seconds))