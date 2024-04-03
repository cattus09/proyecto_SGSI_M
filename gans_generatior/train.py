import os
import numpy as np
from dataset import MNIST
from discriminator import discriminator
from generator import generator
from losses import discriminator_loss, generator_loss
from noise import sample_noise
from optimizers import get_solvers
from setup import show_images
from setup import preprocess_img
import tensorflow as tf
import matplotlib.pyplot as plt


save_dir = "dataset"  
os.makedirs(save_dir, exist_ok=True)  


max_images = 80000

iter_count = 0

def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss,\
              show_every=20, print_every=20, batch_size=128, num_epochs=200, noise_size=96):
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    
    iter_count = 0
    for epoch in range(num_epochs):
        for (x, _) in mnist:

            with tf.GradientTape() as tape:
                real_data = x
                logits_real = D(preprocess_img(real_data))

                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)
                logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))

                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)      
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
            
            with tf.GradientTape() as tape:
                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))
                g_error = generator_loss(gen_logits_fake)
                g_gradients = tape.gradient(g_error, G.trainable_variables)      
                G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))
            if iter_count >= max_images:
                break
            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count,d_total_error,g_error))
                imgs_numpy = fake_images.numpy()
                show_images(imgs_numpy[0:16])
                ##plt.savefig(os.path.join(save_dir, 'imagen_{}.png'.format(iter_count)))
                plt.close() 

                
            iter_count += 1
        
        if iter_count >= max_images:
            break
    
    z = sample_noise(batch_size, noise_size)
    G_sample = G(z)
    print('Final images')
    show_images(G_sample[:16])
    plt.show()

# Make the discriminator
D = discriminator()

# Make the generator
G = generator()

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver, G_solver = get_solvers()

# Run it!
run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss)

G.save('generator_model.h5')
D.save('discriminator_model.h5')
