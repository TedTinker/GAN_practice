import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display

# Change to your own working directory
os.chdir("C:\\Users\\tedjt\\Desktop\\Thinkster\\40 Generative Adversarial Network\\GAN\\attempt_2")
from utilities import *

# Which dataset will you work with?
# As appropriate, swap generate_and_save_shifting_images_gray to generate_and_save_shifting_images_rgb
train_images = get_mnist()
#train_images = get_squids()
#train_images = get_pokemon()

print("Whole dataset: ", train_images.shape)
image_shape = train_images.shape[1:]
print("One image: ", image_shape)

### Various hyperparameters
d = .5              # Dropout percentage
noise_dim = 100     # How much noise to generate with
num_examples_to_generate = 4
BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

### Structure generator
def make_generator_model():
    
    small_size = tuple(int(i/4) for i in image_shape[:-1])
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(small_size[0]*small_size[1]*64, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((small_size[0],small_size[1],64)))

    model.add(layers.Lambda(lambda t: ref_pad(t,paddings = [[0,0],[2,2],[2,2],[0,0]])))
    model.add(layers.Conv2D(
            128, 
            (5, 5), 
            strides=(1, 1), 
            use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Lambda(lambda t: ref_pad(t,paddings = [[0,0],[2,2],[2,2],[0,0]])))
    model.add(layers.Conv2D(
            64, 
            (5, 5), 
            use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Lambda(lambda t: ref_pad(t,paddings = [[0,0],[2,2],[2,2],[0,0]])))
    model.add(layers.Conv2D(
            image_shape[-1], 
            (5, 5), 
            use_bias=False, 
            activation='tanh'))

    model.summary()
    
    return model

generator = make_generator_model()

noise = tf.random.normal([1, noise_dim])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

### Structure discriminator
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.GaussianNoise(stddev = .03))
    model.add(layers.Lambda(lambda t: ref_pad(t,paddings = [[0,0],[2,2],[2,2],[0,0]])))
    model.add(layers.Conv2D(
            64, 
            (5, 5), 
            strides=(2, 2), 
            input_shape=image_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(d))

    model.add(layers.Lambda(lambda t: ref_pad(t,paddings = [[0,0],[2,2],[2,2],[0,0]])))
    model.add(layers.Conv2D(
            128, 
            (5, 5), 
            strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(d))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    model.build(input_shape = (None,) + image_shape)
    model.summary()

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

### Seeds for overseeing training
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
def train(dataset, epochs):
        
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
        train_step(image_batch)
      
    display.clear_output(wait=True)
    if(epoch % 1 == 0):  # If epochs pass too quickly, don't generate images every time
        generate_and_save_images_gray(generator,epoch + 1,seed)
        #generate_and_save_images_rgb(generator,epoch + 1,seed)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images_gray(generator,epochs,seed)
  #generate_and_save_images_rgb(generator,epochs,seed)
  
train(train_dataset, 100)

### Shapeshifting images!
def shape_shifting_image(n = 50, m = 15):
    seed_1 = tf.random.normal([1, noise_dim])
    for i in range(m):
        epoch = 0
        seed_2 = tf.random.normal([1, noise_dim])
        while(epoch < n):
            seed = ((seed_2 * epoch) + (seed_1 * (n-(epoch))))/n
            generate_and_save_shifting_images_gray(generator, epoch, i, seed)
            #generate_and_save_shifting_images_rgb(generator, epoch, i, seed)
            epoch += 1
        seed_1 = seed_2
        
shape_shifting_image()