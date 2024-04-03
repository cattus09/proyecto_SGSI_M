import os
import matplotlib.pyplot as plt
from tensorflow import keras
from gans_generatior.noise import sample_noise
from gans_generatior.setup import show_images

# Definir función para cargar los modelos entrenados
def load_models(generator_path, discriminator_path):
    loaded_generator = keras.models.load_model(generator_path)
    loaded_discriminator = keras.models.load_model(discriminator_path)
    return loaded_generator, loaded_discriminator

# Función para generar imágenes usando el generador cargado
def generate_images(generator, num_images, noise_dim):
    noise = sample_noise(num_images, noise_dim)
    generated_images = generator.predict(noise)
    return generated_images

# Cargar los modelos entrenados
generator_path = os.path.join( 'generator_model.h5')
discriminator_path = os.path.join( 'discriminator_model.h5')
loaded_generator, loaded_discriminator = load_models(generator_path, discriminator_path)

# Generar nuevas imágenes con el generador cargado
num_images_to_generate = 16
noise_dim = 96
generated_images = generate_images(loaded_generator, num_images_to_generate, noise_dim)

# Mostrar las imágenes generadas
show_images(generated_images)
plt.show()