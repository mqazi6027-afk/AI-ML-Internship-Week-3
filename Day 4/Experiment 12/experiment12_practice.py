
import numpy as np

print("=" * 65)
print("     M-Tech AI/ML Internship - Experiment 12")
print(" Generative AI - GANs, VAEs & Diffusion Models")
print("=" * 65)

latent_vector = np.random.randn(8)

print("\nRandom Latent Vector")
print(latent_vector)


generated_data = np.random.randint(0, 256, (4, 4))

print("\nGenerated Sample")

print(generated_data)


models = [
    "GAN (Generative Adversarial Network)",
    "VAE (Variational Autoencoder)",
    "Diffusion Model"
]

print("\nGenerative AI Models")

for model in models:
    print("-", model)


print("\nComparison")

print("GAN       : Produces realistic synthetic data.")
print("VAE       : Learns latent representations.")
print("Diffusion : Generates high-quality images step-by-step.")


applications = [
    "Image Generation",
    "Image Enhancement",
    "Art Creation",
    "Face Generation",
    "Medical Imaging",
    "Content Creation"
]

print("\nApplications")

for app in applications:
    print("-", app)

print("\nExperiment 12 Completed Successfully")
print("=" * 65)