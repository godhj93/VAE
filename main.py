from multiprocessing import Value
from tqdm import tqdm
from utils.dataloader import dataloader
from utils.net import VAE
from utils.utils import train_step
import tensorflow as tf
import numpy as np
def main():

    train_data_loader = dataloader()
    train_data = train_data_loader.get_batched_dataset()
    train_data_loader.length()
    epochs = 10

    model = VAE(latent_dim=128)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    best_elbo = np.inf
    loss_fn = tf.keras.metrics.Mean()
    for epoch in range(epochs):
        
        pbar = tqdm(enumerate(train_data))
        for step, (s, a, s_next) in pbar:

            train_step(model, (s, a, s_next), optimizer, metric_fn = loss_fn)

            if step >= train_data_loader.length():
                break
                
            
            elbo = -loss_fn.result()
            pbar.set_description(f"Epoch: {epoch+1}/{epochs}, Step: {step}/{train_data_loader.length()}, ELBO(Loss): {elbo:.4f}")
        
        
        

if __name__ == "__main__":

    main()