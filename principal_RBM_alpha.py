import numpy as np
import matplotlib.pyplot as plt
from utils import *
import math

class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.random.uniform(
            low=-np.sqrt(6 / (n_visible + n_hidden)),
            high=np.sqrt(6 / (n_visible + n_hidden)),
            size=(n_visible, n_hidden)
        )  # shape (n_visible, n_hidden)
        self.a = np.zeros(n_visible)  # shape (n_visible,)
        self.b = np.zeros(n_hidden)  # shape (n_hidden,) 

    def entree_sortie(self, X):
        # X doit être de taille (*, n_visible)
        return sigmoid(self.b + X @ self.W) # shape (*, n_hidden)

    def sortie_entree(self, H):
        # H doit être de taille (*, n_hidden)
        return sigmoid(self.a + H @ self.W.T)# shape (*, n_visible)

    def train(self, X, epochs, batch_size, learning_rate, verbose=False):
        """Entraîne le modèle RBM."""
        # size de X = (n_samples, n_visible)
        # X doit être un np.array de taille (n_samples, n_visible)
        if not isinstance(X, np.ndarray):
            raise ValueError("X doit être un np.array")
        if self.n_visible != X.shape[1]:
            raise ValueError("La taille de X doit être (n_sample, n_visible)")
        n_samples = X.shape[0]
        average_loss = []
        for epoch in range(epochs):
            loss = 0
            np.random.shuffle(X) # X est mélangé
            for i in range(0, n_samples, batch_size): 
                X_batch = X[i:i + batch_size] #  size X = (n_batch, n_visible)
                n_batch = X_batch.shape[0]
                # algo CD-1
                v0 = X_batch   # (n_batch, n_visible)
                p_h_given_v0 = self.entree_sortie(v0) # (n_batch, n_hidden)
                h0 = sample_binary(p_h_given_v0) # (n_batch, n_hidden)
                p_v_given_h0 = self.sortie_entree(h0) # (n_batch, n_visible)
                v1 = sample_binary(p_v_given_h0) # (n_batch, n_visible)
                p_h_given_v1 = self.entree_sortie(v1) # (n_batch, n_hidden)

                # Gradients
                grad_a = np.sum(v0 - v1, axis=0) # (n_visible,)
                grad_b = np.sum(p_h_given_v0 - p_h_given_v1, axis=0) # (n_hidden,)
                grad_W = v0.T @ p_h_given_v0 - v1.T @ p_h_given_v1 # (n_visible, n_hidden)

                # Montée de gradient car maximisation de la vraisemblance
                self.W += learning_rate * grad_W / n_batch
                self.a += learning_rate * grad_a / n_batch
                self.b += learning_rate * grad_b / n_batch

                # Calcul de l'erreur de reconstruction
                loss += np.mean((v0 - self.sortie_entree(self.entree_sortie(v0))) ** 2)

            # afficher la loss moyenne
            if verbose:
                print(f"epoch {epoch+1}/{epochs} - Perte moyenne : {loss / math.ceil(n_samples / batch_size)}")
            average_loss.append(loss / math.ceil(n_samples / batch_size))
            
        plt.plot(average_loss, label="Perte moyenne")
        plt.xlabel('Époques')
        plt.ylabel('Perte moyenne')
        plt.title('Évolution de la perte')
        plt.legend()
        plt.show()


    def generer_donnees(self, n_iter, n_images):
        """Génère des données après l'entraînement."""
        generated_images = []
        for _ in range(n_images):
            v = (np.random.rand(self.n_visible) < 0.5) * 1
            for _ in range(n_iter):
                h = sample_binary(self.entree_sortie(v))
                v = sample_binary(self.sortie_entree(h))
            generated_images.append(v)
        return generated_images
        
    


