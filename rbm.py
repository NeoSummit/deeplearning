import numpy as np
import matplotlib.pyplot as plt
from utils import *

class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.random.randn(n_visible, n_hidden) * 0.1  # Poids
        self.a = np.zeros(n_visible)  # Biais des neurones visibles
        self.b = np.zeros(n_hidden)  # Biais des neurones cachés

    def entree_sortie(self, X):
        """Calcule la probabilité de sortie (neurones cachés) donnée l'entrée X."""
        return sigmoid(self.b + np.dot(X, self.W))  # X doit être de taille (*, n_visible)

    def sortie_entree(self, H):
        """Calcule la probabilité d'entrée (neurones visibles) donnée la sortie H."""
        return sigmoid(self.a + np.dot(H, self.W.T))# H doit être de taille (*, n_hidden) 

    def train(self, X, epochs, batch_size, learning_rate, verbose=False):
        """Entraîne le modèle RBM."""
        # size de X = (n_samples, n_visible)
        n_samples = X.shape[0]
        n_visible = self.n_visible
        losses = []
        if n_visible != X.shape[1]:
            raise ValueError("La taille de X doit être (n_samples, n_visible)")
        
        for epoch in range(epochs):
            np.random.shuffle(X)
            for batch_start in range(0, n_samples, batch_size): # range(0, n_samples, batch_size) = [0, batch_size, 2*batch_size, ...]
                X_batch = X[batch_start:min(batch_start + batch_size, n_samples)] #  size X = (n_batch, n_visible)
                n_batch = X_batch.shape[0]
                # algo CD-1
                v0 = X_batch   
                p_h_given_v0 = self.entree_sortie(v0)
                h0 = sample_binary(p_h_given_v0)
                p_v_given_h0 = self.sortie_entree(h0)
                v1 = sample_binary(p_v_given_h0)
                p_h_given_v1 = self.entree_sortie(v1)

                # Gradients
                grad_a = np.sum(v0 - v1, axis=0)
                grad_b = np.sum(p_h_given_v0 - p_h_given_v1, axis=0)
                grad_W = np.dot(v0.T, p_h_given_v0) - np.dot(v1.T, p_h_given_v1)

                # Mise à jour des paramètres
                self.W += learning_rate * grad_W / n_batch
                self.a += learning_rate * grad_a / n_batch
                self.b += learning_rate * grad_b / n_batch

            # Calcul de l'erreur de reconstruction
            h = self.entree_sortie(X)
            X_reconstruit = self.sortie_entree(h)
            loss = np.mean((X - X_reconstruit) ** 2)
            losses.append(loss)
            if epoch%50 == 0 and verbose:
                print("epoch " + str(epoch) + "/" + str(epochs) +" - loss : " + str(loss))
        plt.plot(losses)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Evolution of the loss through epochs')
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
        
    

    # Génération de données synthétiques pour l'entraînement

def generer_donnees_synthetiques(n_samples, n_features):
    """Génère des données binaires pour l'entraînement."""
    # np.random.seed(42)
    return (np.random.rand(n_samples, n_features) > 0.5).astype(np.float32)

def sample_binary(probs):
    """Convert probabilities into binary values by sampling."""
    return (np.random.rand(*probs.shape) < probs)*1 # *probs.shape pour déballer le tuple 
