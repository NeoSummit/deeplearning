import numpy as np
import torch.nn as nn
import math
import torch
from principal_RBM_alpha import *
from principal_DBN_alpha import *
from utils import *

class DNN:
    def __init__(self, layer_sizes):
        self.dbn = DBN_alpha(layer_sizes[:-1]) # objet DBN_alpha
        self.layer_sizes = layer_sizes # liste des tailles des couches
        self.L = len(layer_sizes) - 1 # nombre de couches du DNN
        # passage de ref des paramètres des RBM du DBN au DNN
        self.W = { i + 1: self.dbn.rbm_list[i].W for i in range(self.L - 1) } # self.L-1 nombre de couches de RBM
        self.W[self.L] = np.random.uniform(-np.sqrt(6 / (layer_sizes[-2] + layer_sizes[-1])), np.sqrt(6 / (layer_sizes[-2] + layer_sizes[-1])), (layer_sizes[-2], layer_sizes[-1]))
        self.b = { i + 1: self.dbn.rbm_list[i].b for i in range(self.L - 1) }
        self.b[self.L] = np.zeros(layer_sizes[-1])

    def pretrain_DNN(self, donnees_entree, nb_iterations, batch_size, learning_rate, verbose=False):
        donnees_entree_copy = donnees_entree.copy() 
        self.dbn.train_DBN(donnees_entree_copy, nb_iterations, batch_size, learning_rate, verbose)

    def softmax(self, X):
            exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
            return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def entree_sortie_reseau(self, donnees_entree):
        sorties = [donnees_entree]
        for rbm in self.dbn.rbm_list:  # Les RBM du DNN
            sorties.append(rbm.entree_sortie(sorties[-1]))
        
        # Dernière couche (classification)
        sorties.append(self.softmax(sorties[-1] @ self.W[self.L] + self.b[self.L]))
        
        return sorties

    def retropropagation(self, epochs, learning_rate, batch_size, donnees_entree, labels, verbose=False):
        """
        donnees_entree: np.array de taille (n, n_entree)
        labels: np.array de taille (n,)

        Ajuste les poids/biais du réseau via la rétropropagation.
        """
        average_loss = []
        for epoch in range(epochs):
            loss = 0
            n_ = donnees_entree.shape[0]
            subsets = create_random_subsets(list(range(n_)), batch_size)
            for iter, sub in enumerate(subsets):
                X_batch = donnees_entree[sub] # (n_batch, n_entree)
                Y_batch = labels[sub] # (n_batch,)
                n_batch = X_batch.shape[0]
                sorties = self.entree_sortie_reseau(X_batch) # liste des sorties de chaque couche
                # One-hot encoding des labels
                Y_batch_eye = np.eye(self.layer_sizes[-1])[Y_batch] # (n_batch, n_L)
                
                c = {}
                grad_W = {}
                grad_b = {}
                # c_L shape (n_batch, n_L)
                c[self.L] = sorties[self.L] - Y_batch_eye    # pour la cross-entropy loss
                # grad_W_L shape (n_L-1, n_L)
                grad_W[self.L] = sorties[self.L-1].T @ c[self.L]
                # grad_b_L shape (n_L,)
                grad_b[self.L] = c[self.L].sum(axis=0) # shape (n_L,)  
                
                for l in range(self.L - 1, 0, -1): # l = L-1, L-2, ..., 1
                    # Calcul de c[l] shape (n_batch, n_l)
                    c[l] = (c[l + 1] @ self.W[l + 1].T) * sorties[l] * (1 - sorties[l])

                    # grad_W_l shape (n_l-1, n_l)
                    grad_W[l] = sorties[l - 1].T @ c[l]
                    # grad_b_l shape (n_l,)
                    grad_b[l] = c[l].sum(axis=0)
                
                # mise à jour des paramètres
                for l in range(1, self.L + 1): # l = 1, 2, ..., L
                    self.W[l] -= learning_rate * grad_W[l] / n_batch
                    self.b[l] -= learning_rate * grad_b[l] / n_batch
                # calcul de la loss
                loss += cross_entropy_loss(sorties[self.L], Y_batch)
            # Affichage de la loss moyenne
            if verbose:
                print(f"Époque {epoch+1}/{epochs}, Perte moyenne : {loss / len(subsets):.4f}")
            average_loss.append(loss / len(subsets))
        # Plot de la loss moyenne
        plt.plot(average_loss, label="Perte moyenne")
        plt.xlabel('Époques')
        plt.ylabel('Perte moyenne')
        plt.title('Évolution de la perte')
        plt.legend()
        plt.show()

    def test_DNN(self, donnees_entree, labels):
        """
        donnees_entree: np.array de taille (n, n_entree)
        labels: np.array de taille (n,)

        Évalue le modèle sur les données de test.
        """
        labels = labels.reshape(-1)
        predictions = np.argmax(self.entree_sortie_reseau(donnees_entree)[-1], axis=1) # (n_sample,)
        accuracy = np.mean(predictions == labels)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
