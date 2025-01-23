from principal_RBM_alpha import *
import matplotlib.pyplot as plt
from utils import *


class DBN_alpha:
    def __init__(self, layer_sizes):
        self.rbm_list = [RBM(n_visible=layer_sizes[i], n_hidden=layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

    def train_DBN(self, X, epochs, batch_size, learning_rate, verbose=False):
        # Greedy layer-wise procedure
        input_data = X
        for rbm in self.rbm_list:
            print(f"Training RBM with {rbm.n_visible} visible and {rbm.n_hidden} hidden units")
            rbm.train(input_data, epochs, batch_size, learning_rate, verbose)
            # Transform data to hidden representation for the next layer
            input_data = rbm.entree_sortie(input_data)

    def generate_multiple_images(self, n_iter, n_images):
        """
        Génère plusieurs images en partant de la couche supérieure et en descendant.
        
        Args:
            n_iter (int): Nombre d'itérations de Gibbs Sampling pour échantillonner la couche supérieure.
            n_images (int): Nombre d'images à générer.
        
        Returns:
            list of numpy.ndarray: Liste des images générées (une par couche visible).
        """
        generated_images = []
        for i in range(n_images):
            # Étape 1 : Échantillonner la couche supérieure avec Gibbs Sampling
            rbm_top = self.rbm_list[-1]  # La dernière RBM (top layer)
            visible = np.random.rand(rbm_top.n_visible) > 0.5  # Initialisation aléatoire
            for _ in range(n_iter):
                hidden = sample_binary(rbm_top.entree_sortie(visible))
                visible = sample_binary(rbm_top.sortie_entree(hidden))
            
            # La couche supérieure est échantillonnée (visible contient les données échantillonnées)
            top_sample = visible

            # Étape 2 : Descendre dans le réseau couche par couche
            for rbm in reversed(self.rbm_list[-2::-1]):  # Descendre de la couche supérieure à la couche visible
                top_sample = sample_binary(rbm.sortie_entree(top_sample))
            
            # Ajouter l'image générée à la liste
            generated_images.append(top_sample)

        return generated_images

