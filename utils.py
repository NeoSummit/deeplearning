import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math as m
import random

def describe(obj):
    """
    Renvoie le type et les dimensions d'un objet donné.

    Parameters:
        obj (any): L'objet à décrire.

    Returns:
        dict: Un dictionnaire contenant le type et les dimensions (si disponibles).
    """
    description = {
        "type": type(obj).__name__,  # Nom du type de l'objet
        "dimensions": None  # Placeholder pour les dimensions
    }
    
    # Cas pour les objets avec une méthode shape (comme NumPy arrays)
    if hasattr(obj, 'shape'):
        description['dimensions'] = obj.shape
    
    # Cas pour les objets de type list, tuple ou dict
    elif isinstance(obj, (list, tuple, dict)):
        description['dimensions'] = len(obj)
    
    # Cas pour les pandas DataFrame ou Series
    elif 'pandas' in description['type'].lower():
        try:
            description['dimensions'] = obj.shape
        except AttributeError:
            pass
    
    return description

def lire_alpha_digit(data,L):
    X=data['dat'][L[0]]
    for i in range(1,len(L)) :
        X_bis=data['dat'][L[i]]
        X=np.concatenate((X,X_bis),axis=0)
    n=X.shape[0]
    X=np.concatenate(X).reshape((n,320))
    return X

def display_images(images, size):
    for i in range(len(images)):
        nrow = len(images)/5
        nrow = m.ceil(nrow)
        plt.subplot(nrow, 5, i + 1) # 2 lignes, 5 colonnes et i+1 pour le numéro de l'image
        plt.title(f"Image {i + 1}")
        plt.imshow(images[i].reshape(size), cmap='gray')
        plt.axis('off')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_random_subsets(indices, subset_size):
    random.shuffle(indices)  # Mélange aléatoire des indices
    subsets = []
    
    # Diviser en sous-ensembles de taille subset_size
    for i in range(0, len(indices), subset_size):
        subsets.append(indices[i:i + subset_size])
        
    return subsets

def sample_binary(probs):
    """Convert probabilities into binary values by sampling.
    le retour est un tableau de 0 et 1 de la même taille que probs
    """
    return (np.random.rand(*probs.shape) < probs)*1 # *probs.shape pour déballer le tuple

def binarize_images(images):
    """Convert grayscale images to binary (0 or 1)."""
    return (images > 127).astype(np.float32) 

def cross_entropy_loss(probabilities, targets):
    """
    Calcule la Cross Entropy Loss à partir des probabilités et des cibles.

    Args:
        probabilities (np.ndarray): Tableau de taille (n_batch, n_sortie) contenant les probabilités prédites.
        targets (np.ndarray): Tableau de taille (n_batch,) contenant les indices des classes cibles (entiers).

    Returns:
        float: La valeur moyenne de la Cross Entropy Loss pour le batch.
    """
    # Vérification des dimensions
    assert probabilities.ndim == 2, "Les probabilités doivent être un tableau 2D (n_batch, n_sortie)."
    assert targets.ndim == 1, "Les cibles doivent être un tableau 1D (n_batch)."
    assert probabilities.shape[0] == targets.shape[0], "Le nombre d'exemples doit correspondre entre probabilités et targets."

    # Extraire les probabilités des classes cibles
    n_batch = probabilities.shape[0]
    target_probs = probabilities[np.arange(n_batch), targets]

    # Calcul de la perte de cross-entropie
    loss = -np.mean(np.log(target_probs))

    return loss

def convert_labels_to_numeric(labels, mapping=None):
    """
    Convertit une liste de labels en leurs représentations numériques.
    
    Parameters:
        labels (list): Liste de labels à convertir.
        mapping (list, optional): Liste des caractères valides. Par défaut, 0-9 et A-Z.

    """
    if mapping is None:
        mapping = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]  # 0-9 et A-Z
    char_to_index = {char: idx for idx, char in enumerate(mapping)}
    
    try:
        return [char_to_index[label] for label in labels]
    except KeyError as e:
        raise ValueError(f"Label invalide trouvé : {e.args[0]}") from e

