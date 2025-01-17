import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math as m

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

