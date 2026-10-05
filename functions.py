import numpy as np

"""
    Funzione di attivazione ReLU (Rectified Linear Unit)
    g(z) = max{0, z}
    
    Args:
        z: Input della funzione di attivazione
        
    Returns:
        Output della funzione di attivazione ReLU
"""
def relu(z):

    return np.maximum(0, z)

"""
    Funzione di attivazione Lineare
    g(z) = z
    
    Args:
        z: Input della funzione di attivazione
        
    Returns:
        Output della funzione di attivazione Lineare
"""
def linear(z):

    return z

"""
    Funzione di attivazione Tangente Iperbolica
    g(z) = tanh(z)

    Args:
        z: Input della funzione di attivazione
    
    Returns:
        Output della funzione di attivazione Tangente Iperbolica
"""
def tanh(z):

    return np.tanh(z)

"""
    Derivata della funzione di attivazione Tangente Iperbolica
    g'(z) = 1 - tanh(z)^2

    Args:
        z: Input della funzione di attivazione
    
    Returns:
        Derivata della funzione di attivazione Tangente Iperbolica
"""
def tanh_d(z):

    return 1.0 - np.tanh(z) ** 2

"""
    Derivata della funzione di attivazione ReLU
    1 se z > 0, altrimenti 0
    
    Args:
        z: Input della funzione di attivazione
        
    Returns:
        Derivata della funzione di attivazione ReLU
"""
def relu_d(z):

    return (z > 0).astype(float)