import numpy as np
import copy

from itertools import combinations
from config import *
from dynamic_network import *

"""
    Divide il dataset in k-folds per la cross-validation

    Args:
        X: Matrice di input
        Y: Vettore di output
        k: Numero di folds
        seed: Seed per la generazione casuale degli indici

    Returns:
        folds: Lista di tuple contenenti gli indici di training e validation per ogni fold
"""
def k_fold_division(X, Y, k = 5, seed = 42):

    np.random.seed(seed)

    m = X.shape[0]  # Numero di campioni nel training set
    indices = np.random.permutation(m)  # Vettore casuale di indici
    fold_sizes = m // k # Campioni per ogni fold

    # Lista di vettori di indici per ogni fold, dove ogni elemento è una tupla (train_indices, val_indices)
    folds = []
    for i in range(k):

        start = i * fold_sizes
        # Se siamo all'ultimo fold, prendiamo tutti i campioni rimanenti
        end = start + fold_sizes if i != k - 1 else m
        fold_indices = indices[start:end]

        train_indices = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_indices, fold_indices))

    return folds

"""
    Testa tutte le combinazioni possibili di neuroni e valori di lambda per trovare la miglior configurazione della rete neurale
    che minimizza l'errore sul validation set

    Args:
        X: Matrice di input
        Y: Vettore di output
        features: Numero di neuroni nello strato di ingresso
        out_neurons: Numero di neuroni nello strato di uscita
        activation: Funzione di attivazione
        reg: Tipo di regolarizzazione

    Returns:
        best_config: Tupla contenente la miglior configurazione
"""
def cross_validation(X, Y, features, out_neurons, activation = 'relu', reg = 'l2'):

    folds = k_fold_division(X, Y, k = K_FOLDS)

    # Generiamo tutte le combinazioni possibili di neuroni per L strati nascosti
    neuron_combinations = [list(c) for c in combinations(HIDDEN_LAYER_NEURONS_LIST, L)]

    # Inizializziamo il miglior errore e la miglior configurazione
    best_loss = float('inf')
    best_config = None

    # Cicliamo su ogni combinazione di neuroni per L strati nascosti
    for comb in neuron_combinations:

        # Cicliamo su ogni valore di lambda
        for lambd in LAMBDA_LIST:

            fold_losses = [] # Salviamo l'errore di validazione per ogni fold

            # Cicliamo su ogni fold
            for train_indices, val_indices in folds:

                X_train, Y_train = X[train_indices], Y[train_indices]
                X_val, Y_val = X[val_indices], Y[val_indices]

                model = NeuralNetwork(in_neurons = features, hidden_dims = comb, out_neurons = out_neurons, activation = activation)

                model.train_sgd(X_train, Y_train, alpha = ALPHA, num_epochs = 5, lambd = lambd, reg = reg, verbose = False)

                # Calcoliamo l'errore sul validation set
                val_loss = model.evaluation(X_val, Y_val, lambd = lambd, reg = reg)
                fold_losses.append(val_loss)

            # Calcoliamo l'errore medio sul validation set per questa configurazione
            mean_val_loss = np.mean(fold_losses)

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_config = (comb, lambd)

            if best_config is None:
                print(f"\nATTENZIONE: Nessuna configurazione ha superato la Cross-Validation")
                print(f"\nSi utilizza la configurazione di default: Strati = [16, 32], Lambda = 1e-3")
                return [16, 32], 0.001

    print(f"\nMiglior configurazione trovata:\nStrati nascosti = {best_config[0]}\nLambda = {best_config[1]}\n")

    return best_config[0], best_config[1]
