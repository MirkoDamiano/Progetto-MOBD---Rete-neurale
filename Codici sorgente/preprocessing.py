import pandas as pd
import numpy as np

def preprocess_data(path, target_column = 'CO(GT)'):

    # Il dataset Air Quality usa come separatore il punto e virgola e la virgola come separatore decimale. 
    # Inoltre, i valori mancanti sono rappresentati da -200
    dataset = pd.read_csv(path, sep = ';', decimal = ',', na_values = -200)

    # Si eliminano tutte le colonne che contengono solo valori NaN 
    # e quelle relative a 'Date' e 'Time'
    dataset = dataset.dropna(how = 'all', axis = 1).reset_index(drop = True)
    dataset = dataset.drop(columns = ['Date', 'Time'], errors = 'ignore')

    # Si sostituiscono i valori NaN con la media della colonna corrispondente   
    dataset = dataset.fillna(dataset.mean())

    # Separazione feature e label
    X = dataset.drop(columns = [target_column]).values # Matrice di features (N, n0) 
    y = dataset[target_column].values.reshape(-1, 1) # Vettore di label (N, p)

    # Utilizziamo un seed per rendere i risultati riproducibili
    np.random.seed(42)

    # Poichè il dataset raccoglie misurazioni in ordine cronologico, è importante mescolare i dati 
    # per evitare che il modello impari sequenze temporali specifiche e i set siano bilanciati

    # Creiamo un array di indici casuali per mescolare i dati
    num_campioni = X.shape[0]
    indices = np.arange(num_campioni)
    np.random.shuffle(indices)

    # Dividiamo i dati in Training set (80%) e Test set (20%)
    train_size = int(0.8 * num_campioni)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Selezioniamo i dati di training e di test utilizzando gli indici generati
    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    # Calcoliamo la media e la deviazione standard delle features del training set per la normalizzazione
    media = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)

    eps = 1e-8 # Piccola costante per evitare la divisione per zero

    # Standardizzazione dei dati di training e test set (media = 0, deviazione standard = 1)
    X_train = (X_train - media) / (std + eps)
    X_test = (X_test - media) / (std + eps)

    return X_train, y_train, X_test, y_test

'''
    Restituisce la colonna specificata dall'indice sia per le features (X) che per le label (y).
'''
def get_column(X, y, index):

    x_i = X[index].reshape(-1, 1)  # Colonna delle features (n0, 1)
    y_i = y[index].reshape(-1, 1)  # Colonna delle label (p, 1)

    return x_i, y_i

def preprocess_heart(path, target_column = 'time'):

    dataset = pd.read_csv(path)

    # Separazione feature e label
    X = dataset.drop(columns = [target_column]).values.astype(np.float64) # Matrice di features (N, n0) 
    y = dataset[target_column].values.reshape(-1, 1).astype(np.float64) # Vettore di label (N, p)

    # Utilizziamo un seed per rendere i risultati riproducibili
    np.random.seed(42)

    # Creiamo un array di indici casuali per mescolare i dati
    num_campioni = X.shape[0]
    indices = np.arange(num_campioni)
    np.random.shuffle(indices)

    # Dividiamo i dati in Training set (80%) e Test set (20%)
    train_size = int(0.8 * num_campioni)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Selezioniamo i dati di training e di test utilizzando gli indici generati
    X_train = X[train_indices]
    y_train = y[train_indices]
    
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Calcoliamo la media e la deviazione standard delle features del training set per la normalizzazione
    media = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)

    eps = 1e-8 # Piccola costante per evitare la divisione per zero

    # Standardizzazione dei dati di training e test set (media = 0, deviazione standard = 1)
    X_train = (X_train - media) / (std + eps)
    X_test = (X_test - media) / (std + eps)

    return X_train, y_train, X_test, y_test
