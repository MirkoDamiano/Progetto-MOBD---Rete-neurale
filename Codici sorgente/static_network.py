import numpy as np

from functions import *

class NeuralNetwork:

    def __init__(self, in_neurons, hidden1_dim = 32, hidden2_dim = 16, out_neurons = 1, activation = 'relu'):

        # Creiamo un dizionario per memorizzare i pesi e i bias della rete neurale
        self.parameters = {}

        # Utilizziamo un seed per rendere i risultati riproducibili
        np.random.seed(42)

        # In questo modo abbiamo una gestione dinamica della funzione di attivazione
        self.activation = activation.lower()  # Convertiamo l'argomento in minuscolo
        if self.activation == 'relu':
            self.activation_function = relu
            self.activation_derivative = relu_d

        elif self.activation == 'tanh':
            self.activation_function = tanh
            self.activation_derivative = tanh_d

        else:
            raise ValueError("Funzione di attivazione non valida!")

        # PRIMO STRATO (Input Layer -> Hidden Layer 1)
        # Per la matrice di pesi W1 si utilizza l'inizializzazione di He per migliorare la convergenza durante l'addestramento
        self.parameters['W1'] = np.random.randn(hidden1_dim, in_neurons) * np.sqrt(2. / in_neurons)  # Pesi del primo strato
        self.parameters['b1'] = np.zeros((hidden1_dim, 1))  # Bias del primo strato

        # SECONDO STRATO (Hidden Layer 1 -> Hidden Layer 2)
        self.parameters['W2'] = np.random.randn(hidden2_dim, hidden1_dim) * np.sqrt(2. / hidden1_dim)  # Pesi del secondo strato
        self.parameters['b2'] = np.zeros((hidden2_dim, 1))  # Bias del secondo strato

        # TERZO STRATO (Hidden Layer 2 -> Output Layer)
        self.parameters['W3'] = np.random.randn(out_neurons, hidden2_dim) * np.sqrt(2. / hidden2_dim)  # Pesi del terzo strato
        self.parameters['b3'] = np.zeros((out_neurons, 1))  # Bias del terzo strato

    """
        Esegue il forward propagation della rete neurale
    
        Args:
            x: Input della rete neurale
    
        Returns:
            - a3: l'output della rete neurale
            - cache: un dizionario contenente z1, a1, z2, a2, z3, a3 (utili per il back propagation)
    """
    def forward_p(self, x):

        # Poniamo a0 = x
        a0 = x

        # PRIMO STRATO (Input Layer -> Hidden Layer 1)
        # z1 = W1 * a0 + b1
        z1 = np.dot(self.parameters['W1'], a0) + self.parameters['b1']
        a1 = self.activation_function(z1)

        # SECONDO STRATO (Hidden Layer 1 -> Hidden Layer 2)
        # z2 = W2 * a1 + b2
        z2 = np.dot(self.parameters['W2'], a1) + self.parameters['b2']
        a2 = self.activation_function(z2)

        # TERZO STRATO (Hidden Layer 2 -> Output Layer)
        # z3 = W3 * a2 + b3
        z3 = np.dot(self.parameters['W3'], a2) + self.parameters['b3']
        phi = linear(z3) # phi = a3 

        # Memorizziamo i valori intermedi in un dizionario per il back propagation
        cache = {
            'a0': a0,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'phi': phi
        }

        return phi, cache

    """
        Calcola i gradienti della funzione di errore E = 0.5 * ||phi - y||^2

        Args:
            y: Target reale
            cache: Dizionario contenente i valori calcolati nel forward propagation

        Returns:
            - grads: un dizionario contenente i gradienti della funzione d'errore 
    """
    def back_p(self, y, cache):

        grads = {}
        phi = cache['phi']

        # Funzione d'errore
        error_loss = 0.5 * np.sum((phi - y) ** 2)

        # TERZO STRATO (Hidden Layer 2 -> Output Layer)
        # dE/dz3 = (phi - y)
        dE_dz3 = phi - y

        # Gradienti dei pesi e bias del terzo strato
        grads['dW3'] = np.dot(dE_dz3, cache['a2'].T)
        grads['db3'] = dE_dz3

        # SECONDO STRATO (Hidden Layer 1 -> Hidden Layer 2)
        # dE/dz2 = (W3^T * dE/dz3) * g'(z2)
        dE_dz2 = np.dot(self.parameters['W3'].T, dE_dz3) * self.activation_derivative(cache['z2'])

        # Gradienti dei pesi e bias del secondo strato
        grads['dW2'] = np.dot(dE_dz2, cache['a1'].T)
        grads['db2'] = dE_dz2

        # PRIMO STRATO (Input Layer -> Hidden Layer 1)
        # dE/dz1 = (W2^T * dE/dz2) * g'(z1)
        dE_dz1 = np.dot(self.parameters['W2'].T, dE_dz2) * self.activation_derivative(cache['z1'])

        # Gradienti dei pesi e bias del primo strato
        grads['dW1'] = np.dot(dE_dz1, cache['a0'].T)
        grads['db1'] = dE_dz1

        return grads, error_loss

    """
        Aggiorna i pesi ed i bias della rete neurale utilizzando la discesa stocastica del gradiente

        Args:
            grads: Dizionario restituito dalla funzione back_p contenente i gradienti per il campione 
            alpha: Passo di apprendimento (learning rate)
        Returns:
            None
    """
    def update_param_sgd(self, grads, alpha):

        # PRIMO STRATO (Input Layer -> Hidden Layer 1)
        # W1 = W1 - alpha * dW1
        # b1 = b1 - alpha * db1
        self.parameters['W1'] -= alpha * grads['dW1']
        self.parameters['b1'] -= alpha * grads['db1']

        # SECONDO STRATO (Hidden Layer 1 -> Hidden Layer 2)
        # W2 = W2 - alpha * dW2
        # b2 = b2 - alpha * db2
        self.parameters['W2'] -= alpha * grads['dW2']
        self.parameters['b2'] -= alpha * grads['db2']

        # TERZO STRATO (Hidden Layer 2 -> Output Layer)
        # W3 = W3 - alpha * dW3
        # b3 = b3 - alpha * db3
        self.parameters['W3'] -= alpha * grads['dW3']
        self.parameters['b3'] -= alpha * grads['db3']

    """
        Addestramento della rete neurale utilizzando la discesa stocastica del gradiente

        Args:
            X_train: Matrice di input di training (N,n0) (N = numero di campioni, n0 = numero di features)
            y_train: Vettore di output (N,1)
            alpha: Passo di apprendimento (learning rate)
            num_epochs: Numero di epoche di addestramento

        Returns:
            loss_per_epoch: Lista per memorizzare l'errore medio di ogni epoca

    """
    def train_sgd(self, X_train, y_train, alpha, num_epochs):

        campioni = X_train.shape[0]
        loss_per_epoch = [] 

        print(f"\n--- INIZIO ADDDESTRAMENTO RETE NEURALE ---\n")

        for epoch in range(num_epochs):
            total_loss = 0.0 # Errore totale per l'epoca corrente

            for i in range(campioni):

                x_i = X_train[i].reshape(-1, 1)
                y_i = y_train[i].reshape(-1, 1)

                # Forward propagation
                phi, cache = self.forward_p(x_i)

                # Back propagation
                grads, error_loss = self.back_p(y_i, cache)

                # Aggiornamento dei parametri
                self.update_param_sgd(grads, alpha)

                # Salviamo l'errore per il campione corrente
                total_loss += error_loss

            # Alla fine dell'epoca, calcoliamo l'errore medio e lo salviamo nella lista
            epoch_loss = total_loss / campioni

            loss_per_epoch.append(epoch_loss)

            if(epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoca: {epoch + 1}/{num_epochs} - Errore medio: {epoch_loss:.4f}")

        return loss_per_epoch

    """
        Funzione che esegue solo il passo di forward propagation a pesi fissati

        Args:
            X_test: Matrice di input di test (N,n0)

        Returns:
            predictions: Vettore di predizioni della rete neurale
    """
    def predict(self, X_test):

        campioni = X_test.shape[0]
        predictions = np.zeros((campioni, 1)) # Array per memorizzare le predizioni della rete neurale

        for i in range(campioni):

            x_i = X_test[i].reshape(-1, 1)

            # Forward propagation
            phi, _ = self.forward_p(x_i)

            # Salviamo la predizione scalare per il campione corrente
            predictions[i] = phi[0, 0]

        return predictions

    """
        Calcola la media dell'errore quadratico, E = 0.5 * ||phi - y||^2, sul set di dati di test

        Args:
            X_test: Matrice di input di test (N,n0)
            y_test: Vettore di output (N,1)

        Returns:
            mean_loss: Errore medio
    """
    def evaluation(self, X_test, y_test):

        campioni = X_test.shape[0]
        total_loss = 0.0

        for i in range(campioni):

            x_i = X_test[i].reshape(-1, 1)
            y_i = y_test[i].reshape(-1, 1)

            # Forward propagation
            phi, _ = self.forward_p(x_i)

            # Calcoliamo l'errore per il campione corrente
            loss = 0.5 * np.sum((phi - y_i) ** 2)
            total_loss += loss

        # Calcoliamo l'errore medio
        mean_loss = total_loss / campioni

        return mean_loss
