import numpy as np

from functions import *
from config import *

class NeuralNetwork:

    def __init__(self, in_neurons, hidden_dims = [32, 16], out_neurons = 1, activation = 'relu'):

        # Creiamo un dizionario per memorizzare i pesi e i bias della rete neurale
        self.parameters = {}

        self.L = len(hidden_dims) + 1 # Numero totale di strati

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

        # Inizializziamo le dimensioni degli strati
        layer_dims = [in_neurons] + hidden_dims + [out_neurons] 

        # Inizializziamo neuroni in ingresso e uscita allo strato l e pesi-bias per ogni strato
        for l in range(1, len(layer_dims)):
            neuron_in = layer_dims[l - 1]
            neuron_out = layer_dims[l]

            # Usiamo l'inizializzazione di He per migliorare la convergenza durante l'addestramento
            self.parameters[f'W{l}'] = np.random.randn(neuron_out, neuron_in) * np.sqrt(2. / neuron_in)  
            self.parameters[f'b{l}'] = np.zeros((neuron_out, 1))

    """
        Esegue il forward propagation della rete neurale
    
        Args:
            x: Input della rete neurale
    
        Returns:
            - phi: l'output della rete neurale
            - cache: un dizionario contenente z1, a1, z2, a2, z3, a3 (utili per il back propagation)
    """
    def forward_p(self, x):

        cache = {'a0': x}
        a_prev = x

        # Eseguiamo un ciclo sugli strati nascosti da 1 a L-1
        for l in range(1, self.L):

            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            z = np.dot(W, a_prev) + b
            a = self.activation_function(z)

            cache[f'z{l}'] = z
            cache[f'a{l}'] = a
            a_prev = a

        # Strato di output L
        W = self.parameters[f'W{self.L}']
        b = self.parameters[f'b{self.L}']
        z = np.dot(W, a_prev) + b
        phi = linear(z) 

        cache[f'z{self.L}'] = z
        cache[f'phi'] = phi

        return phi, cache

    """
        Calcola i gradienti della funzione di errore con regolarizzazione

        Args:
            y: Target reale
            cache: Dizionario contenente i valori calcolati nel forward propagation
            lambd: Parametro lambda di regolarizzazione
            reg: Tipo di regolarizzazione

        Returns:
            - grads: un dizionario contenente i gradienti della funzione d'errore 
            - error_loss: il valore della funzione d'errore per il campione corrente
    """
    def back_p(self, y, cache, lambd = 0.0, reg = 'l2'):

        grads = {}
        phi = cache['phi']

        reg_loss = 0.0
        if reg.lower() == 'l2':
            # Funzione d'errore con regolarizzazione L2
            sum_w = sum(np.sum(self.parameters[f'W{l}'] ** 2) for l in range(1, self.L + 1))
            reg_loss = 0.5 * lambd * sum_w 
        elif reg.lower() == 'l1':
            # Funzione d'errore con regolarizzazione L1
            sum_w = sum(np.sum(np.abs(self.parameters[f'W{l}'])) for l in range(1, self.L + 1))
            reg_loss = lambd * sum_w

        error_loss = 0.5 * np.sum((phi - y) ** 2) + reg_loss

        # Strato di output L
        dE_dz = phi - y
        WL = self.parameters[f'W{self.L}']

        if reg.lower() == 'l2':
            reg_grad = lambd * WL
        else:
            reg_grad = lambd * np.sign(WL)
        
        grads[f'dW{self.L}'] = np.dot(dE_dz, cache[f'a{self.L - 1}'].T) + reg_grad
        grads[f'db{self.L}'] = dE_dz

        # Ciclo a ritroso sugli strati nascosti da L-1 a 1
        for l in range(self.L - 1, 0, -1):

            W_next = self.parameters[f'W{l + 1}']
            W_curr = self.parameters[f'W{l}']
            z_curr = cache[f'z{l}']
            a_prev = cache[f'a{l - 1}']

            dE_dz = np.dot(W_next.T, dE_dz) * self.activation_derivative(z_curr)

            if reg.lower() == 'l2':
                reg_grad = lambd * W_curr
            else:
                reg_grad = lambd * np.sign(W_curr)

            grads[f'dW{l}'] = np.dot(dE_dz, a_prev.T) + reg_grad
            grads[f'db{l}'] = dE_dz

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

        # Aggiorniamo i pesi e i bias per ogni strato
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= alpha * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= alpha * grads[f'db{l}']

    """
        Addestramento della rete neurale utilizzando la discesa stocastica del gradiente
        con mini-batch

        Args:
            X_train: Matrice di input di training (N,n0) (N = numero di campioni, n0 = numero di features)
            y_train: Vettore di output (N,1)
            alpha: Passo di apprendimento (learning rate)
            num_epochs: Numero di epoche di addestramento
            lambd: Parametro lambda di regolarizzazione
            reg: Tipo di regolarizzazione
            verbose: Booleano se impostato a True, stampa l'errore medio ogni 5 epoche

        Returns:
            loss_per_epoch: Lista per memorizzare l'errore medio di ogni epoca

    """
    def train_sgd(self, X_train, y_train, alpha, num_epochs, lambd = 0.0, reg = 'l2', verbose = False):

        m = X_train.shape[0]
        loss_per_epoch = [] 

        if verbose:
            print(f"\n--- INIZIO ADDDESTRAMENTO RETE NEURALE ---\n")

        for epoch in range(num_epochs):

            # Facciamo una permutazione casuale ad ogni epoca per il mini-batch
            indices = np.random.permutation(m)
            X_shuffle = X_train[indices]
            y_shuffle = y_train[indices]

            total_loss = 0.0 # Errore totale per l'epoca corrente
            num_batches = int(np.ceil(m / MINI_BATCH_SIZE))

            for batch in range(num_batches):

                start_indices = batch * MINI_BATCH_SIZE
                # In questo modo gestiamo l'ultimo blocco di mini-batch nel caso in cui il numero
                # di campioni non sia divisibile per la dimensione del mini-batch
                end_indices = min(start_indices + MINI_BATCH_SIZE, m)

                X_batch = X_shuffle[start_indices:end_indices]
                y_batch = y_shuffle[start_indices:end_indices]
                curr_batch_size = X_batch.shape[0]

                grads = {}

                batch_loss = 0.0

                for i in range(curr_batch_size):

                    x_i = X_batch[i].reshape(-1, 1)
                    y_i = y_batch[i].reshape(-1, 1)

                    # Forward propagation
                    phi, cache = self.forward_p(x_i)

                    # Back propagation
                    back_grads, error_loss = self.back_p(y_i, cache, lambd, reg)

                    # Salviamo l'errore per il campione corrente
                    batch_loss += error_loss

                    # Accumuliamo i gradienti per il mini-batch
                    if i == 0:
                        for key in back_grads:
                            grads[key] = back_grads[key]
                    else:
                        for key in back_grads:
                            grads[key] += back_grads[key]

                # Calcoliamo la stima stocastica del gradiente: 1/|S_k| * sum_grad(E)
                mean_grads = {key: grads[key] / curr_batch_size for key in grads}

                # Aggiorniamo i pesi e i bias della rete neurale
                self.update_param_sgd(mean_grads, alpha)

                total_loss += batch_loss

            # Errore medio sull' intera epoca
            epoch_loss = total_loss / m
            loss_per_epoch.append(epoch_loss)

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
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
            lambd: Parametro lambda di regolarizzazione
            reg: Tipo di regolarizzazione

        Returns:
            mean_loss: Errore medio
    """
    def evaluation(self, X_test, y_test, lambd = 0.0, reg = 'l2'):

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