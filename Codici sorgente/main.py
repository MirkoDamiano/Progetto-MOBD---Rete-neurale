import pandas as pd
import numpy as np

from preprocessing import *
#from static_network import *
from dynamic_network import *
from cross_validation import *
from config import *
from plot import *

def main():

    print("\nBENVENUTO!\n")

    dataset_choice = print_menu("Scegliere il dataset:\n1. Air Quality\n2. Heart Failure Clinical Records\n")

    if dataset_choice == '1':
        path = "AirQuality.csv"
        unit = "mg/m³"
        target = "CO(GT)"
        epocs = 30
        print("\nHai scelto il dataset: Air Quality")
        X_train, y_train, X_test, y_test = preprocess_data(path)
    else:
        path = "Heart_Failure_Dataset.csv"
        unit = "giorni"
        target = "Tempo di sopravvivenza"
        epocs = 30
        print("\nHai scelto il dataset: Heart Failure Clinical Records")
        X_train, y_train, X_test, y_test = preprocess_heart(path, target_column = 'time')

    activation_choice = print_menu('Funzioni di attivazione disponibili:\n1. ReLU\n2. Tangente Iperbolica\n')

    if activation_choice == '1':
        activation_function = 'relu'
        print("Hai scelto la funzione di attivazione: ReLU")
    else:
        activation_function = 'tanh'
        print("Hai scelto la funzione di attivazione: Tangente Iperbolica")

    regolarization_choice = print_menu('Regolarizzazioni disponibili:\n1. L1\n2. L2\n')

    if regolarization_choice == '1':
        reg = 'l1'
        print("Hai scelto la regolarizzazione: L1")
    else:
        reg = 'l2'
        print("Hai scelto la regolarizzazione: L2")

    try:

        n_0 = X_train.shape[1]  # Numero di features (input layer)
        p = y_train.shape[1]  # Numero di uscite (output layer)

        # Fase di cross-validation
        best_hidden_dims, best_lambda = cross_validation(X_train, y_train, features = n_0, out_neurons = p, activation = activation_function, reg = reg)

        # Inizializzazione della rete neurale
        neural_network = NeuralNetwork(in_neurons = n_0, hidden_dims = best_hidden_dims, out_neurons = p, activation = activation_function)

        # Fase di addestramento
        print(f"Addestramento della rete neurale in corso...")
        train = neural_network.train_sgd(X_train, y_train, alpha = ALPHA, num_epochs = epocs, lambd = best_lambda, reg = reg, verbose = False)

        print("\n--- ADDDESTRAMENTO COMPLETATO ---\n")

        y_pred = neural_network.predict(X_test)

        print(f"--- CONFRONTO PREDIZIONI-CAMPIONE ({target}) ---\n")
        print(f"{'Campione':<10} | {'Valore reale':<15} | {'Valore predetto':<15} | {'Scarto':<10}")
        print("-" * 60)

        for i in range(min(8, len(y_test))):

            reale = y_test[i][0]
            predetto = y_pred[i][0]
            scarto = abs(reale - predetto)
            print(f"{i+1:<10} | {reale:<15.2f} | {predetto:<15.2f} | {scarto:<10.2f} {unit}")

        print("-" * 60 + "\n")

        if dataset_choice == '2':
            dis = np.abs(y_test - y_pred)
            mean_dis = np.mean(dis)
            print(f"Media dei giorni di scarto: {mean_dis:.2f} giorni\n")

        # Plot dei risultati
        plot(train, neural_network, X_test, y_test, unit = unit)

    except FileNotFoundError:
        print(
            f"File '{path}' non trovato!"
        )

def print_menu(message):

    print(message)

    return input("Scegliere un'opzione: ")

if __name__ == "__main__":
    main()
