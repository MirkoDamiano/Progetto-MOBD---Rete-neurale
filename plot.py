import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
    Genera i grafici per l'analisi delle prestazioni della rete neurale

    Args:
        train: Lista per memorizzare l'errore medio di ogni epoca
        model: Rete neurale
        X_test: Matrice di input di test (N,n0)
        y_test: Vettore di output (N,1)
        unit: Unità di misura in funzione del dataset scelto

    Returns:
        None
"""
def plot(train, model, X_test, y_test, unit = ""):

    plt.figure(figsize = (12, 5))

    train_s = np.sqrt(2 * np.array(train))

    # PRIMO GRAFICO: Curva di apprendimento (Training loss per epoca)
    plt.subplot(1, 2, 1)
    epocs = np.arange(1, len(train) + 1)
    plt.plot(epocs, train_s, color = 'blue', linewidth = 2)

    points = [0] + list(range(4, len(train), 5))

    x = epocs[points]
    y = train_s[points]

    plt.scatter(x, y, color = 'red', s = 40, zorder = 5)

    plt.title('Curva di apprendimento', fontsize = 14)
    plt.xlabel('Epoca', fontsize = 12)
    plt.ylabel('Errore medio', fontsize = 12)
    plt.grid(True)

    # SECONDO GRAFICO: Predizioni vs Valori reali sul Test set
    y_pred = model.predict(X_test)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, color = 'green', alpha = 0.5)

    # Aggiungiamo una linea di riferimento y = x per indicare l'uguaglianza tra predizioni e valori reali
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color = 'red', linestyle = '--', linewidth = 2, label = 'y = x')

    plt.title('Predizioni vs Valori reali sul Test set', fontsize = 14)
    plt.xlabel('Valori reali', fontsize = 12)
    plt.ylabel('Predizioni', fontsize = 12)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()