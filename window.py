import os
import pandas as pd
import glob

def appliquer_window_sliding(dossier_entree, fichier_sortie, window_size=20, seuil=0.55):
    # Liste des noms des fichiers dans le dossier d'entrée
    fichiers = os.listdir(dossier_entree)

    # Créer une liste pour stocker toutes les lignes
    all_lines = []

    # Parcourir tous les fichiers dans le dossier

    for fichier in fichiers:

        # Chemin complet vers le fichier d'entrée
        chemin_fichier = os.path.join(dossier_entree, fichier)

        # Charger le fichier CSV
        data = pd.read_csv(chemin_fichier)

        # Créer une liste pour stocker les lignes de chaque fichier
        lines = []

        # Parcourir les lignes en utilisant le window sliding
        for i in range(len(data) - window_size + 1):
            window = data.iloc[i:i+window_size]
            # Compter le nombre de valeurs inférieures au seuil dans le sixième élément
            count = (window.iloc[:, 5] < seuil).sum()

            # Déterminer si plus de 75% des valeurs sont inférieures au seuil et assigner 0 ou 1 en conséquence
            if count > 0.75 * window_size:
                label = 0
            else:
                label = 1

            #window.drop(columns=[window.columns[5]], inplace=True)
            # Ajouter le window et le label à la liste
            line = window.values.flatten().tolist() + [label]
            lines.append(line)

        # Ajouter les lignes du fichier à la liste de toutes les lignes
        all_lines.extend(lines)

    # Créer un DataFrame à partir de toutes les lignes
    result = pd.DataFrame(all_lines)

    # Sauvegarder le DataFrame résultant dans un fichier CSV
    result.to_csv(fichier_sortie, index=False, header=False)
#appliquer_window_sliding("video csv/", "fenetress.csv", window_size=20, seuil=0.55)
data = pd.read_csv("fenetres - Copie.csv")

columns_to_drop = []
for i in range(5, 20*13, 13):
    columns_to_drop.append(data.columns[i])


data = data.drop(columns=columns_to_drop)