import pandas as pd  # pandas : pour construire et manipuler notre tableau de données
import numpy as np  # numpy : pour générer des valeurs numériques aléatoires

# On fixe une graine aléatoire (seed) pour garantir la reproductibilité des expériences.
np.random.seed(42)

n = 500  # Nombre de clients fictifs à générer

# On construit le tableau colonne par colonne.
# Chaque colonne est une FEATURE (donnée d'entrée du futur modèle).
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),  # Âge entre 18 et 70 ans
    "salaire": np.random.randint(1200, 8000, n),  # Salaire mensuel simulé
    "abonnement": np.random.choice(["Basique", "Standard", "Premium"], n),  # Type d'offre
    "anciennete": np.random.randint(0, 15, n),  # Ancienneté en années
    "consommation": np.random.randint(0, 100, n),  # Score d'utilisation du service sur 100
})

# On crée maintenant la TARGET (colonne à prédire) : churn.
# Un client peu actif (consommation < 30) ET récent (ancienneté < 2 ans)
# est considéré comme étant à risque de départ.
data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int)  # True/False est converti en 1/0

# On sauvegarde le tableau dans un fichier CSV.
data.to_csv("churn_data.csv", index=False)

# Affichages de contrôle pour vérifier le résultat.
print(data.head())
print(f"Taux de churn : {data['churn'].mean():.1%}")