# 1. On prépare les valeurs dans l'ordre (SANS utiliser de dictionnaire manuel pour éviter les erreurs)
# On crée une liste de valeurs dans le même ordre que tes colonnes d'entraînement
valeurs = [
    20.0,    # Altitude
    11,      # Mois
    1,       # Saison
    38.5,    # SOC
    25.0,    # Clay
    30.0,    # Nitrogen
    16.5,    # CEC
    29.0,    # Temp
    88.0,    # Hum
    5.2,     # pH
    52000.0, # Lux
    750.0,   # CE
    le_region.transform(['Atsinanana'])[0] # L'indice de la région
]

# 2. Création du DataFrame avec les noms EXACTS du modèle
nouveau_sol_df = pd.DataFrame([valeurs], columns=X_train.columns)

# 3. Prédiction
probs = model.predict_proba(nouveau_sol_df)[0]

# 4. Affichage du classement
classement = pd.DataFrame({
    'Plante': le_crop.classes_,
    'Probabilité': probs
}).sort_values(by='Probabilité', ascending=False)

print("--- TOP 5 DES RECOMMANDATIONS ---")
print(classement.head(5))