import gradio as gr
import pickle
import pandas as pd
import numpy as np

# 1. Chargement du modèle
# Grâce à l'étape 'cp' dans le YAML, le modèle est à la racine
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Erreur de chargement du modèle : {e}")

# 2. Fonction de prédiction
def predict_sales(tv, radio, newspaper):
    try:
        # Création du DataFrame avec les noms de colonnes attendus par le modèle
        input_df = pd.DataFrame([[tv, radio, newspaper]], 
                                columns=['tv', 'radio', 'newspaper'])
        
        # Prédiction
        prediction = model.predict(input_df)
        
        # On arrondit à 2 décimales pour la propreté
        result = round(float(prediction[0]), 2)
        return f"Prédiction des ventes : {result} K$"
    except Exception as e:
        return f"Erreur lors de la prédiction : {e}"

# 3. Interface Graphique Gradio
interface = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Budget Pub TV (K$)"),
        gr.Number(label="Budget Pub Radio (K$)"),
        gr.Number(label="Budget Pub Newspaper (K$)")
    ],
    outputs=gr.Textbox(label="Résultat"),
    title="📈 Prédiction du Chiffre d'Affaires",
    description="Entrez vos budgets publicitaires pour obtenir une estimation des ventes générées.",
    examples=[[230.1, 37.8, 69.2], [44.5, 39.3, 45.1]]
)

# 4. Lancement
if __name__ == "__main__":
    interface.launch()