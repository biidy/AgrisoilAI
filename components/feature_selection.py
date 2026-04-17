import pickle
import pandas as pd
import mlrun

def report_importance(context, model_path: mlrun.DataItem):
    # Charger le modèle
    path = model_path.url if hasattr(model_path, 'url') else str(model_path)
    with open(path, 'rb') as f:
        model = pickle.load(f)

    # RÉCUPÉRATION AUTOMATIQUE DES NOMS
    # XGBoost stocke les noms dans 'feature_names_in_' si c'est un modèle Sklearn
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # Si pour une raison X ou Y ils ne sont pas là, on utilise une sécurité
        context.logger.error("Les noms des features ne sont pas dans le modèle !")
        return

    importances = model.feature_importances_
    
    ranking = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Pct': importances * 100
    }).sort_values(by='Importance_Pct', ascending=False)

    # 5. Log dans MLRun (pour que ça apparaisse dans les logs et artefacts)
    context.log_dataset("feature_importance_ranking", df=ranking, index=False)
    
    # Affichage propre dans la console GitHub Actions
    print("\n🏆 CLASSEMENT DES FEATURES :")
    print(ranking.to_string(index=False))
    
    context.logger.info("✅ Analyse des features terminée.")