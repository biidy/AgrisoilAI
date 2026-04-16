import mlrun
from kfp import dsl

# 1. On définit la fonction du pipeline
@dsl.pipeline(
    name="mada-crop-recommendation",
    description="Pipeline d'entraînement pour la recommandation de cultures"
)
def crop_pipeline(source_url):
    # Charger le projet MLRun actuel
    project = mlrun.get_or_create_project("agrisoil-ai", context="./")

    # Étape 1 : Préparation des données
    prep = mlrun.run_function(
        "data-prep", 
        params={"source_url": source_url},
        outputs=["train_set", "test_set", "region_encoder", "crop_encoder"]
    )

    # Étape 2 : Entraînement (consomme les sorties de prep)
    train = mlrun.run_function(
        "train",
        inputs={"train_set": prep.outputs["train_set"]},
        params={"n_estimators": 100}
    )

    # Étape 3 : Évaluation
    evaluate = mlrun.run_function(
        "evaluate",
        inputs={
            "model_item": train.outputs["model"],
            "test_set": prep.outputs["test_set"]
        }
    )

# 2. Bloc d'exécution principal
if __name__ == "__main__":
    # On définit où est le dataset (local ou URL)
    DATA_URL = "./data/master_dataset_mada_30k_FINAL.csv"
    
    # On lance le pipeline localement pour tester
    run_pipeline(crop_pipeline, arguments={"source_url": DATA_URL}, local=True)