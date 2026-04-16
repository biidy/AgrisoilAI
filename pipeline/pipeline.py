from mlrun import mount_v3io, p_runtimes, run_pipeline
import kfp.dsl as dsl

@dsl.pipeline(name="Mada Crop Training", description="Pipeline pour le projet de Sarobidy")
def pipeline(source_url):
    # Étape 1 : Préparation
    prep = mlrun.run_function("data_prep", params={"source_url": source_url}, outputs=["train_set", "test_set"])
    
    # Étape 2 : Entraînement (utilise le train_set de l'étape 1)
    train = mlrun.run_function("train", inputs={"train_set": prep.outputs["train_set"]})