# libraries
import os
import time
import pandas as pd
from fastapi import FastAPI
from automl_pipeline import preprocess_data, train_model, save_model, load_model

app = FastAPI()

# leer variables globales de .env
DATASET = os.getenv("DATASET")
TARGET = os.getenv("TARGET")
MODEL = os.getenv("MODEL")
DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE")
TRIALS = int(os.getenv("TRIALS", 10))
INPUT_FOLDER = os.getenv("INPUT_FOLDER")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
PORT = int(os.getenv("PORT", 8000))

# Batch mode
def batch_mode():
    X, y, preprocessor = preprocess_data(DATASET, TARGET)
    model = train_model(X, y, MODEL, TRIALS)
    save_model(model)
    print("Modelo entrenado y guardado. Inicializando Batch Prediction...")

    while True:
        for file in os.listdir(INPUT_FOLDER):
            if file.endswith(".parquet"):
                input_path = os.path.join(INPUT_FOLDER, file)
                df = pd.read_parquet(input_path)
                X = preprocessor.transform(df)
                predictions = model.predict(X)
                df['predictions'] = predictions
                output_path = os.path.join(OUTPUT_FOLDER, f"{file}_predictions.parquet")
                df.to_parquet(output_path)
                print(f"Archiivo procesado {file}. Resultados guardados en {output_path}.")
        time.sleep(10)

# API mode
@app.post("/predict")
def predict(data: dict):
    model = load_model()
    preprocessor = preprocess_data(DATASET, TARGET)[2]
    X = preprocessor.transform(pd.DataFrame(data['features']))
    predictions = model.predict_proba(X)
    return {"predictions": predictions.tolist()}

# main script
if __name__ == "__main__":
    if DEPLOYMENT_TYPE == "Batch":
        batch_mode()
    elif DEPLOYMENT_TYPE == "API":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=PORT)