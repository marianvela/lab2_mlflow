# libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import joblib


# Preprocessing
def preprocess_data(dataset_path, target_column):
    df = pd.read_parquet(dataset_path, engine='pyarrow')
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # columnas numericas y categoricas
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # pipeline de pre procesamiento
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


# definir modelo y modo de busqueda de hiperparametros
def get_model_and_params(model_name):
    if model_name == "RandomForest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
        }
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
        }
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
        }
    elif model_name == "KNN":
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
        }
    elif model_name == "NaiveBayes":
        model = GaussianNB()
        param_grid = {}  # para NaiveBayes probamos sin hiperparametros
    else:
        raise ValueError(f"Modelo no se puede ejecutar: {model_name}")

    return model, param_grid


# train con GridSearch para hiperparametros
def train_model(X, y, model_name, trials):
    model, param_grid = get_model_and_params(model_name)
    if param_grid:  # si hay hiperparametros en el modelo
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        print(f"Mejores parametros para {model_name}: {grid_search.best_params_}")
        return best_model
    else:  # cuando no se tiene hiperparametros
        model.fit(X, y)
        return model


# guardar el modelo, dump a la carpeta de resultados
def save_model(model, model_path="model.pkl"):
    joblib.dump(model, model_path)


# cargar el modelo
def load_model(model_path="model.pkl"):
    return joblib.load(model_path)
