import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
from mlflow.tracking import MlflowClient

# ── MLflow 연결 설정 ──────────────────────────────────────────
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

if "MLFLOW_TRACKING_USERNAME" in os.environ:
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

experiment_name = "iris_classification"
mlflow.set_experiment(experiment_name)

# ── 1. 데이터 로드 ────────────────────────────────────────────
try:
    df = pd.read_csv("data/titanic.csv")
    df = df.select_dtypes(include=['number']).dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("✅ 데이터 로드 및 전처리 완료")
except FileNotFoundError:
    print("❌ 데이터를 찾을 수 없음. dvc pull 확인 필요.")
    exit(1)

# ── 2. 실험 파라미터 ──────────────────────────────────────────
run_results = []
param_list = [
    {"n_estimators": 50,  "max_depth": 2},
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 300, "max_depth": 4},
]

# ── 3. 실험 실행 ──────────────────────────────────────────────
for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}"
    with mlflow.start_run(run_name=run_name):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**params, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        model_info = mlflow.sklearn.log_model(pipe, name="model")

        run_results.append({
            "run_name": run_name,
            "accuracy": acc,
            "model_uri": model_info.model_uri
        })
        print(f"  {run_name}: {acc:.4f} | uri: {model_info.model_uri}")

# ── 4. 최고 모델 선택 ─────────────────────────────────────────
best = max(run_results, key=lambda x: x["accuracy"])
print(f"🏆 최고 모델: {best['run_name']} | accuracy: {best['accuracy']:.4f}")

# ── 5. Model Registry 등록 ────────────────────────────────────
registered = mlflow.register_model(
    model_uri=best["model_uri"],
    name="iris_classifier"
)
print(f"✅ 등록 완료! Version: {registered.version}")

# ── 6. production alias 설정 (주석 해제!) ─────────────────────
client = MlflowClient()
client.set_registered_model_alias(
    name="iris_classifier",
    alias="production",
    version=registered.version
)
print(f"🚀 production alias → Version {registered.version}")
