import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        logger.info("Preparing features...")

        df_processed = df.copy()

        if "Time" in df_processed.columns:
            df_processed["hour_of_day"] = (df_processed["Time"] % (24 * 3600)) / 3600

            df_processed["hour_sin"] = np.sin(
                2 * np.pi * df_processed["hour_of_day"] / 24
            )
            df_processed["hour_cos"] = np.cos(
                2 * np.pi * df_processed["hour_of_day"] / 24
            )

            df_processed["is_night"] = (
                (df_processed["hour_of_day"] >= 22) | (df_processed["hour_of_day"] <= 6)
            ).astype(int)

        if "Amount" in df_processed.columns:
            df_processed["Amount_log"] = np.log1p(df_processed["Amount"])
            df_processed["is_high_amount"] = (df_processed["Amount"] > 200).astype(int)

        v_cols = [c for c in df_processed.columns if c.startswith("V")]
        if v_cols:
            df_processed["V_mean"] = df_processed[v_cols].mean(axis=1)
            df_processed["V_std"] = df_processed[v_cols].std(axis=1)
            df_processed["V_outlier_count"] = (df_processed[v_cols].abs() > 3).sum(
                axis=1
            )

        feature_cols = [f"V{i}" for i in range(1, 29)]
        feature_cols.extend(
            [
                "Amount_log",
                "hour_of_day",
                "hour_sin",
                "hour_cos",
                "is_night",
                "is_high_amount",
                "V_mean",
                "V_std",
                "V_outlier_count",
            ]
        )

        feature_cols = [col for col in feature_cols if col in df_processed.columns]

        X = df_processed[feature_cols]
        y = df_processed["Class"]

        self.feature_names = feature_cols
        logger.info(f"Features prepared: {len(feature_cols)} features")
        logger.info(f"Features: {feature_cols}")

        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        logger.info("Starting model training...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Fraud ratio in train: {y_train.mean():.4f}")
        logger.info(f"Fraud ratio in test: {y_test.mean():.4f}")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        logger.info(f"Class weights: {class_weight_dict}")

        logger.info("Training Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        self.model.fit(X_train_scaled, y_train)

        logger.info("Model training completed!")

        self.evaluate(X_test_scaled, y_test, X_train_scaled, y_train)

        return self.model

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        logger.info("Evaluating model...")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = self.model.score(X_test, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        self.metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
        }

        logger.info("=" * 60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"ROC-AUC:   {roc_auc:.4f}")
        logger.info("=" * 60)

        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")

        if X_train is not None and y_train is not None:
            train_accuracy = self.model.score(X_train, y_train)
            logger.info(f"\nTraining Accuracy: {train_accuracy:.4f}")
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info(f"Overfit Check: {train_accuracy - accuracy:.4f}")

        self.plot_feature_importance()

        self.plot_roc_curve(y_test, y_pred_proba)

        self.plot_confusion_matrix(cm)

    def plot_feature_importance(self):
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 6))
            plt.title("Feature Importance", fontsize=14, fontweight="bold")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(
                range(len(importances)),
                [self.feature_names[i] for i in indices],
                rotation=45,
                ha="right",
            )
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.tight_layout()

            Path("model/plots").mkdir(parents=True, exist_ok=True)
            plt.savefig("model/plots/feature_importance.png", dpi=300)
            plt.close()
            logger.info("Feature importance plot saved")

    def plot_roc_curve(self, y_test, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            "Receiver Operating Characteristic (ROC) Curve",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig("model/plots/roc_curve.png", dpi=300)
        plt.close()
        logger.info("ROC curve plot saved")

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
        )
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        plt.savefig("model/plots/confusion_matrix.png", dpi=300)
        plt.close()
        logger.info("Confusion matrix plot saved")

    def save_model(self, path: str = "model/saved/fraud_detector.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

        metrics_path = Path(path).parent / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")

    @classmethod
    def load_model(cls, path: str = "model/saved/fraud_detector.pkl"):
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        instance = cls()
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.label_encoder = model_data["label_encoder"]
        instance.feature_names = model_data["feature_names"]
        instance.metrics = model_data["metrics"]

        logger.info(f"Model loaded from {path}")
        return instance


def main():
    data_path = Path("data/raw/transactions.csv")
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.info("Please run: python scripts/prepare_dataset.py")
        return

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} transactions")

    model = FraudDetectionModel()

    X, y = model.prepare_features(df)

    model.train(X, y)

    model.save_model()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
