"""
Module de modèles de machine learning.

Gère l'entraînement, l'évaluation et la prédiction avec
XGBoost et LightGBM pour la classification de trading.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from monitoring import FeatureMonitor, ModelMonitor
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import os

class MLModel:
    """Gestionnaire de modèles de machine learning."""
    
    def __init__(
        self,
        config: dict,
        model_type: str = 'xgboost',
        random_state: int = 42,
        label_encoder: Optional[LabelEncoder] = None
    ):
        """
        Initialise le modèle ML.
        
        Args:
            config: Configuration du modèle
            model_type: Type de modèle ('xgboost' ou 'lightgbm')
            random_state: Seed aléatoire
            label_encoder: Encoder pour les labels textuels (optionnel)
        """
        self.config = config
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.calibrator = None
        self.feature_importance = None
        self.feature_monitor = FeatureMonitor(config)
        self.model_monitor = ModelMonitor(config)
        self.last_training_metrics = None
        self.columns_ = None  # Pour sauvegarder les colonnes utilisées lors du fit
        self.label_encoder = label_encoder  # Pour gérer les labels textuels
        
        # Paramètres par défaut
        self.default_params = {
            'xgboost': {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'tree_method': 'hist',
                'random_state': random_state,
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0
            },
            'lightgbm': {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'random_state': random_state,
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0,
                'reg_lambda': 0
            }
        }
        
    def prepare_data(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            features: DataFrame des features
            labels: Série des labels
            test_size: Taille du set de test
            val_size: Taille du set de validation
            
        Returns:
            Tuple (features, labels) pour train/val/test
        """
        # Vérification de l'alignement
        if not features.index.equals(labels.index):
            raise ValueError("Les index des features et labels ne correspondent pas")
            
        # Split temporel
        n_samples = len(features)
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Features
        X = {
            'train': features.iloc[:val_idx],
            'val': features.iloc[val_idx:test_idx],
            'test': features.iloc[test_idx:]
        }
        
        # Labels
        y = {
            'train': labels.iloc[:val_idx],
            'val': labels.iloc[val_idx:test_idx],
            'test': labels.iloc[test_idx:]
        }
        
        return X, y
        
    def train(
        self,
        X: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        y: Union[pd.Series, Dict[str, pd.Series]],
        params: Optional[Dict] = None,
        early_stopping_rounds: int = 50
    ) -> Dict:
        """
        Entraîne le modèle.
        
        Args:
            X: DataFrame des features ou dictionnaire des features (train/val/test)
            y: Série des labels ou dictionnaire des labels (train/val/test)
            params: Paramètres du modèle
            early_stopping_rounds: Rounds d'early stopping
            
        Returns:
            Dictionnaire des métriques d'entraînement
        """
        # Gestion des données d'entrée
        if isinstance(X, dict) and isinstance(y, dict):
            X_train = X['train']
            y_train = y['train']
            X_val = X.get('val')
            y_val = y.get('val')
        else:
            X_train = X
            y_train = y
            X_val = None
            y_val = None
            
        # Gestion des labels textuels
        if self.label_encoder is not None:
            y_train = self.label_encoder.transform(y_train)
            if y_val is not None:
                y_val = self.label_encoder.transform(y_val)
        elif y_train.dtype == 'object' or y_train.dtype == 'string':
            # Créer un label encoder automatiquement si les labels sont textuels
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            if y_val is not None:
                y_val = self.label_encoder.transform(y_val)
            logger.info(f"Label encoder créé automatiquement. Classes: {self.label_encoder.classes_}")
            
        # Sauvegarder les colonnes pour validation future
        self.columns_ = X_train.columns.tolist()
        
        # Calcul des statistiques de référence pour le monitoring
        self.feature_monitor.compute_reference_stats(X_train)
        
        # Paramètres du modèle
        if params is None:
            params = self.default_params[self.model_type]
        n_classes = len(np.unique(y_train))
        params['num_class'] = n_classes

        if self.model_type == 'xgboost':
            self.model = XGBClassifier(**params)
            if X_val is not None and y_val is not None:
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                self.calibrator = CalibratedClassifierCV(
                    estimator=self.model,
                    method='isotonic'
                )
                self.calibrator.fit(X_val, y_val)
            else:
                self.model.fit(X_train, y_train, verbose=False)

        elif self.model_type == 'lightgbm':
            self.model = LGBMClassifier(**params)
            if X_val is not None and y_val is not None:
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                self.calibrator = CalibratedClassifierCV(
                    estimator=self.model,
                    method='isotonic'
                )
                self.calibrator.fit(X_val, y_val)
            else:
                self.model.fit(X_train, y_train, verbose=False)
            
        # Importance des features
        self._compute_feature_importance(X_train.columns)
        
        # Métriques d'entraînement avec probabilités calibrées
        if self.calibrator is not None:
            y_pred_train = self.calibrator.predict(X_train)
            y_proba_train = self.calibrator.predict_proba(X_train)
        else:
            y_pred_train = self.model.predict(X_train)
            y_proba_train = self.model.predict_proba(X_train)
            
        metrics = self.evaluate(X_train, y_train, y_pred_train, y_proba_train)
        
        # Add validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            if self.calibrator is not None:
                y_pred_val = self.calibrator.predict(X_val)
                y_proba_val = self.calibrator.predict_proba(X_val)
            else:
                y_pred_val = self.model.predict(X_val)
                y_proba_val = self.model.predict_proba(X_val)
                
            val_metrics = self.evaluate(X_val, y_val, y_pred_val, y_proba_val)
            # Prefix validation metrics
            val_metrics_prefixed = {f'val_{k}': v for k, v in val_metrics.items()}
            metrics.update(val_metrics_prefixed)
        
        # Mise à jour du monitoring
        self.model_monitor.update_metrics(
            metrics,
            f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Vérification des performances
        alerts = self.model_monitor.check_performance(
            metrics,
            self.model_type
        )
        
        if alerts:
            logger.warning(f"Alertes de performance détectées: {len(alerts)}")
            for alert in alerts:
                logger.warning(f"Alerte: {alert}")
        
        # Log dans MLflow
        self.model_monitor.log_to_mlflow(
            metrics,
            self.model_type,
            f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Sauvegarde des métriques
        self.last_training_metrics = metrics
        
        return metrics
        
    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = False,
        check_drift: bool = True,
        check_performance: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prédit les labels pour les données fournies.
        
        Args:
            X: DataFrame des features
            return_proba: Retourner les probabilités
            check_drift: Vérifier le drift des features
            check_performance: Vérifier la performance
            
        Returns:
            Labels prédits et/ou probabilités
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant la prédiction")
            
        # Validation des colonnes
        if self.columns_ is not None:
            if not X.columns.equals(pd.Index(self.columns_)):
                raise ValueError("Les colonnes d'entrée ne correspondent pas à celles utilisées lors du fit.")
        else:
            logger.warning("Aucune information sur les colonnes d'entraînement disponible")
            
        # Feature drift detection
        if self.feature_monitor is not None:
            drift_scores = self.feature_monitor.check_drift(X)
            # drift_scores already contains alerts information
            
        # Utiliser le calibrateur si présent
        if self.calibrator is not None:
            if return_proba:
                proba = self.calibrator.predict_proba(X)
                predictions = np.argmax(proba, axis=1)
                # Décoder les labels si nécessaire
                if self.label_encoder is not None:
                    predictions = self.label_encoder.inverse_transform(predictions)
                return predictions, proba
            else:
                predictions = self.calibrator.predict(X)
                # Décoder les labels si nécessaire
                if self.label_encoder is not None:
                    predictions = self.label_encoder.inverse_transform(predictions)
                return predictions
        else:
            # Fallback sur le modèle de base
            if return_proba:
                proba = self.model.predict_proba(X)
                predictions = np.argmax(proba, axis=1)
                # Décoder les labels si nécessaire
                if self.label_encoder is not None:
                    predictions = self.label_encoder.inverse_transform(predictions)
                return predictions, proba
            else:
                predictions = self.model.predict(X)
                # Décoder les labels si nécessaire
                if self.label_encoder is not None:
                    predictions = self.label_encoder.inverse_transform(predictions)
                return predictions
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retourne les probabilités prédites pour chaque classe.
        Args:
            X: DataFrame des features
        Returns:
            np.ndarray: Probabilités pour chaque classe
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant la prédiction")
        # Validation des colonnes
        if self.columns_ is not None:
            if not X.columns.equals(pd.Index(self.columns_)):
                raise ValueError("Les colonnes d'entrée ne correspondent pas à celles utilisées lors du fit.")
        else:
            logger.warning("Aucune information sur les colonnes d'entraînement disponible")
        # Utiliser le calibrateur si présent
        if self.calibrator is not None:
            proba = self.calibrator.predict_proba(X)
        else:
            proba = self.model.predict_proba(X)
        # Si label_encoder est présent, réordonner les colonnes pour correspondre à l'ordre des classes
        if self.label_encoder is not None:
            # On suppose que proba.shape[1] == len(self.label_encoder.classes_)
            # Rien à faire si l'ordre est correct
            pass
        return proba
                
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Évalue le modèle sur les données fournies.
        
        Args:
            X: DataFrame des features
            y: Série des labels
            y_pred: Prédictions pré-calculées (optionnel)
            y_proba: Probabilités pré-calculées (optionnel)
            
        Returns:
            Dictionnaire des métriques
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant l'évaluation")
            
        # Prédictions
        if y_pred is not None and y_proba is not None:
            # Utiliser les prédictions et probabilités fournies
            pass
        else:
            # Calculer les prédictions et probabilités
            y_pred, y_proba = self.predict(X, return_proba=True)
        
        # Métriques de base
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC pour chaque classe
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                # Multiclass ROC-AUC
                metrics['roc_auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
                metrics['roc_auc_ovo'] = roc_auc_score(y, y_proba, multi_class='ovo', average='weighted')
                
                # ROC-AUC pour chaque classe individuellement
                for i in range(y_proba.shape[1]):
                    if i in y.values:  # Only if class exists in data
                        y_binary = (y == i).astype(int)
                        metrics[f'roc_auc_class_{i}'] = roc_auc_score(y_binary, y_proba[:, i])
            else:
                # Binary or single class
                metrics['roc_auc'] = roc_auc_score(y, y_proba)
        except Exception as e:
            logger.warning(f"Erreur lors du calcul du ROC-AUC: {e}")
            metrics['roc_auc'] = None
        
            # Matrice de confusion
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred)
        
        # Rapport de classification détaillé
        try:
            report = classification_report(y, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
            
            # Extraire les métriques par classe
            for class_name, class_metrics in report.items():
                if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                    metrics[f'precision_class_{class_name}'] = class_metrics['precision']
                    metrics[f'recall_class_{class_name}'] = class_metrics['recall']
                    metrics[f'f1_class_{class_name}'] = class_metrics['f1-score']
                    metrics[f'support_class_{class_name}'] = class_metrics['support']
        except Exception as e:
            logger.warning(f"Erreur lors du calcul du rapport de classification: {e}")
        
        # Statistiques des classes
        class_counts = np.bincount(y_pred)
        metrics['predicted_class_distribution'] = {
            f'class_{i}': count for i, count in enumerate(class_counts) if count > 0
        }
        
        # Métriques spécifiques pour trading
        if len(np.unique(y)) >= 3:  # Multi-class (buy/sell/hold)
            # Précision pour les signaux de trading (classes 1 et 2)
            trading_signals_mask = (y_pred == 1) | (y_pred == 2)
            if np.any(trading_signals_mask):
                trading_accuracy = accuracy_score(y[trading_signals_mask], y_pred[trading_signals_mask])
                metrics['trading_signals_accuracy'] = trading_accuracy
                
                # Ratio de signaux prédits vs réels
                predicted_signals = np.sum(trading_signals_mask)
                actual_signals = np.sum((y == 1) | (y == 2))
                metrics['signal_prediction_ratio'] = predicted_signals / len(y) if len(y) > 0 else 0
                metrics['actual_signal_ratio'] = actual_signals / len(y) if len(y) > 0 else 0
            
        return metrics
        
    def _compute_feature_importance(self, feature_names: List[str]) -> None:
        """
        Calcule l'importance des features.
        
        Args:
            feature_names: Liste des noms de features
        """
        if self.model_type == 'xgboost':
            importance_dict = self.model.get_booster().get_score(importance_type='gain')
            # Map importance_dict to the feature_names order, fill missing with 0
            importance = [importance_dict.get(f, 0.0) for f in feature_names]
            self.feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)
            
        elif self.model_type == 'lightgbm':
            importance = self.model.feature_importances_
            self.feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)
            
    def plot_feature_importance(
        self,
        top_n: int = 20,
        title: str = "Importance des Features"
    ) -> None:
        """
        Visualise l'importance des features.
        
        Args:
            top_n: Nombre de features à afficher
            title: Titre du graphique
        """
        if self.feature_importance is None:
            raise ValueError("L'importance des features n'a pas été calculée")
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=self.feature_importance.head(top_n).values,
            y=self.feature_importance.head(top_n).index
        )
        plt.title(title)
        plt.xlabel("Importance (Gain)")
        plt.tight_layout()
        plt.show()
        
    def log_to_mlflow(
        self,
        metrics: Dict,
        params: Dict,
        run_name: Optional[str] = None
    ) -> None:
        """
        Log les résultats dans MLflow.
        
        Args:
            metrics: Métriques d'évaluation
            params: Paramètres du modèle
            run_name: Nom de la run
        """
        with mlflow.start_run(run_name=run_name):
            # Log des paramètres
            mlflow.log_params(params)
            
            # Log des métriques
            for set_name, set_metrics in metrics.items():
                for metric_name, value in set_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{set_name}_{metric_name}", value)
                        
            # Log du modèle
            if self.model_type == 'xgboost':
                mlflow.xgboost.log_model(self.model, "model")
            elif self.model_type == 'lightgbm':
                mlflow.lightgbm.log_model(self.model, "model")
                
            # Log de l'importance des features
            if self.feature_importance is not None:
                importance_df = self.feature_importance.reset_index()
                importance_df.columns = ['feature', 'importance']
                mlflow.log_table(importance_df, "feature_importance.json")
                
            # Log des statistiques de référence
            if hasattr(self.feature_monitor, 'reference_stats'):
                mlflow.log_dict(
                    self.feature_monitor.reference_stats,
                    "feature_reference_stats.json"
                )
                
    def get_model_path(self, symbol: str, timeframe: str) -> str:
        """
        Construit le chemin du modèle pour un symbole et timeframe donnés.
        
        Args:
            symbol: Symbole de trading (ex: 'EURUSD')
            timeframe: Timeframe (ex: 'M15')
            
        Returns:
            Chemin du fichier modèle
        """
        models_dir = self.config.get('paths', {}).get('models', 'models')
        model_filename = f"{symbol}_{timeframe}_model.joblib"
        return os.path.join(models_dir, model_filename)
                
    def save_model(self, path: str) -> None:
        """
        Sauvegarde le modèle avec toutes les informations nécessaires.
        
        Args:
            path: Chemin de sauvegarde
        """
        import joblib
        
        model_data = {
            'model': self.model,
            'calibrator': self.calibrator,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'config': self.config,
            'columns_': self.columns_,  # Sauvegarder les colonnes
            'random_state': self.random_state,
            'last_training_metrics': self.last_training_metrics,
            'label_encoder': self.label_encoder  # Sauvegarder le label encoder
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Modèle sauvegardé: {path}")
        
    def load_model(self, path: str) -> None:
        """
        Charge un modèle avec toutes ses informations.
        
        Args:
            path: Chemin du modèle
        """
        import joblib
        
        try:
            model_data = joblib.load(path)
            
            # Vérifier si c'est un objet MLModel complet
            if hasattr(model_data, 'model') and hasattr(model_data, 'model_type'):
                # C'est un objet MLModel complet
                self.model = model_data.model
                self.calibrator = getattr(model_data, 'calibrator', None)
                self.feature_importance = getattr(model_data, 'feature_importance', None)
                self.model_type = model_data.model_type
                self.config = getattr(model_data, 'config', {})
                self.columns_ = getattr(model_data, 'columns_', None)
                self.random_state = getattr(model_data, 'random_state', 42)
                self.last_training_metrics = getattr(model_data, 'last_training_metrics', None)
                self.label_encoder = getattr(model_data, 'label_encoder', None)
                logger.info(f"Modèle MLModel complet chargé: {path}")
                
            elif isinstance(model_data, dict):
                # C'est un dictionnaire avec les données du modèle
                self.model = model_data['model']
                self.calibrator = model_data.get('calibrator')
                self.feature_importance = model_data['feature_importance']
                self.model_type = model_data['model_type']
                self.config = model_data['config']
                self.columns_ = model_data.get('columns_')
                self.random_state = model_data.get('random_state', 42)
                self.last_training_metrics = model_data.get('last_training_metrics')
                self.label_encoder = model_data.get('label_encoder')
                logger.info(f"Modèle dictionnaire chargé: {path}")
                
            else:
                # Format inconnu
                raise ValueError(f"Format de modèle inconnu: {type(model_data)}")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise

    def save_state(self, path: str) -> None:
        """
        Sauvegarde l'état du modèle.
        
        Args:
            path: Chemin de sauvegarde
        """
        import joblib
        
        state = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'last_training_metrics': self.last_training_metrics,
            'feature_monitor': self.feature_monitor,
            'model_monitor': self.model_monitor
        }
        
        joblib.dump(state, path)
        logger.info(f"État du modèle sauvegardé: {path}")
        
    def load_state(self, path: str) -> None:
        """
        Charge l'état du modèle.
        
        Args:
            path: Chemin de chargement
        """
        import joblib
        
        try:
            state = joblib.load(path)
            self.model = state['model']
            self.feature_importance = state['feature_importance']
            self.last_training_metrics = state['last_training_metrics']
            self.feature_monitor = state['feature_monitor']
            self.model_monitor = state['model_monitor']
            logger.info(f"État du modèle chargé: {path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {e}")
            raise