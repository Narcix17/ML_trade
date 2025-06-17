"""
Module de monitoring des performances des modèles.

Surveille les métriques de performance, détecte la dégradation
et alerte en cas de problèmes.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from datetime import datetime, timedelta

class ModelMonitor:
    """Moniteur de performance des modèles."""
    
    def __init__(self, config: dict):
        """
        Initialise le moniteur de modèles.
        
        Args:
            config: Configuration du monitoring
        """
        self.config = config['monitoring']['performance']
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.reference_metrics: Dict[str, float] = {}
        self.alerts: List[Dict] = []
        self.last_check = None
        
    def compute_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_proba: Optional[np.ndarray] = None,
        task: str = 'classification'
    ) -> Dict[str, float]:
        """
        Calcule les métriques de performance.
        
        Args:
            y_true: Labels réels
            y_pred: Prédictions
            y_proba: Probabilités prédites (optionnel)
            task: Type de tâche ('classification' ou 'regression')
            
        Returns:
            Dictionnaire des métriques
        """
        metrics = {}
        
        if task == 'classification':
            # Métriques de classification
            metrics.update({
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            })
            
            if y_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                
            # Matrice de confusion
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
        elif task == 'regression':
            # Métriques de régression
            metrics.update({
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            })
            
        return metrics
        
    def update_metrics(
        self,
        metrics: Dict[str, float],
        model_name: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Met à jour l'historique des métriques.
        
        Args:
            metrics: Dictionnaire des métriques
            model_name: Nom du modèle
            timestamp: Horodatage (optionnel)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = []
            
        self.metrics_history[model_name].append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Mise à jour des métriques de référence si nécessaire
        if model_name not in self.reference_metrics:
            self.reference_metrics[model_name] = metrics
            
    def check_performance(
        self,
        metrics: Dict[str, float],
        model_name: str,
        window: Optional[int] = None
    ) -> List[Dict]:
        """
        Vérifie la performance du modèle et génère des alertes.
        
        Args:
            metrics: Dictionnaire des métriques
            model_name: Nom du modèle
            window: Fenêtre temporelle pour l'analyse de tendance
            
        Returns:
            Liste des alertes générées
        """
        alerts = []
        thresholds = self.config.get('metrics', {})
        
        # Handle case where thresholds might be a list
        if isinstance(thresholds, list):
            # Convert list to dictionary with default thresholds
            thresholds = {metric: {'threshold': 0.5} for metric in thresholds}
        
        # Vérification des seuils absolus
        for metric_name, threshold_config in thresholds.items():
            if metric_name in metrics:
                threshold_value = threshold_config.get('threshold', 0.5) if isinstance(threshold_config, dict) else threshold_config
                if metrics[metric_name] < threshold_value:
                    alerts.append({
                        'type': 'absolute_threshold',
                        'model': model_name,
                        'metric': metric_name,
                        'value': metrics[metric_name],
                        'threshold': threshold_value,
                        'timestamp': datetime.now()
                    })
                    
        # Vérification de la dégradation relative
        if model_name in self.reference_metrics:
            ref_metrics = self.reference_metrics[model_name]
            for metric_name, value in metrics.items():
                if metric_name in ref_metrics and not isinstance(value, (list, tuple)):
                    degradation = (ref_metrics[metric_name] - value) / ref_metrics[metric_name]
                    degradation_threshold = 0.1  # Default degradation threshold
                    if isinstance(thresholds.get(metric_name), dict):
                        degradation_threshold = thresholds[metric_name].get('degradation_threshold', 0.1)
                    if degradation > degradation_threshold:
                        alerts.append({
                            'type': 'relative_degradation',
                            'model': model_name,
                            'metric': metric_name,
                            'value': value,
                            'reference': ref_metrics[metric_name],
                            'degradation': degradation,
                            'threshold': degradation_threshold,
                            'timestamp': datetime.now()
                        })
                        
        # Vérification de la tendance
        if window and model_name in self.metrics_history:
            recent_metrics = [
                m for m in self.metrics_history[model_name]
                if m['timestamp'] > datetime.now() - timedelta(days=window)
            ]
            
            if len(recent_metrics) > 1:
                for metric_name in metrics:
                    values = [m['metrics'].get(metric_name) for m in recent_metrics]
                    if all(v is not None for v in values):
                        trend = np.polyfit(range(len(values)), values, 1)[0]
                        if abs(trend) > 0.01:  # Tendance significative
                            alerts.append({
                                'type': 'trend',
                                'model': model_name,
                                'metric': metric_name,
                                'trend': trend,
                                'window': window,
                                'timestamp': datetime.now()
                            })
                            
        self.alerts.extend(alerts)
        return alerts
        
    def plot_performance(
        self,
        model_name: str,
        metrics: Optional[List[str]] = None,
        window: Optional[int] = None
    ) -> go.Figure:
        """
        Génère des graphiques de performance.
        
        Args:
            model_name: Nom du modèle
            metrics: Liste des métriques à afficher
            window: Fenêtre temporelle
            
        Returns:
            Figure Plotly
        """
        if model_name not in self.metrics_history:
            raise ValueError(f"Pas d'historique pour le modèle {model_name}")
            
        history = self.metrics_history[model_name]
        if window:
            history = [
                m for m in history
                if m['timestamp'] > datetime.now() - timedelta(days=window)
            ]
            
        if not history:
            raise ValueError("Pas de données dans la fenêtre spécifiée")
            
        # Sélection des métriques
        if metrics is None:
            metrics = list(history[0]['metrics'].keys())
            
        # Création des subplots
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=metrics
        )
        
        for i, metric in enumerate(metrics, 1):
            values = [m['metrics'].get(metric) for m in history]
            timestamps = [m['timestamp'] for m in history]
            
            # Ligne de performance
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    name=metric,
                    mode='lines+markers'
                ),
                row=i,
                col=1
            )
            
            # Seuil de référence
            if model_name in self.reference_metrics:
                ref_value = self.reference_metrics[model_name].get(metric)
                if ref_value is not None:
                    fig.add_hline(
                        y=ref_value,
                        line_dash="dash",
                        line_color="red",
                        row=i,
                        col=1
                    )
                    
            # Seuil de performance
            if metric in self.config['metrics']:
                threshold = self.config['metrics'][metric]['threshold']
                fig.add_hline(
                    y=threshold,
                    line_dash="dot",
                    line_color="orange",
                    row=i,
                    col=1
                )
                
        fig.update_layout(
            height=300 * len(metrics),
            showlegend=True,
            title_text=f"Performance du modèle {model_name}"
        )
        
        return fig
        
    def log_to_mlflow(
        self,
        metrics: Dict[str, float],
        model_name: str,
        run_name: Optional[str] = None
    ) -> None:
        """
        Log les métriques dans MLflow.
        
        Args:
            metrics: Dictionnaire des métriques
            model_name: Nom du modèle
            run_name: Nom de la run
        """
        with mlflow.start_run(run_name=run_name):
            # Log des métriques
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{model_name}_{metric_name}", value)
                    
            # Log des alertes
            if self.alerts:
                alerts_df = pd.DataFrame(self.alerts)
                mlflow.log_table(alerts_df, f"{model_name}_alerts.json")
                
            # Log des métriques de référence
            if model_name in self.reference_metrics:
                mlflow.log_dict(
                    self.reference_metrics[model_name],
                    f"{model_name}_reference_metrics.json"
                )
                
    def get_alert_summary(
        self,
        model_name: Optional[str] = None,
        window: Optional[timedelta] = None
    ) -> pd.DataFrame:
        """
        Génère un résumé des alertes.
        
        Args:
            model_name: Nom du modèle (optionnel)
            window: Fenêtre temporelle (optionnel)
            
        Returns:
            DataFrame avec le résumé des alertes
        """
        if not self.alerts:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.alerts)
        
        if model_name:
            df = df[df['model'] == model_name]
            
        if window:
            cutoff = datetime.now() - window
            df = df[df['timestamp'] > cutoff]
            
        summary = df.groupby(['type', 'model', 'metric']).agg({
            'value': ['count', 'mean', 'min', 'max'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        summary.columns = [
            'type', 'model', 'metric',
            'alert_count', 'mean_value', 'min_value', 'max_value',
            'first_alert', 'last_alert'
        ]
        
        return summary
        
    def save_state(self, path: str) -> None:
        """
        Sauvegarde l'état du moniteur.
        
        Args:
            path: Chemin de sauvegarde
        """
        import joblib
        
        state = {
            'metrics_history': self.metrics_history,
            'reference_metrics': self.reference_metrics,
            'alerts': self.alerts,
            'last_check': self.last_check
        }
        
        joblib.dump(state, path)
        logger.info(f"État du moniteur sauvegardé: {path}")
        
    def load_state(self, path: str) -> None:
        """
        Charge l'état du moniteur.
        
        Args:
            path: Chemin de chargement
        """
        import joblib
        
        try:
            state = joblib.load(path)
            self.metrics_history = state['metrics_history']
            self.reference_metrics = state['reference_metrics']
            self.alerts = state['alerts']
            self.last_check = state['last_check']
            logger.info(f"État du moniteur chargé: {path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {e}")
            raise 