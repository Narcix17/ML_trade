"""
Module de génération des labels.

Implémente la méthode Triple Barrier pour la génération des labels
de trading, avec support pour le meta-labeling.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats

class LabelGenerator:
    """Générateur de labels pour le trading algorithmique."""
    
    def __init__(self, config: dict):
        """
        Initialise le générateur de labels.
        
        Args:
            config: Configuration du labeling
        """
        self.config = config
        self.labels: Dict[str, pd.Series] = {}
        
    def generate_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        take_profit: float,
        stop_loss: float,
        t1: int,
        volatility_window: int = 20,
        min_return: float = 0.0,
        side: Optional[str] = None
    ) -> pd.Series:
        """
        Génère les labels avec la méthode Triple Barrier.
        
        Args:
            df: DataFrame OHLCV
            take_profit: Niveau de take profit (en %)
            stop_loss: Niveau de stop loss (en %)
            t1: Horizon de prédiction (en barres)
            volatility_window: Fenêtre pour le calcul de la volatilité
            min_return: Rendement minimum pour considérer un trade
            side: Direction forcée ('long', 'short' ou None)
            
        Returns:
            Série des labels (-1, 0, 1)
        """
        df = df.copy()
        # Initialiser à 0 (neutre) par défaut
        labels = pd.Series(0, index=df.index, dtype=float)
        
        # Calcul de la volatilité pour ajustement dynamique
        volatility = df['returns'].rolling(window=volatility_window).std()
        
        # Log de debug sur la volatilité
        n_nan_vol = volatility.isna().sum()
        pct_nan_vol = n_nan_vol / len(volatility) * 100
        logger.debug(f"Volatilité: {n_nan_vol} NaN sur {len(volatility)} ({pct_nan_vol:.2f}%)")
        
        for i in range(len(df) - t1):
            if i + t1 >= len(df):
                continue
                
            # Prix de référence
            ref_price = df['close'].iloc[i]
            
            # Ajustement dynamique des barrières
            vol_factor = volatility.iloc[i]
            if np.isnan(vol_factor) or vol_factor == 0:
                labels.iloc[i] = 0
                logger.debug(f"[i={i}] vol_factor NaN ou 0, label neutre")
                continue
            tp_level = ref_price * (1 + take_profit * vol_factor)
            sl_level = ref_price * (1 - stop_loss * vol_factor)
            
            # Fenêtre de prix
            price_window = df['close'].iloc[i:i+t1+1]
            high_window = df['high'].iloc[i:i+t1+1]
            low_window = df['low'].iloc[i:i+t1+1]
            
            # Détermination de la direction
            if side == 'long':
                direction = 1
            elif side == 'short':
                direction = -1
            else:
                # Direction basée sur le rendement forward
                forward_return = (price_window.iloc[-1] / ref_price) - 1
                direction = 1 if forward_return > min_return else -1
                
            # Vérification des barrières
            if direction == 1:  # Long
                # Take profit touché
                if (high_window >= tp_level).any():
                    labels.iloc[i] = 1
                # Stop loss touché
                elif (low_window <= sl_level).any():
                    labels.iloc[i] = 2  # Vente (2 pour cohérence avec le mapping)
                # Timeout
                else:
                    labels.iloc[i] = 0
            else:  # Short
                # Take profit touché
                if (low_window <= sl_level).any():
                    labels.iloc[i] = 1
                # Stop loss touché
                elif (high_window >= tp_level).any():
                    labels.iloc[i] = 2
                # Timeout
                else:
                    labels.iloc[i] = 0
                    
        return labels
        
    def generate_meta_labels(
        self,
        df: pd.DataFrame,
        primary_labels: pd.Series,
        model,
        threshold: float = 0.6
    ) -> pd.Series:
        """
        Génère les meta-labels pour validation secondaire.
        
        Args:
            df: DataFrame avec les features
            primary_labels: Labels primaires
            model: Modèle de classification
            threshold: Seuil de confiance
            
        Returns:
            Série des meta-labels
        """
        # Prédictions du modèle
        probas = model.predict_proba(df)
        confidence = np.max(probas, axis=1)
        
        # Meta-labels basés sur la confiance
        meta_labels = pd.Series(index=df.index, dtype=float)
        meta_labels[confidence >= threshold] = primary_labels[confidence >= threshold]
        meta_labels[confidence < threshold] = 0
        
        return meta_labels
        
    def balance_labels(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        method: str = 'smote'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Équilibre les classes de labels.
        
        Args:
            features: DataFrame des features
            labels: Série des labels
            method: Méthode d'équilibrage ('smote', 'undersample', 'oversample')
            
        Returns:
            Tuple (features équilibrées, labels équilibrés)
        """
        if method == 'smote':
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(features, labels)
            
        elif method == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(features, labels)
            
        elif method == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(features, labels)
            
        else:
            raise ValueError(f"Méthode d'équilibrage non supportée: {method}")
            
        return pd.DataFrame(X_resampled, columns=features.columns), pd.Series(y_resampled)
        
    def get_label_statistics(self, labels: pd.Series) -> Dict:
        """
        Calcule les statistiques des labels.
        
        Args:
            labels: Série des labels
            
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            'total_samples': len(labels),
            'positive_samples': (labels == 1).sum(),
            'negative_samples': (labels == -1).sum(),
            'neutral_samples': (labels == 0).sum(),
            'positive_ratio': (labels == 1).mean(),
            'negative_ratio': (labels == -1).mean(),
            'neutral_ratio': (labels == 0).mean()
        }
        
        return stats
        
    def plot_label_distribution(
        self,
        labels: pd.Series,
        title: str = "Distribution des Labels"
    ) -> None:
        """
        Affiche la distribution des labels.
        
        Args:
            labels: Série des labels
            title: Titre du graphique
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x=labels)
        plt.title(title)
        plt.xlabel("Label")
        plt.ylabel("Nombre d'échantillons")
        plt.show()
        
    def save_labels(self, labels: pd.Series, name: str) -> None:
        """
        Sauvegarde les labels.
        
        Args:
            labels: Série des labels
            name: Nom des labels
        """
        self.labels[name] = labels
        labels.to_csv(f"data/labels/{name}.csv")
        logger.info(f"Labels {name} sauvegardés")
        
    def load_labels(self, name: str) -> pd.Series:
        """
        Charge les labels sauvegardés.
        
        Args:
            name: Nom des labels
            
        Returns:
            Série des labels
        """
        if name in self.labels:
            return self.labels[name]
            
        try:
            labels = pd.read_csv(f"data/labels/{name}.csv", index_col=0, parse_dates=True)
            self.labels[name] = labels
            return labels
        except Exception as e:
            logger.error(f"Erreur lors du chargement des labels {name}: {e}")
            raise 