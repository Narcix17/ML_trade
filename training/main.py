#!/usr/bin/env python3
"""
Script principal pour l'entraînement du système de trading algorithmique.

Ce script orchestre:
1. Chargement des données depuis MetaTrader 5
2. Génération des features
3. Détection des régimes de marché
4. Préparation des labels
5. Entraînement du modèle ML
6. Évaluation et sauvegarde
"""

import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import joblib
import os
import yaml
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import warnings
warnings.filterwarnings('ignore')

from utils.data_loading import get_mt5_connector, load_mt5_data
from utils.config import load_config, get_model_path, get_feature_engineer_path
from utils.logging import setup_logger, log_data_loading, log_feature_generation, log_model_training, log_model_evaluation

from features.feature_engineering import FeatureEngineer
from models.market_regime import MarketRegimeDetector
from models.ml_model import MLModel
from monitoring.feature_monitor import FeatureMonitor
from labeling.label_generator import LabelGenerator

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Système de trading algorithmique avec ML')
    
    parser.add_argument('--config', default='config.yaml', help='Fichier de configuration')
    parser.add_argument('--start-date', help='Date de début (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Date de fin (YYYY-MM-DD)')
    parser.add_argument('--symbol', default='EURUSD', help='Symbole à trader')
    parser.add_argument('--timeframe', default='M5', help='Timeframe')
    parser.add_argument('--model-type', default='xgboost', choices=['xgboost', 'lightgbm'], help='Type de modèle ML')
    parser.add_argument('--list-models', action='store_true', help='Lister les modèles disponibles')
    
    return parser.parse_args()

def load_mt5_data(symbol: str, start_date: str, end_date: str, timeframe: str = 'M5') -> pd.DataFrame:
    """
    Charge les données depuis MetaTrader 5.
    
    Args:
        symbol: Symbole de trading
        start_date: Date de début (YYYY-MM-DD)
        end_date: Date de fin (YYYY-MM-DD)
        timeframe: Timeframe (M5, H1, etc.)
        
    Returns:
        DataFrame avec les données OHLCV
    """
    try:
        import MetaTrader5 as mt5
        
        # Initialisation de MT5
        if not mt5.initialize():
            logger.error("Échec de l'initialisation MT5")
            return pd.DataFrame()
            
        logger.info(f"MT5 initialized. Chargement des données pour {symbol} de {start_date} à {end_date} (en morceaux de 1000 bars, timeframe: {timeframe})...")
        
        # Configuration du timeframe
        tf_map = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
        
        # Conversion des dates
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        # Chargement par morceaux pour éviter les limites
        all_data = []
        chunk_size = 1000
        
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=7), end_dt)
            
            rates = mt5.copy_rates_range(
                symbol,
                mt5_timeframe,
                current_start,
                current_end
            )
            
            if rates is not None and len(rates) > 0:
                all_data.append(pd.DataFrame(rates))
                
            current_start = current_end
            
        mt5.shutdown()
        
        if not all_data:
            logger.warning("Aucune donnée trouvée")
            return pd.DataFrame()
            
        # Concaténation des données
        df = pd.concat(all_data, ignore_index=True)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Renommage des colonnes
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        }, inplace=True)
        
        logger.info(f"Données MT5 chargées (concaténées) : {len(df)} lignes.")
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données MT5: {e}")
        return pd.DataFrame()

def prepare_labels(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Prépare les labels pour l'entraînement.
    
    Args:
        df: DataFrame avec les données OHLCV
        config: Configuration du système
        
    Returns:
        Série des labels
    """
    # Calcul des rendements
    df['returns'] = df['close'].pct_change()
    
    # Paramètres de labeling
    threshold = config.get('labeling', {}).get('threshold', 0.002)  # Augmenté à 0.2%
    horizon = config.get('labeling', {}).get('horizon', 20)
    
    # Calcul des rendements futurs
    future_returns = df['close'].pct_change(horizon).shift(-horizon)
    
    # Génération des labels basée sur les rendements futurs
    labels = pd.Series(index=df.index, dtype=float)
    
    # Labels basés sur les seuils de rendement
    labels[future_returns > threshold] = 1      # Achat
    labels[future_returns < -threshold] = 2     # Vente  
    labels[(future_returns >= -threshold) & (future_returns <= threshold)] = 0  # Neutre
    
    # Filtrer les NaN avant conversion en int
    labels = labels.dropna()
    labels_numeric = labels.astype(int)
    
    # Analyse détaillée des labels
    logger.info("=" * 60)
    logger.info("🔍 ANALYSE DÉTAILLÉE DES SIGNALS")
    logger.info("=" * 60)
    
    # Statistiques des rendements futurs
    logger.info(f"📊 STATISTIQUES DES RENDEMENTS FUTURS:")
    logger.info(f"   Seuil utilisé: ±{threshold:.4f} ({threshold*100:.2f}%)")
    logger.info(f"   Rendement moyen: {future_returns.mean():.6f}")
    logger.info(f"   Rendement médian: {future_returns.median():.6f}")
    logger.info(f"   Écart-type: {future_returns.std():.6f}")
    logger.info(f"   Min: {future_returns.min():.6f}")
    logger.info(f"   Max: {future_returns.max():.6f}")
    
    # Distribution des rendements
    buy_signals = (future_returns > threshold).sum()
    sell_signals = (future_returns < -threshold).sum()
    neutral_signals = len(df) - buy_signals - sell_signals
    
    logger.info(f"\n📈 DISTRIBUTION DES RENDEMENTS:")
    logger.info(f"   Rendements > {threshold:.4f} (achat): {buy_signals} ({buy_signals/len(df)*100:.2f}%)")
    logger.info(f"   Rendements < -{threshold:.4f} (vente): {sell_signals} ({sell_signals/len(df)*100:.2f}%)")
    logger.info(f"   Rendements entre ±{threshold:.4f} (neutre): {neutral_signals} ({neutral_signals/len(df)*100:.2f}%)")
    
    # Distribution des labels
    label_counts = labels_numeric.value_counts().sort_index()
    logger.info(f"\n🎯 DISTRIBUTION DES LABELS:")
    for label, count in label_counts.items():
        label_name = {0: "neutre", 1: "achat", 2: "vente"}[label]
        logger.info(f"   Classe {label} ({label_name}): {count} échantillons ({count/len(df)*100:.2f}%)")
    
    # Analyse temporelle
    logger.info(f"\n⏰ ANALYSE TEMPORELLE:")
    logger.info(f"   Période: {df.index[0]} à {df.index[-1]}")
    logger.info(f"   Durée totale: {df.index[-1] - df.index[0]}")
    logger.info(f"   Nombre total d'échantillons: {len(df)}")
    
    # Exemples de signals
    buy_indices = labels_numeric[labels_numeric == 1].index
    sell_indices = labels_numeric[labels_numeric == 2].index
    
    if len(buy_indices) > 0:
        first_buy = buy_indices[0]
        last_buy = buy_indices[-1]
        first_buy_return = future_returns.loc[first_buy]
        logger.info(f"\n🔍 EXEMPLES DE SIGNALS:")
        logger.info(f"   Premier signal d'achat: {first_buy}")
        logger.info(f"   Dernier signal d'achat: {last_buy}")
        logger.info(f"   Rendement au premier achat: {first_buy_return:.6f}")
    
    if len(sell_indices) > 0:
        first_sell = sell_indices[0]
        last_sell = sell_indices[-1]
        first_sell_return = future_returns.loc[first_sell]
        logger.info(f"   Premier signal de vente: {first_sell}")
        logger.info(f"   Dernier signal de vente: {last_sell}")
        logger.info(f"   Rendement au premier vente: {first_sell_return:.6f}")
    
    logger.info("=" * 60)
    
    logger.info(f"Labels générés: {label_counts.to_dict()}")
    return labels_numeric

def load_models(model_type: str, symbol: str, timeframe: str):
    """
    Charge les modèles sauvegardés avec la structure type/name/timeframe.
    
    Args:
        model_type: Type de modèle (xgboost, lightgbm)
        symbol: Symbole (EURUSD, etc.)
        timeframe: Timeframe (M5, H1, etc.)
        
    Returns:
        Tuple (ml_model, regime_detector, feature_monitor, feature_engineer, metadata)
    """
    import joblib
    
    model_dir = f"models/saved/{model_type}/{symbol}/{timeframe}"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Répertoire de modèles non trouvé: {model_dir}")
    
    # Chargement des modèles
    ml_model = joblib.load(f"{model_dir}/ml_model.joblib")
    regime_detector = joblib.load(f"{model_dir}/regime_detector.joblib")
    feature_monitor = joblib.load(f"{model_dir}/feature_monitor.joblib")
    feature_engineer = joblib.load(f"{model_dir}/feature_engineer.joblib")
    
    # Chargement des métadonnées
    with open(f"{model_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Modèles chargés depuis: {model_dir}")
    logger.info(f"  - Type: {metadata['model_type']}")
    logger.info(f"  - Symbole: {metadata['symbol']}")
    logger.info(f"  - Timeframe: {metadata['timeframe']}")
    logger.info(f"  - Date d'entraînement: {metadata['training_date']}")
    
    return ml_model, regime_detector, feature_monitor, feature_engineer, metadata

def list_available_models():
    """
    Liste tous les modèles disponibles avec la structure type/name/timeframe.
    
    Returns:
        Liste des modèles disponibles
    """
    models_dir = "models/saved"
    available_models = []
    
    if not os.path.exists(models_dir):
        return available_models
    
    for model_type in os.listdir(models_dir):
        type_dir = os.path.join(models_dir, model_type)
        if os.path.isdir(type_dir):
            for symbol in os.listdir(type_dir):
                symbol_dir = os.path.join(type_dir, symbol)
                if os.path.isdir(symbol_dir):
                    for timeframe in os.listdir(symbol_dir):
                        timeframe_dir = os.path.join(symbol_dir, timeframe)
                        if os.path.isdir(timeframe_dir):
                            metadata_file = os.path.join(timeframe_dir, "metadata.json")
                            if os.path.exists(metadata_file):
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                available_models.append({
                                    'type': model_type,
                                    'symbol': symbol,
                                    'timeframe': timeframe,
                                    'training_date': metadata.get('training_date', 'Unknown'),
                                    'accuracy': metadata.get('model_metrics', {}).get('accuracy', 0)
                                })
    
    return available_models

def main():
    """Fonction principale."""
    # Parse des arguments
    args = parse_arguments()
    
    # Si l'option --list-models est utilisée
    if args.list_models:
        logger.info("📋 Modèles disponibles :")
        available_models = list_available_models()
        
        if not available_models:
            logger.info("   Aucun modèle trouvé.")
            return
        
        logger.info("   Type | Symbole | Timeframe | Date d'entraînement | Accuracy")
        logger.info("   " + "-" * 70)
        for model in available_models:
            training_date = model['training_date'][:10] if model['training_date'] != 'Unknown' else 'Unknown'
            accuracy = f"{model['accuracy']:.4f}" if model['accuracy'] > 0 else 'N/A'
            logger.info(f"   {model['type']:6} | {model['symbol']:7} | {model['timeframe']:9} | {training_date:19} | {accuracy}")
        return
    
    # Chargement de la configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Mise à jour de la configuration avec les arguments
    config['broker']['symbols'][0]['name'] = args.symbol
    config['broker']['timeframes'][0]['name'] = args.timeframe
    config['ml']['model']['type'] = args.model_type
    
    logger.info("🚀 Démarrage de l'entraînement du système de trading")
    logger.info(f"📅 Période: {args.start_date} à {args.end_date}")
    logger.info(f"💱 Symbole: {args.symbol}")
    logger.info(f"⏰ Timeframe: {args.timeframe}")
    logger.info(f"🤖 Modèle: {args.model_type}")
    
    # Chargement des données
    df = load_mt5_data(args.symbol, args.start_date, args.end_date, args.timeframe)
    
    if df.empty:
        logger.error("Impossible de charger les données")
        return
    
    # Préparation des labels
    labels = prepare_labels(df, config)
    
    # Génération des features
    logger.info("Génération des features...")
    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.generate_features(df)
    
    # Détection des régimes de marché
    logger.info("Détection des régimes de marché...")
    regime_detector = MarketRegimeDetector(config)
    regime_labels, regime_metrics = regime_detector.detect_regimes(features)
    regime_distribution = regime_labels.value_counts().sort_index()
    logger.info(f"Caractéristiques des régimes: {regime_distribution}")
    
    # Alignement des features et labels
    common_index = features.index.intersection(labels.index)
    features = features.loc[common_index]
    labels = labels.loc[common_index]
    
    # Nettoyage minimal : suppression des NaN et conversion en int
    valid_mask = ~labels.isna()
    features = features.loc[valid_mask]
    labels = labels.loc[valid_mask]
    labels = labels.astype(int)
    
    logger.info(f"Features et labels alignés (shapes: features: {features.shape}, labels: {labels.shape})")
    logger.info(f"Distribution finale des labels: {labels.value_counts().to_dict()}")
    
    # Entraînement du modèle ML
    logger.info("Entraînement du modèle ML...")
    ml_model = MLModel(config)
    ml_model.train(features, labels)
    
    # Évaluation du modèle
    logger.info("Évaluation du modèle...")
    metrics = ml_model.evaluate(features, labels)
    logger.info(f"Métriques du modèle: {metrics}")
    
    # Initialisation du moniteur de features
    logger.info("Initialisation du moniteur de features...")
    feature_monitor = FeatureMonitor(config['monitoring'])
    feature_monitor.compute_reference_stats(features)
    
    # Sauvegarde des modèles et des états
    logger.info("Sauvegarde des modèles et des états...")
    
    # Création du répertoire de sauvegarde avec structure type/name/timeframe
    model_type = config['ml']['model']['type']  # xgboost ou lightgbm
    symbol = config['broker']['symbols'][0]['name']  # EURUSD
    timeframe = config['broker']['timeframes'][0]['name']  # M5
    
    # Création de la structure de dossiers
    model_dir = f"models/saved/{model_type}/{symbol}/{timeframe}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Noms de fichiers sans timestamp
    ml_model_filename = f"{model_dir}/ml_model.joblib"
    regime_detector_filename = f"{model_dir}/regime_detector.joblib"
    feature_monitor_filename = f"{model_dir}/feature_monitor.joblib"
    feature_engineer_filename = f"{model_dir}/feature_engineer.joblib"
    metadata_filename = f"{model_dir}/metadata.json"
    
    # Sauvegarde des modèles
    joblib.dump(ml_model, ml_model_filename)
    joblib.dump(regime_detector, regime_detector_filename)
    joblib.dump(feature_monitor, feature_monitor_filename)
    joblib.dump(feature_engineer, feature_engineer_filename)
    
    # Sauvegarde des métadonnées
    metadata = {
        'model_type': model_type,
        'symbol': symbol,
        'timeframe': timeframe,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'training_date': datetime.now().isoformat(),
        'features_shape': features.shape,
        'labels_shape': labels.shape,
        'regime_distribution': regime_labels.value_counts().to_dict(),
        'model_metrics': metrics,
        'config': config
    }
    
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Modèles sauvegardés dans: {model_dir}")
    logger.info(f"  - ML Model: {ml_model_filename}")
    logger.info(f"  - Régime Detector: {regime_detector_filename}")
    logger.info(f"  - Feature Monitor: {feature_monitor_filename}")
    logger.info(f"  - Feature Engineer: {feature_engineer_filename}")
    logger.info(f"  - Métadonnées: {metadata_filename}")
    
    # Debug final détaillé
    logger.info("=" * 60)
    logger.info("🎯 RÉSULTATS FINAUX DU PIPELINE")
    logger.info("=" * 60)
    
    logger.info(f"\n📊 DONNÉES FINALES:")
    logger.info(f"   Features shape: {features.shape}")
    logger.info(f"   Labels shape: {labels.shape}")
    logger.info(f"   Régimes shape: {regime_labels.shape}")
    
    # Distribution finale des labels
    final_label_counts = labels.value_counts().sort_index()
    logger.info(f"\n🎯 DISTRIBUTION FINALE DES LABELS:")
    for label, count in final_label_counts.items():
        label_name = {0: "Neutre", 1: "Achat", 2: "Vente"}[label]
        logger.info(f"   {label_name}: {count} échantillons ({count/len(labels)*100:.2f}%)")
    
    # Régimes détectés
    logger.info(f"\n📈 RÉGIMES DE MARCHÉ DÉTECTÉS:")
    for regime, count in regime_distribution.items():
        logger.info(f"   Régime {regime}: {count} échantillons ({count/len(regime_labels)*100:.2f}%)")
    
    # Analyse croisée labels/régimes
    # S'assurer que les index sont alignés et uniques
    common_index = labels.index.intersection(regime_labels.index)
    labels_aligned = labels.loc[common_index]
    regime_labels_aligned = regime_labels.loc[common_index]
    
    # Vérifier qu'il n'y a pas de doublons
    if labels_aligned.index.duplicated().any():
        logger.warning("Index dupliqués détectés dans labels, suppression...")
        labels_aligned = labels_aligned[~labels_aligned.index.duplicated()]
        regime_labels_aligned = regime_labels_aligned.loc[labels_aligned.index]
    
    if regime_labels_aligned.index.duplicated().any():
        logger.warning("Index dupliqués détectés dans regime_labels, suppression...")
        regime_labels_aligned = regime_labels_aligned[~regime_labels_aligned.index.duplicated()]
        labels_aligned = labels_aligned.loc[regime_labels_aligned.index]
    
    cross_tab = pd.crosstab(labels_aligned, regime_labels_aligned)
    logger.info(f"\n🔍 ANALYSE CROISÉE LABELS/RÉGIMES:")
    logger.info(f"{cross_tab}")
    
    # Métriques du modèle
    logger.info(f"\n📋 MÉTRIQUES DU MODÈLE:")
    logger.info(f"\n🎯 MÉTRIQUES GLOBALES:")
    logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"   F1-Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"   F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"   Precision (Macro): {metrics['precision_macro']:.4f}")
    logger.info(f"   Precision (Weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"   Recall (Macro): {metrics['recall_macro']:.4f}")
    logger.info(f"   Recall (Weighted): {metrics['recall_weighted']:.4f}")
    
    logger.info(f"\n📊 ROC-AUC:")
    logger.info(f"   ROC-AUC (One-vs-Rest): {metrics['roc_auc_ovr']:.4f}")
    logger.info(f"   ROC-AUC (One-vs-One): {metrics['roc_auc_ovo']:.4f}")
    
    logger.info(f"\n🎯 MÉTRIQUES PAR CLASSE:")
    for i in range(3):
        logger.info(f"   Classe {i} ({['Neutre', 'Achat', 'Vente'][i]}):")
        logger.info(f"     Precision: {metrics[f'precision_class_{i}']:.4f}")
        logger.info(f"     Recall: {metrics[f'recall_class_{i}']:.4f}")
        logger.info(f"     F1-Score: {metrics[f'f1_class_{i}']:.4f}")
        logger.info(f"     Support: {metrics[f'support_class_{i}']:.1f}")
        logger.info(f"     ROC-AUC: {metrics[f'roc_auc_class_{i}']:.4f}")
    
    logger.info(f"\n📈 MÉTRIQUES SPÉCIFIQUES AU TRADING:")
    logger.info(f"   Précision des signaux de trading: {metrics['trading_signals_accuracy']:.4f}")
    logger.info(f"   Ratio de signaux prédits: {metrics['signal_prediction_ratio']:.4f}")
    logger.info(f"   Ratio de signaux réels: {metrics['actual_signal_ratio']:.4f}")
    
    # Distribution des prédictions
    pred_dist = metrics['predicted_class_distribution']
    logger.info(f"\n📊 DISTRIBUTION DES PRÉDICTIONS:")
    for class_name, count in pred_dist.items():
        class_num = int(class_name.split('_')[1])
        class_label = ['Neutre', 'Achat', 'Vente'][class_num]
        logger.info(f"   {class_label}: {count} prédictions ({count/len(labels)*100:.2f}%)")
    
    # Matrice de confusion
    cm = metrics['confusion_matrix']
    logger.info(f"\n🔍 MATRICE DE CONFUSION:")
    logger.info(f"   Prédit →")
    logger.info(f"   Réel ↓")
    logger.info(f"           Neutre  Achat  Vente")
    for i, row in enumerate(cm):
        label = ['Neutre', 'Achat', 'Vente'][i]
        logger.info(f"   {label:8} {row[0]:6d} {row[1]:6d} {row[2]:6d}")
    
    logger.info(f"\n💾 MODÈLES SAUVEGARDÉS:")
    logger.info(f"   - ML Model: {ml_model_filename}")
    logger.info(f"   - Régime Detector: {regime_detector_filename}")
    logger.info(f"   - Feature Monitor: {feature_monitor_filename}")
    logger.info(f"   - Feature Engineer: {feature_engineer_filename}")
    logger.info("=" * 60)
    
    logger.info("Entraînement terminé avec succès!")

if __name__ == "__main__":
    main() 