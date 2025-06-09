# Système de Trading Algorithmique

Système de trading algorithmique modulaire et robuste, utilisant des techniques de machine learning avancées pour le trading d'indices boursiers.

## 🎯 Objectifs

- Trading automatisé sur indices (DAX, S&P500, NASDAQ)
- Multi-timeframes (5min, 15min, 1h)
- Apprentissage supervisé (XGBoost/LightGBM)
- Clustering de régimes (HDBSCAN)
- Reinforcement Learning (PPO/SAC)
- Exécution via MetaTrader 5
- Monitoring en temps réel des performances
- Gestion robuste des risques

## 📁 Structure

```
.
├── data_pipeline/          # Pipeline de données
│   ├── broker_api.py      # Connexion MT5
│   ├── data_processor.py  # Prétraitement
│   └── timeframe_sync.py  # Synchronisation
├── features/              # Feature engineering
│   └── feature_engineering.py
├── labeling/             # Génération des labels
│   └── label_generator.py
├── models/               # Modèles ML
│   ├── market_regime.py  # Clustering
│   ├── ml_model.py       # XGBoost/LightGBM
│   └── rl_agent.py       # PPO/SAC
├── backtesting/          # Backtesting
│   └── backtest_engine.py
├── execution/            # Exécution
│   └── order_manager.py
├── monitoring/           # Monitoring
│   ├── feature_monitor.py  # Monitoring des features
│   └── model_monitor.py    # Monitoring des modèles
├── config.yaml          # Configuration
├── requirements.txt     # Dépendances
└── README.md           # Documentation
```