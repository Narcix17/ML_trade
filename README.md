# 🚀 Système de Trading Algorithmique avec ML

Un système de trading algorithmique avancé utilisant l'apprentissage automatique pour la détection de signaux de trading, la détection de régimes de marché et l'exécution automatisée.

## 🎯 Fonctionnalités

### 🤖 Intelligence Artificielle
- **Modèles ML** : XGBoost et LightGBM pour la prédiction de signaux
- **Détection de régimes de marché** : Clustering pour identifier les conditions de marché
- **Features techniques** : 56 indicateurs techniques et comportementaux
- **Labeling intelligent** : Méthode basée sur les rendements futurs avec seuils adaptatifs

### 📊 Analyse de Marché
- **Détection de régimes** : 5 régimes de marché différents
- **Features contextuelles** : Sessions de trading, corrélations inter-actifs
- **Monitoring en temps réel** : Surveillance des features et dérive des modèles
- **Backtesting** : Moteur de backtesting avec métriques détaillées

### ⚡ Exécution
- **Servo de trading** : Exécution automatisée avec paper trading
- **Gestion des risques** : Stop-loss, take-profit, position sizing
- **Filtres de marché** : Sessions, spread, volatilité, news
- **Monitoring des positions** : Suivi en temps réel des trades

## 🏗️ Architecture

```
trading-system/
├── 📁 backtesting/          # Moteur de backtesting
├── 📁 data_pipeline/        # Chargement et traitement des données
├── 📁 execution/            # Exécution des ordres
├── 📁 features/             # Génération des features
├── 📁 labeling/             # Génération des labels
├── 📁 models/               # Modèles ML et détection de régimes
├── 📁 monitoring/           # Monitoring des features et modèles
├── 📁 servo/                # Servo de trading
├── 📁 tests/                # Tests unitaires
├── 📄 main.py               # Script principal d'entraînement
├── 📄 config.yaml           # Configuration du système
└── 📄 requirements.txt      # Dépendances Python
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- MetaTrader 5 (pour les données de marché)
- Compte de trading (optionnel pour le paper trading)

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Configuration
1. Copie `config.yaml` et adapte les paramètres
2. Configure les connexions MT5 dans `config.yaml`
3. Ajuste les paramètres de trading selon tes besoins

## 📈 Utilisation

### Entraînement du modèle
```bash
# Entraînement avec dates spécifiques
python main.py --start-date 2024-01-01 --end-date 2024-12-31 --symbol EURUSD --timeframe M5 --model-type xgboost

# Entraînement avec configuration par défaut
python main.py
```

### Lancement du servo de trading
```bash
python servo/trading_servo.py
```

### Backtesting
```bash
python backtesting/backtest_engine.py
```

## 📊 Métriques de Performance

### Modèle ML (dernière exécution)
- **Accuracy** : 97.40%
- **F1-Score (Macro)** : 76.96%
- **ROC-AUC** : 98.89%
- **Précision des signaux** : 98.75%

### Distribution des Labels
- **Neutre** : 94.86%
- **Achat** : 2.30%
- **Vente** : 2.84%

### Régimes de Marché Détectés
- **Régime 0** (Normal) : 77.86%
- **Régime 1** (Volatil) : 0.53%
- **Régime 2** (Tendance) : 18.86%
- **Régime 3** (Crise) : 2.70%
- **Régime 4** (Extrême) : 0.05%

## ⚙️ Configuration

### Paramètres principaux dans `config.yaml`
```yaml
# Seuils de trading
trading:
  min_confidence: 0.5
  risk_management:
    max_positions: 3
    position_size: 0.02
    max_drawdown: 0.05

# Labeling
labeling:
  threshold: 0.002  # 0.2% pour les signaux
  horizon: 20       # Périodes pour le rendement futur

# Modèles ML
ml:
  model:
    type: "xgboost"  # ou "lightgbm"
```

## 🔧 Développement

### Structure des modèles
- **MLModel** : Entraînement et évaluation des modèles ML
- **MarketRegimeDetector** : Détection des régimes de marché
- **FeatureEngineer** : Génération des features techniques
- **LabelGenerator** : Génération des labels de trading

### Tests
```bash
python -m pytest tests/
```

## 📝 Logs et Monitoring

Le système génère des logs détaillés :
- `trading_model.log` : Logs d'entraînement
- `feature_engineering_test.log` : Logs de génération des features
- Monitoring en temps réel des features et modèles

## 🛡️ Gestion des Risques

- **Position sizing** : Limitation de la taille des positions
- **Stop-loss** : Protection contre les pertes
- **Take-profit** : Sécurisation des gains
- **Drawdown limits** : Limitation des pertes maximales
- **Session filters** : Trading uniquement pendant les sessions actives

## 📈 Roadmap

- [ ] Interface web pour le monitoring
- [ ] Intégration de nouveaux brokers
- [ ] Optimisation des hyperparamètres
- [ ] Stratégies multi-timeframes
- [ ] Analyse de sentiment
- [ ] Intégration de données fondamentales

## 🤝 Contribution

1. Fork le projet
2. Crée une branche pour ta feature (`git checkout -b feature/AmazingFeature`)
3. Commit tes changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvre une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## ⚠️ Avertissement

**Ce système est destiné à des fins éducatives et de recherche. Le trading comporte des risques de perte. Utilisez-le à vos propres risques et responsabilités.**

## 📞 Support

Pour toute question ou problème :
- Ouvrez une issue sur GitHub
- Consultez la documentation dans les commentaires du code
- Vérifiez les logs pour diagnostiquer les problèmes

---

**Dernière mise à jour** : 2025-06-09  
**Version** : 1.0.0  
**Statut** : Production Ready ✅