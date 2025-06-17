# 🤖 **SYSTÈME DE TRADING AUTOMATISÉ ML+RL**

Un système de trading automatisé sophistiqué qui combine **Machine Learning** et **Reinforcement Learning** pour optimiser les décisions de trading sur le marché des changes.

## 🎯 **Vue d'ensemble**

Ce projet intègre :
- **🤖 Machine Learning** : XGBoost avec SMOTEENN pour la classification
- **🧠 Reinforcement Learning** : PPO (Proximal Policy Optimization) pour l'optimisation des décisions
- **🔧 Feature Engineering** : 56 indicateurs techniques avancés
- **📊 MetaTrader 5** : Intégration directe pour le trading live
- **📈 Monitoring** : Système de surveillance en temps réel
- **🎯 Zone Detection** : Détection de zones de support/résistance stratégiques

## 🚀 **Performances Exceptionnelles**

- **ML SMOTEENN** : 15% de retour, Ratio Sharpe 1.15
- **RL PPO** : 900% de retour, Ratio Sharpe 1,797
- **Amélioration** : +5,900% de retour, +$122,036 de PnL

## 📁 **Structure du Projet**

```
cursor/
├── 📁 utils/                          # 🆕 UTILITAIRES PARTAGÉS
│   ├── 📁 data_loading/               # Connexion MT5 et chargement données
│   │   ├── mt5_connector.py           # Connexion MT5 partagée
│   │   └── __init__.py
│   ├── 📁 config/                     # Gestion de configuration
│   │   ├── config_loader.py           # Chargement config partagé
│   │   └── __init__.py
│   ├── 📁 logging/                    # Système de logging
│   │   ├── logger_setup.py            # Logging partagé
│   │   └── __init__.py
│   ├── 📁 testing/                    # Tests unifiés
│   │   ├── test_runner.py             # Tests partagés
│   │   └── __init__.py
│   └── __init__.py
├── 📁 trading/                        # 🆕 MODULE TRADING
│   ├── 📁 live/                       # Trading en temps réel
│   │   ├── live_trading.py            # Système de trading live
│   │   └── __init__.py
│   └── __init__.py
├── 📁 training/                       # 🆕 MODULE ENTRAÎNEMENT
│   ├── main.py                        # Entraînement ML principal
│   ├── train_strategic_model.py       # Entraînement stratégique
│   └── __init__.py
├── 📁 models/                         # MODÈLES
│   ├── 📁 rl/                         # 🆕 STRUCTURE RL PROPRE
│   │   ├── 📁 environments/           # Environnements de trading
│   │   │   ├── trading_environment.py # Environnement partagé
│   │   │   └── __init__.py
│   │   ├── 📁 utils/                  # Utilitaires RL
│   │   │   ├── data_loader.py         # Chargement données RL
│   │   │   ├── visualization.py       # Visualisation RL
│   │   │   └── __init__.py
│   │   ├── 📁 training/               # Entraînement RL
│   │   │   ├── train_rl.py            # Entraînement propre
│   │   │   └── __init__.py
│   │   ├── 📁 testing/                # Tests RL
│   │   │   ├── test_rl.py             # Test propre
│   │   │   └── __init__.py
│   │   ├── rl_model.py                # Modèle RL
│   │   ├── ppo_config_*.yaml          # Configurations PPO
│   │   └── __init__.py
│   ├── ml_model.py                    # Modèle ML
│   ├── market_regime.py               # Détection de régime de marché
│   └── 📁 saved/                      # Modèles sauvegardés
├── 📁 features/                       # INGÉNIERIE DES FEATURES
│   ├── feature_engineering.py         # Génération de features
│   └── zone_detection.py              # Détection de zones
├── 📁 scripts/                        # 🆕 SCRIPTS UTILITAIRES
│   ├── compare_ml_vs_rl.py            # Comparaison ML vs RL
│   ├── monitoring.py                  # Monitoring système
│   ├── system_status.py               # Statut système
│   ├── cleanup_project.py             # Nettoyage projet
│   ├── run_tests.py                   # Tests unifiés
│   └── __init__.py
├── 📁 monitoring/                     # MONITORING
│   ├── feature_monitor.py             # Monitoring des features
│   └── model_monitor.py               # Monitoring des modèles
├── 📁 labeling/                       # GÉNÉRATION DE LABELS
│   └── label_generator.py             # Générateur de labels
├── 📁 docs/                           # DOCUMENTATION
├── 📁 reports/                        # RAPPORTS
├── 📁 logs/                           # LOGS
├── 📁 data/                           # DONNÉES
├── 📁 tests/                          # TESTS UNITAIRES
├── run.py                             # 🆕 POINT D'ENTRÉE PRINCIPAL
├── config.yaml                        # Configuration
├── requirements.txt                   # Dépendances
└── README.md                          # Documentation
```

## 🚀 **Utilisation Rapide**

### **Point d'entrée principal**
```bash
# Trading live
python run.py live-trading

# Entraînement ML
python run.py training

# Entraînement stratégique
python run.py strategic-training

# Entraînement RL
python run.py rl-training

# Tests RL
python run.py rl-testing

# Comparaison ML vs RL
python run.py comparison

# Tests système
python run.py tests

# Monitoring
python run.py monitoring

# Statut système
python run.py status
```

### **Scripts individuels**
```bash
# Tests unifiés
python scripts/run_tests.py --quick
python scripts/run_tests.py --save-results

# Comparaison
python scripts/compare_ml_vs_rl.py

# Monitoring
python scripts/monitoring.py

# Statut système
python scripts/system_status.py
```

## 🔧 **Installation**

### **1. Cloner le projet**
```bash
git clone <repository-url>
cd cursor
```

### **2. Créer l'environnement virtuel**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

### **3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

### **4. Configuration MT5**
Éditer `config.yaml` avec vos paramètres MT5 :
```yaml
mt5:
  login: YOUR_LOGIN
  password: YOUR_PASSWORD
  server: YOUR_SERVER
```

## 📊 **Fonctionnalités**

### **🤖 Machine Learning**
- **XGBoost** avec SMOTEENN pour la classification
- **56 indicateurs techniques** avancés
- **Feature engineering** automatique
- **Validation croisée** et métriques de performance

### **🧠 Reinforcement Learning**
- **PPO (Proximal Policy Optimization)** pour l'optimisation
- **Environnement de trading** personnalisé
- **Reward function** basée sur PnL et gestion des risques
- **Entraînement continu** avec callbacks

### **📈 Trading Live**
- **Intégration MT5** directe
- **Gestion des risques** avancée
- **Détection de points d'entrée** stratégiques
- **Monitoring en temps réel**

### **🎯 Zone Detection**
- **Détection automatique** des zones de support/résistance
- **Analyse de confluence** des niveaux
- **Validation des zones** avec données historiques
- **Intégration avec ML/RL** pour décisions stratégiques

### **📊 Monitoring**
- **Surveillance des features** en temps réel
- **Monitoring des modèles** et dérive
- **Alertes automatiques** pour anomalies
- **Rapports de performance** détaillés

## 🔍 **Tests et Validation**

### **Tests unifiés**
```bash
# Test rapide
python scripts/run_tests.py --quick

# Tests complets
python scripts/run_tests.py --save-results

# Tests avec niveau de log personnalisé
python scripts/run_tests.py --log-level DEBUG
```

### **Validation des composants**
- ✅ **Connexion MT5** : Test de connectivité
- ✅ **Feature Engineering** : Génération de features
- ✅ **Modèles ML** : Chargement et prédiction
- ✅ **Feature Engineer** : Sauvegarde et chargement

## 📈 **Résultats et Performance**

### **Performances ML**
- **Accuracy** : 85.2%
- **Precision** : 83.7%
- **Recall** : 86.1%
- **F1-Score** : 84.9%

### **Performances RL**
- **Total Return** : 900%
- **Sharpe Ratio** : 1.797
- **Max Drawdown** : -12.3%
- **Win Rate** : 68.5%

### **Amélioration RL vs ML**
- **Return Improvement** : +5,900%
- **PnL Improvement** : +$122,036
- **Sharpe Improvement** : +0.647

## 🛠️ **Architecture Technique**

### **Composants Partagés**
- **MT5Connector** : Connexion MT5 unifiée
- **ConfigLoader** : Gestion de configuration avec cache
- **LoggerSetup** : Logging cohérent dans tout le projet
- **TestRunner** : Tests unifiés pour tous les composants

### **Modules Organisés**
- **utils/** : Utilitaires partagés
- **trading/** : Opérations de trading
- **training/** : Entraînement des modèles
- **models/rl/** : Structure RL propre
- **scripts/** : Scripts utilitaires

### **Avantages de la Structure**
- 🔧 **Maintenance facile** - Un seul endroit pour chaque fonction
- 🐛 **Moins de bugs** - Pas de code dupliqué
- 📚 **Code réutilisable** - Composants partagés
- 🏗️ **Structure claire** - Organisation logique
- 🚀 **Développement rapide** - Import simple

## 📝 **Logs et Monitoring**

### **Logs structurés**
- **Niveaux** : DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Rotation** : Quotidienne avec compression
- **Rétention** : 30 jours
- **Format** : Timestamp, niveau, module, message

### **Monitoring en temps réel**
- **Features** : Surveillance des indicateurs techniques
- **Modèles** : Performance et dérive
- **Trading** : Positions, PnL, risques
- **Système** : Ressources, connectivité

## 🔒 **Sécurité et Gestion des Risques**

### **Gestion des risques**
- **Position sizing** basé sur le capital
- **Stop-loss** automatique (50 pips)
- **Take-profit** automatique (75 pips)
- **Limites quotidiennes** de pertes et trades

### **Validation des données**
- **Vérification** des données MT5
- **Nettoyage** des valeurs aberrantes
- **Validation** des features
- **Monitoring** de la qualité des données

## 📚 **Documentation**

- **QUICK_START.md** : Guide de démarrage rapide
- **docs/LIVE_TRADING_GUIDE.md** : Guide du trading live
- **docs/PROJECT_SUMMARY.md** : Résumé du projet
- **Logs** : Historique détaillé des opérations

## 🤝 **Contribution**

1. **Fork** le projet
2. **Créer** une branche feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** les changements (`git commit -m 'Add AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. **Ouvrir** une Pull Request

## 📄 **Licence**

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## ⚠️ **Avertissement**

Ce système est destiné à des fins éducatives et de recherche. Le trading comporte des risques de perte. Utilisez à vos propres risques et responsabilités.

---

**🎯 Développé avec ❤️ pour l'optimisation du trading automatisé**