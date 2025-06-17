# 🚀 **GUIDE DE DÉMARRAGE RAPIDE**

Guide rapide pour utiliser le système de trading automatisé ML+RL.

## 📋 **Prérequis**

- Python 3.8+
- MetaTrader 5
- Compte de trading (demo ou réel)

## ⚡ **Installation Express**

### **1. Cloner et installer**
```bash
git clone <repository-url>
cd cursor
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### **2. Configuration MT5**
Éditer `config.yaml` :
```yaml
mt5:
  login: YOUR_LOGIN
  password: YOUR_PASSWORD
  server: YOUR_SERVER
```

## 🎯 **Utilisation Rapide**

### **Point d'entrée principal**
```bash
# Trading live
python run.py live-trading

# Entraînement ML
python run.py training

# Entraînement RL
python run.py rl-training

# Tests système
python run.py tests
```

### **Scripts individuels**
```bash
# Tests rapides
python scripts/run_tests.py --quick

# Comparaison ML vs RL
python scripts/compare_ml_vs_rl.py

# Monitoring
python scripts/monitoring.py
```

## 🔧 **Commandes Principales**

### **Trading Live**
```bash
python run.py live-trading --config config.yaml --log-level INFO
```

### **Entraînement**
```bash
# ML avec SMOTEENN
python run.py training

# RL avec PPO
python run.py rl-training

# Modèle stratégique
python run.py strategic-training
```

### **Tests et Validation**
```bash
# Tests rapides
python run.py tests

# Tests détaillés
python scripts/run_tests.py --save-results

# Comparaison de performance
python run.py comparison
```

### **Monitoring**
```bash
# Monitoring système
python run.py monitoring

# Statut système
python run.py status
```

## 📊 **Structure des Commandes**

```
python run.py <command> [options]

Commands:
  live-trading      - Trading en temps réel
  training          - Entraînement ML
  strategic-training - Entraînement stratégique
  rl-training       - Entraînement RL
  rl-testing        - Tests RL
  comparison        - Comparaison ML vs RL
  tests             - Tests système
  monitoring        - Monitoring
  status            - Statut système
  help              - Aide

Options:
  --config PATH     - Fichier de configuration
  --log-level LEVEL - Niveau de logging
```

## 🧪 **Tests Rapides**

### **1. Test de connectivité MT5**
```bash
python scripts/run_tests.py --quick
```

### **2. Test des modèles**
```bash
python scripts/run_tests.py --save-results
```

### **3. Validation complète**
```bash
python run.py tests
```

## 📈 **Workflow Typique**

### **1. Première utilisation**
```bash
# Tests système
python run.py tests

# Entraînement ML
python run.py training

# Entraînement RL
python run.py rl-training

# Comparaison
python run.py comparison
```

### **2. Trading live**
```bash
# Vérification système
python run.py status

# Démarrage trading
python run.py live-trading
```

### **3. Monitoring**
```bash
# Monitoring en temps réel
python run.py monitoring

# Vérification statut
python run.py status
```

## 🔍 **Dépannage**

### **Erreur de connexion MT5**
```bash
# Vérifier la configuration
python scripts/system_status.py

# Tester la connectivité
python scripts/run_tests.py --quick
```

### **Erreur de modèles**
```bash
# Réentraîner les modèles
python run.py training
python run.py rl-training

# Vérifier les fichiers
ls models/saved/
```

### **Erreur de features**
```bash
# Tester la génération de features
python scripts/run_tests.py --log-level DEBUG
```

## 📁 **Fichiers Importants**

- `config.yaml` - Configuration principale
- `run.py` - Point d'entrée principal
- `scripts/run_tests.py` - Tests unifiés
- `trading/live/live_trading.py` - Trading live
- `training/main.py` - Entraînement ML
- `models/rl/training/train_rl.py` - Entraînement RL

## 🎯 **Prochaines Étapes**

1. **Lire la documentation complète** : `README.md`
2. **Explorer les exemples** : `docs/`
3. **Personnaliser la configuration** : `config.yaml`
4. **Tester en mode demo** avant trading réel
5. **Monitorer les performances** régulièrement

---

**🚀 Prêt à trader ! Utilisez `python run.py help` pour plus d'options.** 