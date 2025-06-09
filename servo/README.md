# 🤖 Trading Servo System

Le **Trading Servo** est le composant d'exécution en temps réel du système de trading algorithmique. Il fait le pont entre les prédictions ML et l'exécution réelle des ordres.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Model      │    │ Market Regime   │    │ Feature         │
│   Predictions   │───▶│   Detection     │───▶│   Engineering   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Trading Servo  │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │Signal Gen.  │ │
                    │ └─────────────┘ │
                    │ ┌─────────────┐ │
                    │ │Risk Manager │ │
                    │ └─────────────┘ │
                    │ ┌─────────────┐ │
                    │ │Execution    │ │
                    │ └─────────────┘ │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Order Manager  │
                    │  (MetaTrader5)  │
                    └─────────────────┘
```

## 📁 Structure des Fichiers

```
servo/
├── trading_servo.py      # Composant principal du servo
├── risk_manager.py       # Gestionnaire de risque avancé
├── servo_controller.py   # Contrôleur principal
├── launch_servo.py       # Script de lancement
├── rl_agent.py          # Agent RL (futur)
└── README.md            # Ce fichier
```

## 🚀 Utilisation

### 1. Prérequis

Avant de lancer le servo, assurez-vous d'avoir :

1. **Modèles entraînés** :
   ```bash
   python main.py  # Entraîne les modèles
   ```

2. **Configuration MT5** :
   - Compte MetaTrader5 configuré
   - Fichier `config.yaml` avec les paramètres broker

3. **Dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

### 2. Lancement

#### Mode Paper Trading (Recommandé pour les tests)
```bash
python servo/launch_servo.py --mode paper --config config.yaml
```

#### Mode Live Trading (ATTENTION!)
```bash
python servo/launch_servo.py --mode live --config config.yaml --log-level DEBUG
```

#### Mode Backtest
```bash
python servo/launch_servo.py --mode backtest --config config.yaml
```

#### Mode Dry-Run (Test sans exécution)
```bash
python servo/launch_servo.py --mode paper --dry-run
```

### 3. Options de Lancement

```bash
python servo/launch_servo.py [OPTIONS]

Options:
  --config PATH     Fichier de configuration (défaut: config.yaml)
  --mode MODE       Mode de trading: paper, live, backtest (défaut: paper)
  --log-level LEVEL Niveau de log: DEBUG, INFO, WARNING, ERROR (défaut: INFO)
  --dry-run         Mode test sans exécution réelle
  -h, --help        Affiche l'aide
```

## 🔧 Composants

### 1. Trading Servo (`trading_servo.py`)

**Fonction** : Composant principal qui orchestre tout le système

**Fonctionnalités** :
- Génération de signaux en temps réel
- Intégration ML + Régimes de marché
- Exécution des ordres
- Monitoring des performances

**Classes principales** :
- `TradingServo` : Composant principal
- `TradingSignal` : Représentation d'un signal
- `Position` : Gestion des positions

### 2. Risk Manager (`risk_manager.py`)

**Fonction** : Gestion avancée des risques

**Fonctionnalités** :
- Position sizing dynamique
- Stop loss et take profit adaptatifs
- Monitoring du drawdown
- Alertes de risque en temps réel

**Classes principales** :
- `RiskManager` : Gestionnaire de risque
- `RiskMetrics` : Métriques de risque
- `PositionRisk` : Risque d'une position

### 3. Servo Controller (`servo_controller.py`)

**Fonction** : Contrôleur principal qui orchestre tous les composants

**Fonctionnalités** :
- Démarrage/arrêt du système
- Monitoring global
- Gestion des erreurs
- Rapports de performance

**Classes principales** :
- `ServoController` : Contrôleur principal

## 📊 Flux de Données

### 1. Génération de Signaux

```
Données MT5 → Feature Engineering → Régime Detection → ML Prediction → Signal
```

### 2. Validation et Exécution

```
Signal → Risk Validation → Position Sizing → Order Execution → Position Management
```

### 3. Monitoring

```
Performance Metrics → Risk Monitoring → Alerts → Logging
```

## ⚙️ Configuration

### Configuration du Servo

```yaml
# Dans config.yaml
execution:
  order:
    type: "market"  # market ou limit
    retry_attempts: 3
    retry_delay: 1
    max_slippage: 0.0002
  position:
    partial_close: true
    min_close_size: 0.1
    trailing_stop: true
    trailing_step: 0.0001
  risk:
    max_position_size: 0.1
    max_daily_trades: 10
    max_daily_loss: 0.02
  monitoring:
    check_interval: 1
    timeout: 30
    max_errors: 5
    error_cooldown: 300
```

### Configuration du Risk Management

```yaml
trading:
  risk_management:
    max_positions: 3
    position_size: 0.02
    max_drawdown: 0.05
    max_daily_loss: 0.02
    max_weekly_loss: 0.05
    max_monthly_loss: 0.10
```

## 📈 Monitoring et Alertes

### Métriques Surveillées

1. **Performance** :
   - P&L total
   - Win rate
   - Sharpe ratio
   - Drawdown

2. **Risque** :
   - Drawdown actuel/maximum
   - Perte quotidienne/hebdomadaire/mensuelle
   - VaR (Value at Risk)
   - Ratio de risque

3. **Système** :
   - Connexion MT5
   - Latence d'exécution
   - Erreurs système

### Alertes Automatiques

- **Drawdown excessif** : Arrêt automatique
- **Perte quotidienne** : Réduction des positions
- **Erreurs système** : Notification immédiate
- **Risque extrême** : Arrêt d'urgence

## 🛡️ Sécurité

### Contrôles de Sécurité

1. **Validation des Signaux** :
   - Confiance minimale requise
   - Filtres de trading
   - Contraintes de risque

2. **Gestion des Risques** :
   - Position sizing dynamique
   - Stops adaptatifs
   - Limites de perte

3. **Monitoring Continu** :
   - Surveillance en temps réel
   - Alertes automatiques
   - Arrêt d'urgence

### Modes de Trading

1. **Paper Trading** :
   - Simulation complète
   - Pas de risque financier
   - Test des stratégies

2. **Live Trading** :
   - Exécution réelle
   - Contrôles stricts
   - Monitoring intensif

3. **Backtest** :
   - Test historique
   - Optimisation
   - Validation

## 🔍 Debugging

### Logs

Les logs sont stockés dans :
- `logs/servo_YYYYMMDD_HHMMSS.log` : Logs détaillés
- `logs/trading.log` : Logs de trading

### Commandes de Debug

```bash
# Logs en temps réel
tail -f logs/servo_*.log

# Statut du système
python -c "from servo.servo_controller import ServoController; c = ServoController(); print(c.get_system_status())"

# Rapport de performance
python -c "from servo.servo_controller import ServoController; c = ServoController(); print(c.get_performance_report())"
```

## 🚨 Arrêt d'Urgence

### Arrêt Manuel
```bash
# Ctrl+C dans le terminal
# Ou
kill -TERM <PID>
```

### Arrêt Automatique
Le système s'arrête automatiquement si :
- Drawdown > seuil configuré
- Perte quotidienne > limite
- Erreurs système répétées
- Risque extrême détecté

## 🔮 Évolutions Futures

### RL Agent (`rl_agent.py`)
- Agent de Reinforcement Learning
- Apprentissage par récompenses
- Adaptation automatique

### Améliorations Planifiées
- Interface web de monitoring
- Alertes Telegram/Email
- Optimisation automatique
- Multi-instruments
- Stratégies avancées

## 📞 Support

Pour toute question ou problème :
1. Vérifiez les logs
2. Consultez la documentation
3. Testez en mode paper trading
4. Contactez l'équipe de développement

---

**⚠️ ATTENTION** : Le trading en direct comporte des risques financiers. Testez toujours en mode paper trading avant de passer en live. 