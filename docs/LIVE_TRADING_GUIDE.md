# Guide du Trading Live avec MT5

## 🚀 Vue d'ensemble

Ce guide explique comment utiliser le système de trading live intégrant les modèles ML+RL avec MetaTrader 5 (MT5).

## 📋 Prérequis

### 1. Installation de MT5
- Télécharger et installer MetaTrader 5 depuis le site officiel
- Créer un compte démo ou réel
- Configurer les paramètres de connexion

### 2. Installation des dépendances Python
```bash
pip install MetaTrader5 pandas numpy loguru joblib stable-baselines3 pyyaml
```

### 3. Configuration du projet
- Vérifier que les modèles ML et RL sont entraînés et sauvegardés
- Configurer le fichier `config.yaml` avec les paramètres de risque

## 🔧 Configuration

### Paramètres de risque dans config.yaml
```yaml
risk_management:
  position_size: 0.02  # 2% du capital par trade
  max_daily_loss: 0.05  # 5% de perte quotidienne max
  max_daily_trades: 10  # Maximum 10 trades par jour
  min_confidence: 0.6   # Confiance minimale pour trader
  trading_hours: [0, 24]  # Heures de trading (0-24h)
```

## 🏦 Informations du compte MT5

### Données récupérées automatiquement
- **Balance**: Capital initial du compte
- **Equity**: Valeur actuelle du compte (balance + P&L)
- **Margin**: Marge utilisée pour les positions ouvertes
- **Free Margin**: Marge disponible pour de nouveaux trades
- **Leverage**: Effet de levier du compte (ex: 1:100)
- **Currency**: Devise du compte

### Calcul de la taille de position
```python
# Formule utilisée
balance = account_info.balance
risk_percent = config['risk_management']['position_size']
max_risk_amount = balance * risk_percent
confidence_multiplier = min(action_confidence * 2, 1.0)
adjusted_risk = max_risk_amount * confidence_multiplier
volume = adjusted_risk / tick_value
```

## ⚠️ Limites et Contraintes

### 1. Limites techniques MT5
- **Volume minimum**: Généralement 0.01 lots
- **Volume maximum**: Dépend du broker (souvent 100-1000 lots)
- **Volume step**: Incrément minimum (souvent 0.01)
- **Spread**: Différence bid/ask (peut être fixe ou variable)
- **Slippage**: Déviation de prix autorisée (20 points par défaut)

### 2. Limites de risque
- **Perte quotidienne**: Maximum 5% du capital
- **Nombre de trades**: Maximum 10 par jour
- **Confiance minimale**: 60% pour exécuter un trade
- **Heures de trading**: Configurables (par défaut 24h/24)

### 3. Limites de marché
- **Heures de trading**: Dépend des paires de devises
- **Spread élevé**: Trading suspendu si spread > seuil
- **Volatilité**: Limitation pendant les news importantes
- **Liquidité**: Vérification de la liquidité avant trading

### 4. Limites de connexion
- **Latence réseau**: Impact sur l'exécution des ordres
- **Déconnexion**: Gestion automatique des reconnexions
- **Erreurs MT5**: Gestion des codes d'erreur de trading

## 🔄 Fonctionnement du système

### 1. Boucle principale
```python
while True:
    # 1. Récupération des données de marché
    df = get_market_data(symbol, timeframe, bars=100)
    
    # 2. Génération des features
    features = generate_features(df)
    
    # 3. Prédiction ML
    ml_result = get_ml_prediction(features)
    
    # 4. Prédiction RL
    rl_result = get_rl_action(features, ml_result['prediction'])
    
    # 5. Vérifications de sécurité
    if check_trading_conditions():
        # 6. Calcul de la taille de position
        volume = calculate_position_size(symbol, confidence)
        
        # 7. Exécution du trade
        if action != 0:  # Pas d'action Hold
            execute_trade(symbol, action, volume)
    
    # 8. Attente (30 secondes)
    time.sleep(30)
```

### 2. Gestion des risques
- **Vérification quotidienne**: Reset des compteurs à minuit
- **Limite de perte**: Arrêt automatique si seuil atteint
- **Limite de trades**: Arrêt après nombre maximum atteint
- **Confiance minimale**: Filtrage des signaux faibles

### 3. Exécution des ordres
```python
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": volume,
    "type": order_type,  # BUY ou SELL
    "price": price,
    "deviation": 20,     # Slippage autorisé
    "magic": 123456,     # Identifiant unique
    "comment": "ML+RL Trade",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
```

## 📊 Monitoring et Logging

### 1. Logs automatiques
- **Fichier de log**: `logs/live_trading_YYYYMMDD.log`
- **Rotation**: Nouveau fichier chaque jour
- **Rétention**: 30 jours de logs conservés
- **Niveau**: INFO pour les détails, ERROR pour les problèmes

### 2. Informations loggées
- Connexion/déconnexion MT5
- Récupération des données de marché
- Prédictions ML et RL
- Exécution des trades
- Erreurs et avertissements
- Statistiques de performance

### 3. Historique des trades
```python
trade_info = {
    'ticket': result.order,
    'symbol': symbol,
    'action': 'BUY' or 'SELL',
    'volume': volume,
    'price': price,
    'time': datetime.now(),
    'retcode': result.retcode
}
```

## 🚨 Codes d'erreur MT5

### Erreurs courantes
- **10004**: Requête traitée
- **10006**: Requête reçue
- **10007**: Requête en cours de traitement
- **10008**: Requête complétée
- **10009**: Requête partiellement complétée
- **10010**: Requête rejetée
- **10011**: Requête annulée par le trader
- **10012**: Requête annulée par le système

### Gestion des erreurs
```python
if result.retcode != mt5.TRADE_RETCODE_DONE:
    logger.error(f"Erreur d'exécution: {result.retcode} - {result.comment}")
    return None
```

## 🔒 Sécurité et bonnes pratiques

### 1. Test en démo d'abord
- **Compte démo**: Tester pendant au moins 1 mois
- **Petits volumes**: Commencer avec 0.01 lots
- **Monitoring**: Surveiller les performances
- **Ajustements**: Optimiser les paramètres

### 2. Gestion des risques
- **Stop Loss**: Toujours définir des stops
- **Take Profit**: Objectifs de profit réalistes
- **Diversification**: Ne pas trader une seule paire
- **Capital**: Ne jamais risquer plus que 2% par trade

### 3. Maintenance
- **Mise à jour**: Maintenir les modèles à jour
- **Backup**: Sauvegarder les configurations
- **Monitoring**: Surveiller les logs quotidiennement
- **Optimisation**: Ajuster les paramètres selon les résultats

## 📈 Performance et optimisation

### 1. Métriques à surveiller
- **Win Rate**: Pourcentage de trades gagnants
- **Profit Factor**: Ratio gains/pertes
- **Drawdown**: Perte maximale depuis le pic
- **Sharpe Ratio**: Rendement ajusté au risque
- **Maximum Consecutive Losses**: Pertes consécutives max

### 2. Optimisation des paramètres
- **Confiance minimale**: Ajuster selon les résultats
- **Taille de position**: Optimiser selon la volatilité
- **Heures de trading**: Éviter les périodes de faible liquidité
- **Fréquence**: Ajuster l'intervalle entre les vérifications

## 🛠️ Dépannage

### Problèmes courants
1. **Connexion MT5 échoue**
   - Vérifier que MT5 est installé et ouvert
   - Vérifier les paramètres de connexion
   - Redémarrer MT5 si nécessaire

2. **Erreurs d'exécution**
   - Vérifier les permissions de trading
   - Vérifier la liquidité du marché
   - Ajuster les paramètres de slippage

3. **Données manquantes**
   - Vérifier la connexion internet
   - Vérifier que le symbole est disponible
   - Attendre la prochaine barre

4. **Performance dégradée**
   - Vérifier les ressources système
   - Optimiser les paramètres
   - Redémarrer le système si nécessaire

## 📞 Support

Pour toute question ou problème :
1. Consulter les logs dans `logs/live_trading_*.log`
2. Vérifier la configuration dans `config.yaml`
3. Tester la connexion avec `test_mt5_connection.py`
4. Consulter la documentation MT5 officielle

---

**⚠️ AVERTISSEMENT**: Le trading automatique comporte des risques. Testez toujours en démo avant de passer en réel. Ne risquez jamais plus que vous ne pouvez vous permettre de perdre. 