#!/usr/bin/env python3
"""
Script d'état du système de trading
Affiche un résumé complet de l'état du système
"""

import os
import yaml
import MetaTrader5 as mt5
from datetime import datetime
from loguru import logger


def check_mt5_status():
    """Vérifie l'état de MT5."""
    try:
        if not mt5.initialize():
            return False, f"MT5 non initialisé: {mt5.last_error()}"
        
        account_info = mt5.account_info()
        if account_info is None:
            mt5.shutdown()
            return False, "Impossible de récupérer les informations du compte"
        
        status = {
            'connected': True,
            'login': account_info.login,
            'server': account_info.server,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'leverage': account_info.leverage,
            'currency': account_info.currency,
            'type': 'RÉEL' if account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL else 'DÉMO'
        }
        
        mt5.shutdown()
        return True, status
        
    except Exception as e:
        return False, f"Erreur MT5: {e}"


def check_models():
    """Vérifie l'état des modèles."""
    models_status = {}
    
    # Vérification du modèle ML
    try:
        symbol = "EURUSD"
        timeframe = "M5"
        ml_path = f"models/saved/xgboost/{symbol}/{timeframe}/ml_model.joblib"
        models_status['ml'] = {
            'exists': os.path.exists(ml_path),
            'path': ml_path,
            'size': os.path.getsize(ml_path) if os.path.exists(ml_path) else 0
        }
    except Exception as e:
        models_status['ml'] = {'exists': False, 'error': str(e)}
    
    # Vérification du modèle RL
    try:
        rl_path = "models/ppo_smoteenn/ppo_smoteenn_final.zip"
        models_status['rl'] = {
            'exists': os.path.exists(rl_path),
            'path': rl_path,
            'size': os.path.getsize(rl_path) if os.path.exists(rl_path) else 0
        }
    except Exception as e:
        models_status['rl'] = {'exists': False, 'error': str(e)}
    
    return models_status


def check_config():
    """Vérifie la configuration."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        return {
            'exists': True,
            'symbol': config.get('training', {}).get('symbol', 'N/A'),
            'timeframes': config.get('training', {}).get('timeframes', []),
            'position_size': config.get('risk_management', {}).get('position_size', 0),
            'max_daily_loss': config.get('risk_management', {}).get('max_daily_loss', 0),
            'max_daily_trades': config.get('risk_management', {}).get('max_daily_trades', 0)
        }
    except Exception as e:
        return {'exists': False, 'error': str(e)}


def check_directories():
    """Vérifie les répertoires nécessaires."""
    required_dirs = ['logs', 'data', 'models', 'reports', 'docs']
    dir_status = {}
    
    for directory in required_dirs:
        dir_status[directory] = {
            'exists': os.path.exists(directory),
            'writable': os.access(directory, os.W_OK) if os.path.exists(directory) else False
        }
    
    return dir_status


def display_status():
    """Affiche le statut complet du système."""
    print("🚀 ÉTAT DU SYSTÈME DE TRADING")
    print("=" * 50)
    
    # État MT5
    print("\n🏦 ÉTAT MT5:")
    mt5_ok, mt5_status = check_mt5_status()
    if mt5_ok:
        print(f"   ✅ Connecté - Compte {mt5_status['type']}")
        print(f"   📊 Login: {mt5_status['login']}")
        print(f"   🌐 Serveur: {mt5_status['server']}")
        print(f"   💰 Balance: {mt5_status['balance']:.2f} {mt5_status['currency']}")
        print(f"   📈 Equity: {mt5_status['equity']:.2f} {mt5_status['currency']}")
        print(f"   🔧 Levier: 1:{mt5_status['leverage']}")
    else:
        print(f"   ❌ Erreur: {mt5_status}")
    
    # État des modèles
    print("\n🤖 ÉTAT DES MODÈLES:")
    models = check_models()
    
    if models['ml']['exists']:
        size_mb = models['ml']['size'] / (1024 * 1024)
        print(f"   ✅ ML (XGBoost): {size_mb:.1f} MB")
    else:
        print(f"   ❌ ML (XGBoost): Manquant")
    
    if models['rl']['exists']:
        size_mb = models['rl']['size'] / (1024 * 1024)
        print(f"   ✅ RL (PPO): {size_mb:.1f} MB")
    else:
        print(f"   ❌ RL (PPO): Manquant")
    
    # Configuration
    print("\n⚙️ CONFIGURATION:")
    config = check_config()
    if config['exists']:
        print(f"   ✅ Config chargée")
        print(f"   📊 Symbole: {config['symbol']}")
        print(f"   ⏰ Timeframes: {', '.join(config['timeframes'])}")
        print(f"   📈 Taille position: {config['position_size']*100:.1f}%")
        print(f"   🛡️ Perte max quotidienne: {config['max_daily_loss']*100:.1f}%")
        print(f"   🔢 Trades max quotidiens: {config['max_daily_trades']}")
    else:
        print(f"   ❌ Config: {config.get('error', 'Erreur inconnue')}")
    
    # Répertoires
    print("\n📁 RÉPERTOIRES:")
    dirs = check_directories()
    for dir_name, dir_status in dirs.items():
        if dir_status['exists']:
            if dir_status['writable']:
                print(f"   ✅ {dir_name}/")
            else:
                print(f"   ⚠️ {dir_name}/ (non accessible en écriture)")
        else:
            print(f"   ❌ {dir_name}/ (manquant)")
    
    # Fichiers de trading
    print("\n🚀 FICHIERS DE TRADING:")
    trading_files = [
        'live_trading.py',
        'start_live_trading.py',
        'test_mt5_connection.py'
    ]
    
    for file in trading_files:
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"   ✅ {file} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {file} (manquant)")
    
    # Documentation
    print("\n📚 DOCUMENTATION:")
    doc_files = [
        'README.md',
        'docs/LIVE_TRADING_GUIDE.md',
        'docs/LIVE_TRADING_SUMMARY.md',
        'docs/trading_system_documentation.ipynb'
    ]
    
    for file in doc_files:
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"   ✅ {file} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {file} (manquant)")
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS:")
    
    if not mt5_ok:
        print("   🔧 Installer/configurer MT5")
    
    if not models['ml']['exists'] or not models['rl']['exists']:
        print("   🤖 Entraîner les modèles ML et RL")
    
    if not config['exists']:
        print("   ⚙️ Créer le fichier config.yaml")
    
    # Statut global
    print("\n🎯 STATUT GLOBAL:")
    
    all_ok = (
        mt5_ok and 
        models['ml']['exists'] and 
        models['rl']['exists'] and 
        config['exists']
    )
    
    if all_ok:
        print("   🚀 SYSTÈME PRÊT POUR LE TRADING LIVE")
        print("   📋 Prochaines étapes:")
        print("      1. python test_mt5_connection.py")
        print("      2. python start_live_trading.py")
    else:
        print("   ⚠️ SYSTÈME INCOMPLET")
        print("   📋 Actions nécessaires:")
        if not mt5_ok:
            print("      - Configurer MT5")
        if not models['ml']['exists']:
            print("      - python main.py (entraînement ML)")
        if not models['rl']['exists']:
            print("      - python train_rl_with_smoteenn.py (entraînement RL)")
        if not config['exists']:
            print("      - Créer config.yaml")
    
    print("\n" + "=" * 50)


def main():
    """Fonction principale."""
    display_status()


if __name__ == "__main__":
    main()
