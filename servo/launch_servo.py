#!/usr/bin/env python3
"""
Launcher script pour le système de trading servo.

Usage:
    python servo/launch_servo.py --config config.yaml --mode live
    python servo/launch_servo.py --config config.yaml --mode paper
    python servo/launch_servo.py --config config.yaml --mode backtest
"""

import argparse
import sys
import os
from loguru import logger
from datetime import datetime

# Ajout du répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo.servo_controller import ServoController

def setup_logging(log_level: str = "INFO"):
    """Configure le système de logging."""
    # Suppression des handlers par défaut
    logger.remove()
    
    # Handler pour la console
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Handler pour les fichiers
    logger.add(
        f"logs/servo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="50 MB",
        retention="30 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

def check_prerequisites():
    """Vérifie les prérequis avant le démarrage."""
    logger.info("🔍 Vérification des prérequis...")
    
    # Vérification des modèles entraînés
    model_files = [
        "models/saved/ml_model.joblib",
        "models/saved/regime_detector.joblib",
        "models/saved/feature_engineer.joblib"
    ]
    
    missing_models = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        logger.warning("⚠️ Modèles manquants:")
        for model in missing_models:
            logger.warning(f"   - {model}")
        logger.warning("💡 Exécutez d'abord: python main.py")
        return False
    
    # Vérification des dossiers
    required_dirs = ["logs", "reports", "data"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info(f"📁 Dossier créé: {dir_name}")
    
    logger.info("✅ Tous les prérequis sont satisfaits")
    return True

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Trading Servo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python servo/launch_servo.py --mode paper --config config.yaml
  python servo/launch_servo.py --mode live --config config.yaml --log-level DEBUG
  python servo/launch_servo.py --mode backtest --config config.yaml
        """
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Chemin vers le fichier de configuration (défaut: config.yaml)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Mode de trading (défaut: paper)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Niveau de log (défaut: INFO)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode test sans exécution réelle"
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    setup_logging(args.log_level)
    
    # Affichage du banner
    print("=" * 60)
    print("🤖 TRADING SERVO LAUNCHER")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Configuration: {args.config}")
    print(f"Log Level: {args.log_level}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 60)
    
    # Vérification des prérequis
    if not check_prerequisites():
        logger.error("❌ Prérequis non satisfaits")
        sys.exit(1)
    
    # Vérification du fichier de configuration
    if not os.path.exists(args.config):
        logger.error(f"❌ Fichier de configuration non trouvé: {args.config}")
        sys.exit(1)
    
    # Modification de la configuration selon le mode
    if args.mode == "paper":
        logger.info("📋 Mode PAPER TRADING activé")
        # TODO: Modifier la configuration pour le mode paper
    elif args.mode == "live":
        logger.warning("🚨 Mode LIVE TRADING activé - ATTENTION!")
        # TODO: Vérifications supplémentaires pour le mode live
    elif args.mode == "backtest":
        logger.info("📊 Mode BACKTEST activé")
        # TODO: Configuration pour le backtest
    
    # Mode dry-run
    if args.dry_run:
        logger.info("🧪 Mode DRY-RUN activé - Aucune exécution réelle")
    
    try:
        # Démarrage du contrôleur
        logger.info("🚀 Démarrage du Trading Servo...")
        
        controller = ServoController(args.config, args.mode)
        
        # Configuration spécifique au mode
        if args.dry_run:
            # TODO: Configuration pour le mode dry-run
            pass
        
        # Démarrage
        controller.start()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)
    finally:
        logger.info("🛑 Arrêt du système")

if __name__ == "__main__":
    main() 