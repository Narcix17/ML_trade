"""
Servo Controller - Contrôleur principal du système de trading servo.

Orchestre:
1. Trading Servo
2. Risk Manager
3. Performance Monitor
4. Alert System
5. Configuration Management
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any
from loguru import logger
from datetime import datetime
import yaml
import os
import signal
import sys

from servo.trading_servo import TradingServo
from servo.risk_manager import RiskManager

class ServoController:
    """Contrôleur principal du servo de trading."""
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "paper"):
        """
        Initialise le contrôleur.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            mode: Mode de trading ("paper", "live", "backtest")
        """
        # Chargement de la configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.mode = mode
            
        # Composants du servo
        self.trading_servo = None
        self.risk_manager = None
        
        # État du système
        self.is_running = False
        self.start_time = None
        
        # Threads
        self.main_thread = None
        self.monitoring_thread = None
        
        # Gestion des signaux
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Servo Controller initialisé en mode {mode}")
        
    def _signal_handler(self, signum, frame):
        """Gestionnaire de signaux système."""
        logger.info(f"Signal reçu: {signum}")
        self.stop()
        
    def start(self) -> None:
        """Démarre le système de trading."""
        if self.is_running:
            logger.warning("Le système est déjà en cours d'exécution")
            return
            
        try:
            logger.info("🚀 Démarrage du système de trading servo...")
            
            # Initialisation des composants
            self._initialize_components()
            
            # Vérifications préliminaires
            if not self._preflight_checks():
                logger.error("❌ Échec des vérifications préliminaires")
                return
                
            # Démarrage du système
            self.is_running = True
            self.start_time = datetime.now()
            
            # Démarrage des threads
            self._start_threads()
            
            logger.info("✅ Système de trading servo démarré avec succès")
            
            # Boucle principale
            self._main_loop()
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage: {e}")
            self.stop()
            
    def _initialize_components(self) -> None:
        """Initialise les composants du système."""
        logger.info("Initialisation des composants...")
        
        # Risk Manager - pass config dict
        self.risk_manager = RiskManager(self.config)
        logger.info("✅ Risk Manager initialisé")
        
        # Trading Servo - pass config dict and mode
        self.trading_servo = TradingServo(self.config, self.mode)
        logger.info("✅ Trading Servo initialisé")
        # Start the main trading loop in a background thread
        self.trading_servo_thread = threading.Thread(target=self.trading_servo.start, daemon=True)
        self.trading_servo_thread.start()
        logger.info("🟢 Trading Servo main loop started in background thread")
        
    def _preflight_checks(self) -> bool:
        """Vérifications préliminaires avant démarrage."""
        logger.info("Vérifications préliminaires...")
        
        # Vérification de la configuration
        if not self._check_configuration():
            return False
            
        # Vérification des modèles
        if not self._check_models():
            return False
            
        # Vérification de la connexion broker
        if not self._check_broker_connection():
            return False
            
        # Vérification des permissions
        if not self._check_permissions():
            return False
            
        logger.info("✅ Toutes les vérifications préliminaires réussies")
        return True
        
    def _check_configuration(self) -> bool:
        """Vérifie la configuration."""
        required_sections = ['broker', 'trading', 'execution', 'monitoring']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Section manquante dans la configuration: {section}")
                return False
                
        # Vérification des paramètres critiques
        broker_config = self.config['broker']
        if not broker_config.get('login') or not broker_config.get('server'):
            logger.error("Configuration broker incomplète")
            return False
            
        return True
        
    def _check_models(self) -> bool:
        """Vérifie la présence des modèles entraînés."""
        model_files = [
            "models/saved/ml_model.joblib",
            "models/saved/regime_detector.joblib",
            "models/saved/feature_engineer.joblib"
        ]
        
        for model_file in model_files:
            if not os.path.exists(model_file):
                logger.warning(f"Modèle non trouvé: {model_file}")
                # Continue sans arrêter le système
                
        return True
        
    def _check_broker_connection(self) -> bool:
        """Vérifie la connexion au broker."""
        try:
            # Test de connexion via le trading servo
            if self.trading_servo and self.trading_servo.order_manager:
                account_info = self.trading_servo.order_manager.get_account_info()
                if account_info:
                    logger.info(f"✅ Connexion broker OK - Compte: {account_info.get('login', 'N/A')}")
                    return True
                    
            logger.warning("⚠️ Connexion broker non vérifiée")
            return True  # Continue même sans vérification
            
        except Exception as e:
            logger.error(f"❌ Erreur de connexion broker: {e}")
            return False
            
    def _check_permissions(self) -> bool:
        """Vérifie les permissions système."""
        # Vérification des dossiers de logs
        log_dirs = ['logs', 'reports', 'data']
        
        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir)
                    logger.info(f"Dossier créé: {log_dir}")
                except Exception as e:
                    logger.error(f"Impossible de créer le dossier {log_dir}: {e}")
                    return False
                    
        return True
        
    def _start_threads(self) -> None:
        """Démarre les threads de monitoring."""
        # Thread de monitoring principal
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("✅ Threads de monitoring démarrés")
        
    def _main_loop(self) -> None:
        """Boucle principale du contrôleur."""
        logger.info("🔄 Démarrage de la boucle principale...")
        
        try:
            while self.is_running:
                # Vérification du statut du système
                if not self._check_system_status():
                    logger.error("❌ Problème détecté dans le système")
                    break
                    
                # Mise à jour des métriques
                self._update_system_metrics()
                
                # Log du statut
                self._log_system_status()
                
                time.sleep(10)  # Vérification toutes les 10 secondes
                
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {e}")
        finally:
            self.stop()
            
    def _monitoring_loop(self) -> None:
        """Boucle de monitoring."""
        while self.is_running:
            try:
                # Monitoring du risk manager
                if self.risk_manager:
                    risk_metrics = self.risk_manager.get_risk_metrics()
                    risk_level = self.risk_manager.get_risk_level()
                    
                    # Alertes de risque
                    if not self.risk_manager.is_risk_ok:
                        logger.warning(f"🚨 ALERTE RISQUE: Niveau {risk_level.name}")
                        
                # Monitoring du trading servo
                if self.trading_servo:
                    servo_status = self.trading_servo.get_status()
                    
                    # Vérification des positions
                    positions = self.trading_servo.get_positions()
                    if len(positions) > 0:
                        logger.info(f"Positions ouvertes: {len(positions)}")
                        
                time.sleep(30)  # Monitoring toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Erreur dans le monitoring: {e}")
                
    def _check_system_status(self) -> bool:
        """Vérifie le statut du système."""
        # Vérification du risk manager
        if self.risk_manager and not self.risk_manager.is_risk_ok:
            logger.error("❌ Risk Manager signale un problème")
            return False
            
        # Vérification du trading servo
        if self.trading_servo and not self.trading_servo.is_running:
            logger.error("❌ Trading Servo n'est plus en cours d'exécution")
            return False
            
        return True
        
    def _update_system_metrics(self) -> None:
        """Met à jour les métriques système."""
        # Métriques de performance
        if self.trading_servo:
            performance = self.trading_servo.performance_metrics
            # TODO: Stocker et analyser les métriques
            
        # Métriques de risque
        if self.risk_manager:
            risk_metrics = self.risk_manager.get_risk_metrics()
            # TODO: Stocker et analyser les métriques de risque
            
    def _log_system_status(self) -> None:
        """Log le statut du système."""
        if not self.is_running:
            return
            
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        # Statut du trading servo
        servo_status = "🟢" if (self.trading_servo and self.trading_servo.is_running) else "🔴"
        
        # Statut du risk manager
        risk_status = "🟢" if (self.risk_manager and self.risk_manager.is_risk_ok) else "🔴"
        
        # Positions ouvertes
        positions_count = len(self.trading_servo.get_positions()) if self.trading_servo else 0
        
        logger.info(
            f"📊 STATUT SYSTÈME | "
            f"Uptime: {uptime} | "
            f"Servo: {servo_status} | "
            f"Risk: {risk_status} | "
            f"Positions: {positions_count}"
        )
        
    def stop(self) -> None:
        """Arrête le système de trading."""
        if not self.is_running:
            return
            
        logger.info("🛑 Arrêt du système de trading servo...")
        
        self.is_running = False
        
        # Arrêt du trading servo
        if self.trading_servo:
            self.trading_servo.stop()
            logger.info("✅ Trading Servo arrêté")
            
        # Arrêt du risk manager
        if self.risk_manager:
            self.risk_manager.shutdown()
            logger.info("✅ Risk Manager arrêté")
            
        # Attente des threads
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
            
        logger.info("✅ Système de trading servo arrêté")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Retourne le statut complet du système."""
        status = {
            'is_running': self.is_running,
            'start_time': self.start_time,
            'uptime': datetime.now() - self.start_time if self.start_time else None
        }
        
        # Statut du trading servo
        if self.trading_servo:
            status['trading_servo'] = self.trading_servo.get_status()
            
        # Statut du risk manager
        if self.risk_manager:
            status['risk_manager'] = {
                'is_risk_ok': self.risk_manager.is_risk_ok,
                'risk_level': self.risk_manager.get_risk_level().name,
                'risk_metrics': self.risk_manager.get_risk_metrics().__dict__
            }
            
        return status
        
    def emergency_stop(self) -> None:
        """Arrêt d'urgence du système."""
        logger.critical("🚨 ARRÊT D'URGENCE DU SYSTÈME")
        
        # Force l'arrêt du trading
        if self.trading_servo:
            self.trading_servo._close_all_positions()
            
        # Force l'arrêt du risk manager
        if self.risk_manager:
            self.risk_manager.force_stop_trading()
            
        # Arrêt complet
        self.stop()
        
    def restart(self) -> None:
        """Redémarre le système."""
        logger.info("🔄 Redémarrage du système...")
        
        self.stop()
        time.sleep(5)  # Attente avant redémarrage
        self.start()
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance."""
        report = {
            'timestamp': datetime.now(),
            'system_status': self.get_system_status(),
            'performance_metrics': {},
            'risk_metrics': {},
            'positions': {},
            'signals': {}
        }
        
        # Métriques de performance du trading servo
        if self.trading_servo:
            report['performance_metrics'] = self.trading_servo.performance_metrics
            report['positions'] = {
                symbol: {
                    'side': pos.side,
                    'volume': pos.volume,
                    'entry_price': pos.entry_price,
                    'entry_time': pos.entry_time,
                    'regime': pos.regime,
                    'confidence': pos.signal_confidence
                }
                for symbol, pos in self.trading_servo.get_positions().items()
            }
            report['signals'] = {
                'total_signals': len(self.trading_servo.get_signal_history()),
                'last_signal': self.trading_servo.get_signal_history()[-1].__dict__ if self.trading_servo.get_signal_history() else None
            }
            
        # Métriques de risque
        if self.risk_manager:
            report['risk_metrics'] = self.risk_manager.get_risk_metrics().__dict__
            
        return report

def main():
    """Fonction principale pour démarrer le servo controller."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Servo Controller')
    parser.add_argument('--config', default='config.yaml', help='Chemin vers le fichier de configuration')
    parser.add_argument('--log-level', default='INFO', help='Niveau de log')
    
    args = parser.parse_args()
    
    # Configuration du logger
    logger.remove()
    logger.add(
        sys.stdout,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/servo_controller.log",
        rotation="10 MB",
        retention="7 days",
        level=args.log_level
    )
    
    # Démarrage du contrôleur
    controller = ServoController(args.config)
    
    try:
        controller.start()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        controller.emergency_stop()
    finally:
        controller.stop()

if __name__ == "__main__":
    main() 