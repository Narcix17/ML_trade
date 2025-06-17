#!/usr/bin/env python3
"""
Script de nettoyage du projet trading system.
Supprime tous les fichiers temporaires, caches et fichiers non nécessaires.
"""

import os
import shutil
import glob
import time
from pathlib import Path
from loguru import logger


def cleanup_pycache():
    """Supprime tous les dossiers __pycache__."""
    logger.info("🧹 Nettoyage des dossiers __pycache__...")
    
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    logger.info(f"✅ Supprimé: {cache_path}")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la suppression de {cache_path}: {e}")


def cleanup_temp_files():
    """Supprime les fichiers temporaires."""
    logger.info("🗑️ Nettoyage des fichiers temporaires...")
    
    temp_patterns = [
        '*.tmp',
        '*.temp',
        '*.bak',
        '*.backup',
        '*.swp',
        '*.swo',
        '*~'
    ]
    
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                os.remove(file_path)
                logger.info(f"✅ Supprimé: {file_path}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la suppression de {file_path}: {e}")


def cleanup_logs():
    """Nettoie les logs anciens."""
    logger.info("📝 Nettoyage des logs...")
    
    log_patterns = [
        'logs/*.log',
        'logs/*.txt',
        '*.log'
    ]
    
    for pattern in log_patterns:
        for file_path in glob.glob(pattern):
            try:
                # Garder seulement les logs récents (moins de 7 jours)
                if os.path.getmtime(file_path) < (time.time() - 7 * 24 * 3600):
                    os.remove(file_path)
                    logger.info(f"✅ Supprimé (ancien): {file_path}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la suppression de {file_path}: {e}")


def cleanup_reports():
    """Nettoie les rapports générés."""
    logger.info("📊 Nettoyage des rapports...")
    
    report_patterns = [
        'reports/*.png',
        'reports/*.jpg',
        'reports/*.pdf',
        'reports/*.html'
    ]
    
    for pattern in report_patterns:
        for file_path in glob.glob(pattern):
            try:
                # Garder seulement les rapports récents (moins de 30 jours)
                if os.path.getmtime(file_path) < (time.time() - 30 * 24 * 3600):
                    os.remove(file_path)
                    logger.info(f"✅ Supprimé (ancien): {file_path}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la suppression de {file_path}: {e}")


def cleanup_model_checkpoints():
    """Nettoie les checkpoints de modèles anciens."""
    logger.info("🤖 Nettoyage des checkpoints de modèles...")
    
    checkpoint_patterns = [
        'models/ppo/*_steps.zip',
        'models/ppo_smoteenn/*_steps.zip'
    ]
    
    for pattern in checkpoint_patterns:
        for file_path in glob.glob(pattern):
            try:
                # Garder seulement les checkpoints récents (moins de 7 jours)
                if os.path.getmtime(file_path) < (time.time() - 7 * 24 * 3600):
                    os.remove(file_path)
                    logger.info(f"✅ Supprimé (ancien): {file_path}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la suppression de {file_path}: {e}")


def cleanup_data_files():
    """Nettoie les fichiers de données temporaires."""
    logger.info("📊 Nettoyage des fichiers de données...")
    
    data_patterns = [
        'data/*.csv',
        'data/*.json',
        'data/*.parquet',
        'data/*.h5'
    ]
    
    for pattern in data_patterns:
        for file_path in glob.glob(pattern):
            try:
                # Garder seulement les données récentes (moins de 7 jours)
                if os.path.getmtime(file_path) < (time.time() - 7 * 24 * 3600):
                    os.remove(file_path)
                    logger.info(f"✅ Supprimé (ancien): {file_path}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la suppression de {file_path}: {e}")


def cleanup_ide_files():
    """Nettoie les fichiers d'IDE."""
    logger.info("💻 Nettoyage des fichiers d'IDE...")
    
    ide_patterns = [
        '.vscode/',
        '.idea/',
        '*.swp',
        '*.swo',
        '*~'
    ]
    
    for pattern in ide_patterns:
        if pattern.endswith('/'):
            # Dossier
            if os.path.exists(pattern):
                try:
                    shutil.rmtree(pattern)
                    logger.info(f"✅ Supprimé: {pattern}")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la suppression de {pattern}: {e}")
        else:
            # Fichier
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    os.remove(file_path)
                    logger.info(f"✅ Supprimé: {file_path}")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la suppression de {file_path}: {e}")


def cleanup_os_files():
    """Nettoie les fichiers système."""
    logger.info("🖥️ Nettoyage des fichiers système...")
    
    os_patterns = [
        '.DS_Store',
        '.DS_Store?',
        '._*',
        '.Spotlight-V100',
        '.Trashes',
        'ehthumbs.db',
        'Thumbs.db'
    ]
    
    for pattern in os_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                os.remove(file_path)
                logger.info(f"✅ Supprimé: {file_path}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la suppression de {file_path}: {e}")


def get_directory_size(path):
    """Calcule la taille d'un répertoire."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                pass
    return total_size


def print_project_stats():
    """Affiche les statistiques du projet."""
    logger.info("📊 Statistiques du projet:")
    
    # Taille totale
    total_size = get_directory_size('.')
    logger.info(f"📁 Taille totale: {total_size / (1024*1024):.2f} MB")
    
    # Nombre de fichiers Python
    py_files = len(glob.glob('**/*.py', recursive=True))
    logger.info(f"🐍 Fichiers Python: {py_files}")
    
    # Nombre de dossiers
    dirs = len([d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')])
    logger.info(f"📁 Dossiers: {dirs}")
    
    # Taille des modèles
    if os.path.exists('models'):
        models_size = get_directory_size('models')
        logger.info(f"🤖 Taille des modèles: {models_size / (1024*1024):.2f} MB")
    
    # Taille des données
    if os.path.exists('data'):
        data_size = get_directory_size('data')
        logger.info(f"📊 Taille des données: {data_size / (1024*1024):.2f} MB")


def main():
    """Fonction principale de nettoyage."""
    logger.info("🚀 DÉMARRAGE DU NETTOYAGE DU PROJET")
    
    # Nettoyage des différents types de fichiers
    cleanup_pycache()
    cleanup_temp_files()
    cleanup_logs()
    cleanup_reports()
    cleanup_model_checkpoints()
    cleanup_data_files()
    cleanup_ide_files()
    cleanup_os_files()
    
    # Affichage des statistiques
    print_project_stats()
    
    logger.info("✅ NETTOYAGE TERMINÉ AVEC SUCCÈS!")
    logger.info("💡 Conseil: Exécutez ce script régulièrement pour maintenir le projet propre")


if __name__ == "__main__":
    main() 