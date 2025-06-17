#!/usr/bin/env python3
"""
Script de monitoring pour le système de trading live.
Permet de surveiller les performances sans interrompre le trading.
"""

import time
import json
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5

class TradingMonitor:
    def __init__(self):
        self.connect_mt5()
    
    def connect_mt5(self):
        """Connexion à MT5."""
        if not mt5.initialize():
            print("❌ Échec de l'initialisation MT5")
            return False
        
        # Connexion au compte
        if not mt5.login(login=92887059, server="MetaQuotes-Demo"):
            print("❌ Échec de la connexion au compte")
            return False
        
        print("✅ Connecté à MT5")
        return True
    
    def get_account_info(self):
        """Récupère les informations du compte."""
        account_info = mt5.account_info()
        if account_info is None:
            return None
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage
        }
    
    def get_open_positions(self):
        """Récupère les positions ouvertes."""
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        positions_data = []
        for pos in positions:
            positions_data.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == 0 else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'sl': pos.sl,
                'tp': pos.tp,
                'time': datetime.fromtimestamp(pos.time)
            })
        
        return positions_data
    
    def get_trade_history(self, days=1):
        """Récupère l'historique des trades."""
        from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        from_date = from_date.replace(day=from_date.day - days)
        
        deals = mt5.history_deals_get(from_date, datetime.now())
        if deals is None:
            return []
        
        deals_data = []
        for deal in deals:
            deals_data.append({
                'ticket': deal.ticket,
                'symbol': deal.symbol,
                'type': 'BUY' if deal.type == 0 else 'SELL',
                'volume': deal.volume,
                'price': deal.price,
                'profit': deal.profit,
                'time': datetime.fromtimestamp(deal.time)
            })
        
        return deals_data
    
    def calculate_performance_metrics(self, deals):
        """Calcule les métriques de performance."""
        if not deals:
            return {}
        
        df = pd.DataFrame(deals)
        
        total_trades = len(df)
        winning_trades = len(df[df['profit'] > 0])
        losing_trades = len(df[df['profit'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = df['profit'].sum()
        avg_profit = df['profit'].mean()
        max_profit = df['profit'].max()
        max_loss = df['profit'].min()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss
        }
    
    def display_status(self):
        """Affiche le statut complet du système."""
        print("\n" + "="*60)
        print(f"📊 MONITORING DU SYSTÈME DE TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Informations du compte
        account_info = self.get_account_info()
        if account_info:
            print(f"\n💰 INFORMATIONS DU COMPTE:")
            print(f"   Balance: {account_info['balance']:.2f} USD")
            print(f"   Equity: {account_info['equity']:.2f} USD")
            print(f"   Profit: {account_info['profit']:.2f} USD")
            print(f"   Marge libre: {account_info['free_margin']:.2f} USD")
            print(f"   Levier: 1:{account_info['leverage']}")
        
        # Positions ouvertes
        positions = self.get_open_positions()
        print(f"\n📈 POSITIONS OUVERTES ({len(positions)}):")
        if positions:
            for pos in positions:
                print(f"   {pos['symbol']} {pos['type']} {pos['volume']} lots @ {pos['price_open']:.5f}")
                print(f"     Profit: {pos['profit']:.2f} USD | SL: {pos['sl']:.5f} | TP: {pos['tp']:.5f}")
        else:
            print("   Aucune position ouverte")
        
        # Historique des trades
        deals = self.get_trade_history(days=1)
        print(f"\n📋 HISTORIQUE DES TRADES (24h):")
        if deals:
            for deal in deals[-5:]:  # Derniers 5 trades
                print(f"   {deal['time'].strftime('%H:%M:%S')} - {deal['symbol']} {deal['type']} "
                      f"{deal['volume']} lots @ {deal['price']:.5f} | Profit: {deal['profit']:.2f} USD")
        
        # Métriques de performance
        metrics = self.calculate_performance_metrics(deals)
        if metrics:
            print(f"\n📊 MÉTRIQUES DE PERFORMANCE:")
            print(f"   Total trades: {metrics['total_trades']}")
            print(f"   Taux de réussite: {metrics['win_rate']:.1f}%")
            print(f"   Profit total: {metrics['total_profit']:.2f} USD")
            print(f"   Profit moyen: {metrics['avg_profit']:.2f} USD")
            print(f"   Plus gros gain: {metrics['max_profit']:.2f} USD")
            print(f"   Plus grosse perte: {metrics['max_loss']:.2f} USD")
        
        print("\n" + "="*60)
    
    def run_monitoring(self, interval=60):
        """Lance le monitoring en continu."""
        print(f"🚀 Démarrage du monitoring (intervalle: {interval}s)")
        print("Appuyez sur Ctrl+C pour arrêter")
        
        try:
            while True:
                self.display_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring arrêté")
        finally:
            mt5.shutdown()

def main():
    monitor = TradingMonitor()
    monitor.run_monitoring(interval=60)  # Mise à jour toutes les 60 secondes

if __name__ == "__main__":
    main() 