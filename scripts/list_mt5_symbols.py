#!/usr/bin/env python3
"""
Liste tous les symboles disponibles et actifs sur MetaTrader 5.
"""
import MetaTrader5 as mt5

# Initialisation MT5
if not mt5.initialize():
    print("Erreur d'initialisation MT5")
    exit(1)

symbols = mt5.symbols_get()
print(f"Nombre total de symboles : {len(symbols)}\n")

print("Symboles actifs (Market Watch) :")
active_symbols = [s.name for s in symbols if s.visible]
for s in active_symbols:
    print(f"  - {s}")

print("\nSymboles disponibles (tous) :")
for s in symbols:
    print(f"  - {s.name}")

mt5.shutdown() 