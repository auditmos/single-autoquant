# Program agenta — Autoquant

Jesteś autonomicznym agentem optymalizującym strategie tradingowe na krypto.

## Twoje zadanie

Modyfikuj `strategy.py` aby maksymalizować metrykę `score` (wyższy = lepiej).
Po każdej modyfikacji uruchom `python3 strategy.py` i sprawdź wynik.

## Zasady

1. Modyfikuj **wyłącznie** `strategy.py` — `prepare.py` jest read-only
2. Nie dodawaj nowych zależności (pip install)
3. Każdy eksperyment musi się zakończyć w ~2 min
4. Wyniki zapisują się automatycznie do `results.tsv`
5. Zmień zmienną `OPIS` w strategy.py na krótki opis co zmieniłeś

## Protokół eksperymentu

1. Przeczytaj aktualny `strategy.py` i `results.tsv`
2. Wymyśl ulepszenie
3. Zmodyfikuj `strategy.py`
4. Uruchom: `python3 strategy.py`
5. Sprawdź `score:` w outputcie
6. Jeśli score lepszy niż poprzedni najlepszy → zachowaj, idź do kroku 2
7. Jeśli score gorszy → cofnij zmiany (`git checkout strategy.py`), idź do kroku 2
8. Powtarzaj w nieskończoność

## Obecny baseline

- Strategia: Ichimoku + RSI + MACD + ATR trailing stop + barometry makro
- Score: ~0.06
- Timeframe: 4H
- Aktywa: BTC, ETH, XMR, SOL, TAO (long/short)
- Barometry: SPY, QQQ, UUP (kontekst, nie handlowane)

## Co możesz zmieniać w strategy.py

- Parametry wskaźników (periody RSI, MACD, Ichimoku, ATR multiplier)
- Dodawać wskaźniki: Bollinger Bands, Stochastic, OBV, VWAP, Williams %R
- Regime detection (wykrywanie trendu vs range)
- Ensemble voting (łączenie wielu sygnałów)
- Sieci neuronowe PyTorch (GPU RTX 4090 dostępne)
- Multi-timeframe (łączenie sygnałów z 1H i 4H)
- Lepsze wejścia/wyjścia (breakout, pullback)
- Optymalizacja ATR stop multiplier
- Volume analysis
- Korelacje między aktywami

## Czego NIE zmieniać

- Sygnatura `strategy(df, context) -> pd.Series`
- Import z prepare.py
- Sekcja Runner (if __name__ == "__main__")
- Zapis do results.tsv

## Metryka score

Composite: 35% Sharpe + 20% Sortino + 20% (1-drawdown) + 15% return + 10% win_rate
× trade_penalty (min 50 transakcji) × consistency (train vs val)

Ważne: score musi być dobry na VALIDATION (2025-03 do 2026-03), nie tylko train.
Unikaj overfittingu — duża różnica train vs val to zły znak.

## PRIORYTET: Sieci neuronowe PyTorch (GPU RTX 4090)

Masz do dyspozycji RTX 4090 24GB — UŻYWAJ JEJ. Łącz wskaźniki techniczne z modelami neuronowymi.

Architektura hybrydowa — najlepsze podejście:
1. Wskaźniki (Ichimoku, RSI, MACD, ATR, volume) jako **features**
2. Model neuronowy jako **decydent** — uczy się na train, generuje sygnały na val
3. ATR trailing stop nadal chroni pozycje

Przykładowe modele do wypróbowania:
- **LSTM/GRU** — dobry na szeregi czasowe, lookback 20-50 świec
- **1D CNN** — szybki, łapie lokalne wzorce cenowe
- **Transformer** — attention na historię, dobry na dłuższe zależności
- **Ensemble** — kilka małych modeli głosuje razem

Wskazówki implementacji:
- `import torch; device = torch.device("cuda")` — zawsze GPU
- Features: close_pct, rsi, macd_hist, atr_norm, volume_ratio, cloud_dist, tenkan_kijun_diff, spy_trend, uup_trend
- Normalizacja: StandardScaler na train, ten sam transform na val (unikaj data leakage!)
- Train na TRAIN_START:TRAIN_END, predykcja na val
- Output modelu: sygnał [-1, 1] (continuous), potem threshold na pozycje
- Regularyzacja: dropout, weight decay, early stopping — KRYTYCZNE żeby nie overfitować
- Małe modele (1-3 warstwy, 32-128 neuronów) — lepsze niż duże na tym zbiorze
- Trenuj per-asset lub jeden model na wszystkie (z asset embedding)

## Wskazówki ogólne

- Małe zmiany, jeden parametr naraz — łatwiej ocenić co działa
- XMR traci -89% na train — duży potencjał poprawy
- BTC train jest lekko stratny — też do poprawy
- ETH i SOL działają dobrze — nie psuj tego co działa
- ATR trailing stop bardzo pomógł — rozważ inne stopy (Chandelier, Parabolic SAR)
- Proporcjonalne pozycje (0.5/1.0) pomagają — rozważ więcej gradacji
- Jeśli model neuronowy daje gorszy score niż wskaźniki — spróbuj ensemble (model + wskaźniki razem)
