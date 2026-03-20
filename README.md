# autoquant — autonomiczny optymalizator strategii tradingowych

> **Fork od:** [auditmos/autoquant](https://github.com/auditmos/autoquant)
> **Który jest forkiem od:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

---

## Rodowód projektu

```
karpathy/autoresearch          ← oryginał: autonomiczny agent trenuje modele GPT
    └── auditmos/autoquant     ← adaptacja: agent optymalizuje strategie tradingowe (SPY+BTC+ETH)
            └── popek1990/autoquant  ← ten fork: pełne krypto (BTC/ETH/XMR/SOL/TAO), GPU, 1H, LSTM
```

**[karpathy/autoresearch](https://github.com/karpathy/autoresearch)** (43k gwiazdek) — oryginalny framework Andreja Karpathy'ego, w którym autonomiczny agent AI iteracyjnie modyfikuje kod trenowania modelu językowego GPT, uruchamia 5-minutowe eksperymenty i optymalizuje metrykę `val_bpb` (bits per byte — im niżej, tym lepiej).

**[auditmos/autoquant](https://github.com/auditmos/autoquant)** — pierwsza adaptacja do tradingu: ten sam protokół agenta, ale zamiast trenować LLM — optymalizuje strategie tradingowe na danych SPY+BTC+ETH (daily). Metryka: composite score (Sharpe + drawdown + return). Rekord auditmos: 0.630.

**Ten fork (popek1990/autoquant)** — znacznie rozbudowana wersja:
- **5 kryptowalut** na danych 1H: BTC, ETH, XMR, SOL, TAO
- **Dane makro:** SPY, QQQ, UUP, GLD, VIXY, FED_RATE, CPI, krzywa rentowności, funding rates futures, sentyment newsów
- **Sieć neuronowa LSTM** na GPU (RTX 4090) zamiast prostych reguł
- **Rekord score: 3.456** (vs 0.408 na starcie — wzrost 8×)

---

## Jak to działa

Agent AI (Claude) w pętli:
1. Czyta `strategy.py` i historię wyników (`results.tsv`)
2. Wymyśla ulepszenie architektury / cech / hiperparametrów
3. Modyfikuje `strategy.py`
4. Uruchamia backtest: `python3 strategy.py`
5. Jeśli `score` lepszy → zachowuje zmianę. Jeśli gorszy → cofa i próbuje inaczej
6. Powtarza bez końca

Modyfikowany jest **wyłącznie** plik `strategy.py`. `prepare.py` (dane + silnik backtestów) jest read-only.

---

## Aktualna najlepsza strategia — exp #109 (score: 3.456)

**Konfiguracja:** `LSTM_h384_3L_drop03_target24h_discrete_funding_1H`

```
Model:    LSTM 3-warstwowy, hidden=384, BatchNorm, dropout=0.3
Target:   zwrot ceny za kolejne 24h
Lookback: 168 świec (7 dni na interwale 1H)
Trening:  300 epok, lr=0.002, AdamW wd=0.02, CosineAnnealingLR
Features: 23 (21 technicznych + market_funding + vixy_trend)
Sygnały:  dyskretne progi → pozycja 50/75/100% kapitału
ATR stop: multiplier=1.9, profit_target=3.0×ATR, cooldown=24h
```

### Wyniki per kryptowaluta (walidacja: marzec 2025 – marzec 2026)

| Krypto | Sharpe | Zwrot (val.) | Buy & Hold |
|--------|--------|-------------|-----------|
| **XMR** (Monero) | 7.7 | +1402% | +76% |
| **TAO** (Bittensor) | 7.0 | +1702% | +12% |
| **ETH** (Ethereum) | 5.6 | +368% | +22% |
| **SOL** (Solana) | 4.1 | +254% | -27% |
| **BTC** (Bitcoin) | 3.4 | +74% | -11% |

> ⚠️ Backtest bez kosztów transakcji. Strategia robi ~1100 transakcji/rok/asset → ok. 110% w opłatach (0.1% fee Binance). Przy XMR +1402% wciąż opłacalne, ale przy słabszych assetach warto uwzględnić w kalkulacjach.

---

## Ewolucja wyników

```
Faza 1 — rule-based:     0.408
MLP (sieć neuronowa):    1.064   (+161%)
LSTM 2L h=256:           1.903   (+79%)
LSTM h=384 3L:           2.116
+dropout=0.4:            2.188
+target=24h:             3.341   (+53%)
+dropout=0.3:            3.456   ← REKORD
```

Kluczowe skoki:
- **LSTM >> MLP** (+79%) — sieci rekurencyjne radzą sobie znacznie lepiej na szeregach czasowych
- **target=24h >> target=48h** (+53%) — krótszy horyzont predykcji lepiej generalizuje
- **BatchNorm obowiązkowy** — bez niego LSTM nie stabilizuje się

---

## Dane

**Krypto (ccxt/Binance):** BTC/USDT, ETH/USDT, SOL/USDT, TAO/USDT, XMR/USDT (KuCoin) — interwał 1H. Lista konfigurowalna przez `ASSETS` env var.

**Makro (Alpha Vantage):** SPY, QQQ, UUP, GLD, VIXY (1H ETF); FED_RATE, CPI, TREASURY_10Y/2Y; funding rates BTC/ETH/SOL futures (co 8H); sentyment newsów BTC/ETH (daily)

**Train:** 2023-03 → 2025-03 | **Val:** 2025-03 → 2026-03 | **Cache:** `~/.cache/autoquant/data/`

---

## Struktura plików

```
autoquant/
├── Dockerfile          ← nvidia/cuda + uv + Claude CLI
├── docker-compose.yml  ← GPU, bind-mount, env passthrough
├── entrypoint.sh       ← tryby: login, agent, live, default
├── notify.sh           ← generic webhook (NOTIFY_ENDPOINT)
├── pyproject.toml      ← zależności (uv/pip)
├── .env.example        ← szablon zmiennych środowiskowych
├── prepare.py          ← pobieranie danych + silnik backtestów + scoring (read-only)
├── strategy.py         ← strategia tradingowa — tu agent wprowadza zmiany
├── program.md          ← protokół agenta + historia odkryć
├── STRATEGIA.md        ← opis strategii w języku naturalnym
├── progress.py         ← wizualizacja progresu eksperymentów
├── live_signals.py     ← sygnały live (bez ponownego treningu, ładuje modele .pt)
├── test_robustness.py  ← test stabilności na wielu seedach
├── results.tsv         ← log wszystkich eksperymentów
├── logi/               ← pełne logi każdego uruchomienia (run_NNN.log)
└── testy/              ← szczegółowe notatki z eksperymentów
```

---

## Szybki start (Docker)

```bash
# Wymagania: Docker, nvidia-container-toolkit, Docker Compose v2
cp .env.example .env
# uzupełnij ALPHA_VANTAGE_API_KEY (reszta opcjonalna)

docker compose build
docker compose run autoquant login    # jednorazowa auth Claude
docker compose run autoquant agent    # pętla autonomiczna
```

### Warianty

```bash
# Tylko wybrane assety:
ASSETS=BTC,ETH docker compose run autoquant agent

# Inna karta GPU:
GPU_ID=1 docker compose run autoquant agent

# Inny model Claude:
CLAUDE_MODEL=claude-sonnet-4-5 docker compose run autoquant agent

# Webhook notyfikacje:
NOTIFY_ENDPOINT=https://my-api.com/hook docker compose run autoquant agent

# Live signals:
docker compose run autoquant live

# Dowolny skrypt:
docker compose run autoquant progress.py
```

### Webhook notyfikacje

`notify.sh` wysyła POST JSON na `NOTIFY_ENDPOINT` (no-op gdy pusty). Auth opcjonalny przez `NOTIFY_BEARER`.

**Payload:**
```json
{
  "event": "agent_start | agent_restart | agent_stuck",
  "message": "Agent starting (assets: BTC,ETH,XMR,SOL,TAO)",
  "timestamp": "2026-03-20T12:00:00Z",
  "container": "single-autoquant"
}
```

**Eventy:**

| Event | Kiedy |
|-------|-------|
| `agent_start` | Start pętli agenta |
| `agent_restart` | Koniec iteracji Claude (z exit code) |
| `agent_stuck` | 5 iteracji bez zmiany w results.tsv |

### Bez Dockera (lokalnie)

```bash
# Wymagania: Python 3.10+, uv, CUDA GPU
uv sync
uv run strategy.py
uv run live_signals.py --loop
```

---

## Metryka score

```
Score = (35% × Sharpe + 20% × Sortino + 20% × (1 - maxDrawdown) + 15% × return + 10% × win_rate)
        × trade_penalty (min 50 transakcji)
        × consistency (train vs val)
```

> ⚠️ **target < 24h** generuje nierealistycznie wysokie score w bezkosztowym backteście (artefakt compoundingu). Używaj **target ≥ 24h**.

---

## Hardware

Testowane na **RTX 4090** (24GB VRAM, Ada Lovelace sm_89):
- PyTorch z CUDA, bfloat16, `torch.compile` — działają
- Flash Attention 3 (FA3) — **nie działa** na sm_89, używamy SDPA
- Backtesty: CPU (~2s), trening LSTM: GPU (~5 min / 300 epok)

---

## Licencja

MIT — zgodnie z oryginalnym projektem [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
