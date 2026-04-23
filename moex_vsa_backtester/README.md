# VSA Backtester for MOEX

Алгоритмический движок для бэктестинга стратегий Volume Spread Analysis на Московской бирже.

## Требования

- Python 3.10+
- MySQL/MariaDB

## Установка

```bash
pip install -r requirements.txt
cp .env.example .env
```

Настройте `.env` файл:
```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=moex
DB_USER=root
DB_PASSWORD=your_password
RISK_PER_TRADE=0.01
RR_RATIO=2.0
ALLOW_SHORT=false
TICKER_LIST=SBER,VTBR,LKOH,NVTK
```

## Режимы запуска

### Бэктест
```bash
python main.py --backtest --ticker SBER --tf H1 --start 2023-01-01 --end 2024-01-01
```

### Мониторинг (сканер)
```bash
python main.py --scan
python main.py --scan --interval 60
```

### Виртуальная торговля
```bash
python main.py --virtual --capital 1000000 --risk 0.01
```

### Обучение AI модели
```bash
# Для конкретного тикера
python main.py --backtest --ticker SBER --train-ai

# Глобальная модель (все инструменты + SHORT)
python main.py --train-global --start 2023-01-01 --end 2024-01-01
```

### Отчет виртуальной торговли
```bash
python main.py --report
```

## Аргументы командной строки

### Основные
| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--ticker` | Тикер (SBER, GAZP, PLZL...) | - |
| `--tf` | Таймфрейм (D1, H1) | H1 |
| `--start` | Дата начала | 2023-01-01 |
| `--end` | Дата окончания | 2024-01-01 |
| `--capital` | Начальный капитал | 1000000 |
| `--risk` | Риск на сделку (0.01 = 1%) | 0.01 |
| `--rr` | Risk-Reward ratio | 2.0 |

### Режимы
| Параметр | Описание |
|----------|----------|
| `--backtest` | Запуск бэктеста |
| `--scan` | Запуск сканера (мониторинг) |
| `--virtual` | Виртуальная торговля |
| `--train-ai` | Обучить AI модель для тикера |
| `--train-global` | Обучить глобальную AI модель на всех инструментах |
| `--report` | Сгенерировать ежедневный отчет |

### Дополнительные
| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--allow-short` | Разрешить SHORT сделки | false |
| `--ai-model` | Путь к AI модели | models/trade_model.pt |
| `--ai-threshold` | Порог вероятности AI | 0.6 |
| `--ai-filter` | Фильтровать сигналы по AI | false |
| `--train-ai` | Обучить AI модель | false |
| `--interval` | Интервал проверки (сек) | 60 |
| `--max-positions` | Макс. позиций в вирт. торговле | 5 |

## Структура проекта

```
moex_vsa_backtester/
├── config/         # Конфигурация
├── db/            # Загрузчик OHLCV
├── core/          # VSA движок, риск-менеджмент, виртуальный трейдер
├── backtest/      # Бэктестер, метрики
├── ai/            # AI модель, датасет, инференс
├── scanner/       # Сканер сигналов, шедулер
├── tests/         # Юнит-тесты
└── logs/          # Логи, сделки, отчеты
```

## Метрики бэктеста

После бэктеста выводятся:
- Total Trades
- Win Rate (%)
- Profit Factor
- Max Drawdown
- Avg RR
- Total PnL

## Логи

| Файл | Описание |
|------|----------|
| `logs/vsa_backtest.log` | Лог бэктеста |
| `logs/signals.log` | Лог сигналов |
| `logs/trades_TICKER_TF.csv` | Сделки |
| `logs/virtual_trading/journal.csv` | Журнал виртуальных сделок |
| `logs/virtual_trading/reports/` | Ежедневные отчеты |

## Тесты

```bash
pytest -v
```

## Примеры использования

### Бэктест с SHORT
```bash
python main.py --backtest --ticker SBER --allow-short --start 2023-01-01 --end 2024-01-01
```

### Мониторинг с виртуальной торговлей
```bash
python main.py --virtual --capital 500000 --max-positions 3
```

### Обучение и бэктест с AI
```bash
python main.py --backtest --ticker SBER --train-ai --ai-filter
```

## Лицензия

MIT