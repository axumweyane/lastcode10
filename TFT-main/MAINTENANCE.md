# APEX Maintenance Schedule

All commands run from `TFT-main/`.

## Daily (Pre-Market, 8:30 AM ET)

Automated via systemd timer (`apex-health-check.timer`). Manual:

```bash
bash run_all_checks.sh
```

Checks: data audit, model validation, paper trader e2e, infrastructure health.

**Manual review:**
- Check `results/` for any FAIL results
- Review paper trader dashboard: `http://localhost:8010/dashboard`
- Check Grafana dashboards: `http://localhost:3000`

## Weekly (Monday Morning)

### 1. Retrain TFT-Stocks model

```bash
python train_postgres.py --symbols AAPL GOOGL MSFT AMZN NVDA META TSLA \
    --start-date 2024-01-01 --target-type returns --max-epochs 50
```

### 2. Run strategy optimizer

```bash
python optimize_strategies.py --folds 5 --max-samples 100 --output optimization_results.json
```

### 3. Populate supplementary data

```bash
python scripts/populate_tables.py --tables vix fundamentals sentiment
```

### 4. Run full backtest validation

```bash
python validate_backtest.py
```

### 5. Check Docker container health

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
docker system df
```

### 6. Review paper trading performance

```bash
curl -s http://localhost:8010/history | python3 -m json.tool | head -50
curl -s http://localhost:8010/weights | python3 -m json.tool
```

## Biweekly (1st and 15th)

### 1. Database maintenance

```bash
# VACUUM and ANALYZE
psql -h localhost -p 15432 -U apex_user -d apex -c "VACUUM ANALYZE;"

# Check table sizes
psql -h localhost -p 15432 -U apex_user -d apex -c "
    SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
    FROM pg_catalog.pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC LIMIT 10;
"
```

### 2. Log rotation

```bash
# Rotate paper trader logs
journalctl --user -u apex-paper-trader --vacuum-time=14d

# Clean old results
find results/ -name "*.json" -mtime +30 -delete
```

### 3. Docker cleanup

```bash
docker system prune -f
docker volume prune -f
```

### 4. Retrain all TFT models

```bash
# Stocks
python train_postgres.py --symbols AAPL GOOGL MSFT AMZN NVDA META TSLA \
    --start-date 2024-01-01 --target-type returns --max-epochs 50

# Check forex and volatility models are still valid
python validate_models.py
```

## Monthly (1st of Month)

### 1. Full system audit

```bash
python audit_data.py --fix
python validate_models.py
python test_e2e.py
python validate_backtest.py
python test_infra.py
```

### 2. Security review

```bash
# Check for exposed credentials
grep -r "password\|secret\|api_key" .env* --include="*.env*"

# Update Python dependencies
pip list --outdated
pip install --upgrade -r requirements.txt
```

### 3. Backup database

```bash
pg_dump -h localhost -p 15432 -U apex_user apex > backups/apex_$(date +%Y%m%d).sql
```

### 4. Review model performance trends

```bash
# Compare recent results
ls -la results/models_*.json | tail -4
python3 -c "
import json, glob
files = sorted(glob.glob('results/models_*.json'))[-4:]
for f in files:
    d = json.load(open(f))
    print(f'{f}: {d.get(\"summary\", d)}')
"
```

### 5. Verify systemd timers

```bash
systemctl --user list-timers
systemctl --user status apex-paper-trader
systemctl --user status apex-health-check.timer
```

## Emergency Procedures

### Paper trader is down

```bash
# Check status
systemctl --user status apex-paper-trader

# Restart
systemctl --user restart apex-paper-trader

# Check logs
journalctl --user -u apex-paper-trader -n 50 --no-pager
```

### Database connection issues

```bash
# Check container
docker ps | grep timescaledb

# Restart if needed
docker restart apex-timescaledb

# Check connections
psql -h localhost -p 15432 -U apex_user -d apex -c "SELECT count(*) FROM pg_stat_activity;"
```

### Circuit breaker tripped

```bash
# Check status
curl -s http://localhost:8010/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('CB tripped:', d.get('circuit_breaker_tripped'))"

# Reset via Redis (paper trader must be restarted after)
# The circuit breaker auto-resets after the configured cooldown period
systemctl --user restart apex-paper-trader
```

### Redis down

```bash
docker restart infra-redis-1
# Or for tp-redis:
docker restart tp-redis
```

### Kafka issues

```bash
docker logs infra-kafka-1 --tail 20
docker restart infra-kafka-1
```

### GPU out of memory

```bash
# Check what's using GPU
nvidia-smi

# Kill stuck processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# Restart paper trader
systemctl --user restart apex-paper-trader
```

### Model producing bad predictions

```bash
# Validate current model
python validate_models.py

# Retrain if needed
python train_postgres.py --symbols AAPL GOOGL MSFT AMZN NVDA META TSLA \
    --start-date 2024-01-01 --target-type returns --max-epochs 50

# Restart paper trader to pick up new model
systemctl --user restart apex-paper-trader
```

## Key Paths

| Item | Path |
|------|------|
| Test results | `results/` |
| Model checkpoints | `models/*.pth` |
| Backtest data | `data/backtest_stocks.csv` |
| Environment config | `.env` |
| Paper trader logs | `journalctl --user -u apex-paper-trader` |
| Health check logs | `results/health_*.txt` |
| Docker compose | `docker-compose.yml` |

## Monitoring URLs

| Service | URL |
|---------|-----|
| Paper Trader Dashboard | http://localhost:8010/dashboard |
| Paper Trader Health | http://localhost:8010/health |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5001 |
| Schema Registry | http://localhost:8081 |
