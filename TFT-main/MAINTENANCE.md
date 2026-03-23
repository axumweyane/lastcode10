# APEX Maintenance Schedule

All commands run from `TFT-main/`.

## Daily (Pre-Market, 8:30 AM ET)

Automated via systemd timer (`apex-health-check.timer`). Manual:

```bash
bash run_all_checks.sh
```

Checks: data audit, model validation, paper trader e2e, backtest validation, infrastructure health.

**Manual review checklist:**
- [ ] Check `results/` for any FAIL results
- [ ] Review paper trader dashboard: `http://localhost:8010/dashboard`
- [ ] Check Grafana dashboards: `http://localhost:3000`
- [ ] Verify circuit breaker is not tripped:
  ```bash
  curl -s http://localhost:8010/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('CB:', d.get('circuit_breaker_tripped'), '| Status:', d.get('status'))"
  ```
- [ ] Review yesterday's P&L in daily snapshots:
  ```bash
  curl -s http://localhost:8010/history | python3 -c "import sys,json; h=json.load(sys.stdin); [print(f\"{e.get('date','?')}: \${e.get('portfolio_value',0):,.0f}\") for e in h[-3:]]" 2>/dev/null || echo "No history"
  ```
- [ ] Check for ERROR in recent logs:
  ```bash
  journalctl --user -u apex-paper-trader --since '24 hours ago' -p err --no-pager | tail -5
  ```
- [ ] If Monday, check GTC orders from last week executed:
  ```bash
  curl -s http://localhost:8010/positions | python3 -m json.tool | head -20
  ```

## Weekly (Sunday Evening)

### 1. Run full health check suite

```bash
bash run_all_checks.sh
```

### 2. Retrain TFT-Stocks model

```bash
python train_postgres.py --symbols AAPL GOOGL MSFT AMZN NVDA META TSLA \
    --start-date 2024-01-01 --target-type returns --max-epochs 50
```

### 3. Run strategy optimizer

```bash
python optimize_strategies.py --folds 5 --max-samples 100 --output optimization_results.json
```

### 4. Populate supplementary data

```bash
python scripts/populate_tables.py --tables vix fundamentals sentiment
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

### 7. Check strategy weights — flag any stuck at 0

```bash
curl -s http://localhost:8010/weights/bayesian | python3 -c "import sys,json; w=json.load(sys.stdin); [print(f'  {k}: {v:.4f}') for k,v in sorted(w.items())]" 2>/dev/null || echo "No weights"
```

### 8. Review optimization results

```bash
cat optimization_results.json | python3 -c "
import sys,json; d=json.load(sys.stdin)
for s,r in d.get('results',{}).items():
    if r: print(f'{s}: best Sharpe={r[0].get(\"avg_oos_sharpe\",0):.3f}')
"
```

### 9. Check Alpaca account equity trend

```bash
curl -s http://localhost:8010/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Portfolio: \${d.get('portfolio_value',0):,.2f}\")"
```

### 10. Pull latest code and check CI

```bash
git pull --rebase
git log --oneline -5
```

## Biweekly (1st and 15th)

### 1. Retrain all 3 TFT models

```bash
# Stocks
python train_postgres.py --symbols AAPL GOOGL MSFT AMZN NVDA META TSLA \
    --start-date 2024-01-01 --target-type returns --max-epochs 50

# Validate all models after retrain
python validate_models.py
```

### 2. Compare new model val_loss vs previous

```bash
python3 -c "
import json, glob
files = sorted(glob.glob('results/models_*.json'))[-4:]
for f in files:
    d = json.load(open(f))
    grades = d.get('grades', {})
    print(f'{f}: {grades}')
"
```

### 3. Database maintenance

```bash
# VACUUM and ANALYZE
psql -h localhost -p 15432 -U apex_user -d apex -c "VACUUM ANALYZE;"

# Check table sizes
psql -h localhost -p 15432 -U apex_user -d apex -c "
    SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
    FROM pg_catalog.pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC LIMIT 10;
"
```

### 4. Log rotation

```bash
# Rotate paper trader logs
journalctl --user -u apex-paper-trader --vacuum-time=14d

# Clean old results
find results/ -name "*.json" -mtime +30 -delete
find results/ -name "*.txt" -mtime +30 -delete
```

### 5. Docker cleanup

```bash
docker system prune -f
docker volume prune -f
```

### 6. Review TimescaleDB retention policies

```bash
psql -h localhost -p 15432 -U apex_user -d apex -c "
    SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_retention';
"
```

### 7. Update requirements if security advisories

```bash
pip list --outdated | head -20
pip audit 2>/dev/null || echo "pip-audit not installed"
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
grep -rn "password\|secret\|api_key" .env* --include="*.env*" 2>/dev/null

# Update Python dependencies
pip list --outdated
pip install --upgrade -r requirements.txt
```

### 3. Backup database

```bash
mkdir -p backups
pg_dump -h localhost -p 15432 -U apex_user apex > backups/apex_$(date +%Y%m%d).sql
```

### 4. Review P&L trend

```bash
curl -s http://localhost:8010/history | python3 -c "
import sys,json
h=json.load(sys.stdin)
if h:
    vals = [e.get('portfolio_value',0) for e in h]
    print(f'Start: \${vals[0]:,.0f}  End: \${vals[-1]:,.0f}  Change: \${vals[-1]-vals[0]:,.0f}')
    print(f'Days: {len(h)}')
else:
    print('No history data')
"
```

### 5. Review strategy kill history

```bash
psql -h localhost -p 15432 -U apex_user -d apex -c "
    SELECT * FROM paper_risk_reports ORDER BY created_at DESC LIMIT 5;
" 2>/dev/null || echo "No risk reports table"
```

### 6. Review model performance trends

```bash
python3 -c "
import json, glob
files = sorted(glob.glob('results/models_*.json'))[-4:]
for f in files:
    d = json.load(open(f))
    print(f'{f}: {d.get(\"grades\", d)}')
"
```

### 7. Verify systemd timers

```bash
systemctl --user list-timers
systemctl --user status apex-paper-trader
systemctl --user status apex-health-check.timer
```

### 8. Check GitHub Actions CI

```bash
gh run list --limit 5 2>/dev/null || echo "gh CLI not available"
```

## Emergency Procedures

### Paper trader crashed

```bash
# 1. Check status
systemctl --user status apex-paper-trader

# 2. Check logs for root cause
journalctl --user -u apex-paper-trader -n 100 --no-pager | tail -30

# 3. Restart
systemctl --user restart apex-paper-trader

# 4. Verify recovery
sleep 5 && curl -s http://localhost:8010/health | python3 -m json.tool
```

### Circuit breaker tripped

```bash
# 1. Check status and reason
curl -s http://localhost:8010/health | python3 -c "
import sys,json; d=json.load(sys.stdin)
print('CB tripped:', d.get('circuit_breaker_tripped'))
print('Portfolio:', d.get('portfolio_value'))
"

# 2. Check risk reports for details
psql -h localhost -p 15432 -U apex_user -d apex -c "
    SELECT * FROM paper_risk_reports ORDER BY created_at DESC LIMIT 3;
" 2>/dev/null

# 3. The circuit breaker auto-resets after cooldown period
# To force reset, restart the paper trader:
systemctl --user restart apex-paper-trader

# 4. Verify
sleep 5 && curl -s http://localhost:8010/health | python3 -c "import sys,json; print('CB:', json.load(sys.stdin).get('circuit_breaker_tripped'))"
```

### Database connection lost

```bash
# 1. Check container
docker ps | grep timescaledb

# 2. Check logs
docker logs apex-timescaledb --tail 20

# 3. Restart if needed
docker restart apex-timescaledb

# 4. Wait for recovery and check
sleep 10 && psql -h localhost -p 15432 -U apex_user -d apex -c "SELECT 1;"

# 5. Check active connections
psql -h localhost -p 15432 -U apex_user -d apex -c "SELECT count(*) FROM pg_stat_activity;"
```

### Model producing NaN predictions

```bash
# 1. Validate current models
python validate_models.py

# 2. Check which model is broken (look for grade F in output)

# 3. Retrain the broken model
python train_postgres.py --symbols AAPL GOOGL MSFT AMZN NVDA META TSLA \
    --start-date 2024-01-01 --target-type returns --max-epochs 50

# 4. Validate again
python validate_models.py

# 5. Restart paper trader to pick up new model
systemctl --user restart apex-paper-trader
```

### Alpaca API 403 errors

```bash
# 1. Check if API keys are valid
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
    https://paper-api.alpaca.markets/v2/account | python3 -m json.tool

# 2. Cancel all pending orders
curl -s -X DELETE -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
    https://paper-api.alpaca.markets/v2/orders

# 3. Verify positions
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
    https://paper-api.alpaca.markets/v2/positions | python3 -m json.tool
```

### Redis down

```bash
# 1. Check container status
docker ps | grep redis

# 2. Restart
docker restart infra-redis-1
docker restart tp-redis

# 3. Verify circuit breaker state
sleep 3 && curl -s http://localhost:8010/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('Redis:', d.get('redis',{}).get('connected'))"
```

### Disk full

```bash
# 1. Check disk usage
df -h /

# 2. Find large files
du -sh models/*.pth lightning_logs/ results/ data/ 2>/dev/null

# 3. Clean up (safe operations first)
find results/ -name "*.json" -mtime +14 -delete
find results/ -name "*.txt" -mtime +14 -delete
rm -rf lightning_logs/version_*/
docker system prune -f

# 4. If still full, compress old backups
gzip backups/*.sql 2>/dev/null
```

### GPU out of memory

```bash
# 1. Check what's using GPU
nvidia-smi

# 2. Kill stuck Python processes using GPU
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# 3. Restart paper trader
systemctl --user restart apex-paper-trader

# 4. Verify GPU is available
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Kafka issues

```bash
# 1. Check logs
docker logs infra-kafka-1 --tail 30

# 2. Restart
docker restart infra-kafka-1

# 3. Check DLQ for stuck messages
curl -s http://localhost:8010/dlq | python3 -m json.tool
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
| Strategy configs | `strategies/config.py` |
| Optimization results | `optimization_results.json` |

## Monitoring URLs

| Service | URL |
|---------|-----|
| Paper Trader Dashboard | http://localhost:8010/dashboard |
| Paper Trader Health | http://localhost:8010/health |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5001 |
| Schema Registry | http://localhost:8081 |
| APEX Dashboard | http://localhost:3001 |
