# Deployment Guide

## Local Deployment

### Development

```bash
pip install -e ".[dev]"
crypto-portfolio --output-dir outputs --run-dir runs
```

### Production

```bash
pip install .
crypto-portfolio \
  --output-dir /var/data/outputs \
  --run-dir /var/data/runs \
  --forecast-method arima \
  --seed 42
```

## Docker (Planned)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install .
ENTRYPOINT ["crypto-portfolio"]
```

## Environment Considerations

- Ensure sufficient memory for large correlation matrices
- ARIMA forecasting is CPU-intensive; consider hardware resources
- Output directories need write permissions
- Run directories should persist across executions for idempotency

## Monitoring

The system outputs structured events and metrics:

- **events.jsonl** - Real-time pipeline events for monitoring
- **metrics.json** - Counters and timing for performance tracking

## Scheduling

For periodic rebalancing, use cron or a workflow scheduler:

```bash
# Example: run weekly
0 0 * * 0 cd /path/to/project && crypto-portfolio --output-dir outputs --run-dir runs
```
