# NIODOO Production Readiness Guide

## Overview
This guide covers operational procedures for running NIODOO in production environments.

## Security Configuration

### Rate Limiting
- Default: 45 requests per 60 seconds
- Configure via `SECURITY_PROMPT_RATE_LIMIT` and `SECURITY_PROMPT_RATE_WINDOW_SECS`
- Rate limit violations are logged to `logs/security_audit.log`

### Content Filtering
- Default patterns block SQL injection, XSS, and command injection attempts
- Configure via `SECURITY_BANNED_PATTERNS` (comma-separated regex patterns)
- Filtered prompts are logged with reason and hash

### Audit Logging
- **Security Audit**: `logs/security_audit.log`
  - All prompt accept/reject events
  - Rate limit violations
  - Content filter matches
  - Configuration snapshots
- **Config Audit**: `logs/config_audit.log`
  - All configuration overrides
  - Value hashes for tamper detection

### Prompt Sanitization
- Control characters stripped by default (configurable via `SECURITY_ALLOW_CONTROL_CHARS`)
- Carriage returns normalized to newlines
- Maximum prompt length enforced (default: 512 chars, configurable via `PROMPT_MAX_CHARS`)

## Configuration Validation

The system validates configuration on startup. Invalid configurations will fail fast with clear error messages.

### Validated Parameters
- Numeric ranges: `prompt_max_chars` ≤ 1M, `generation_max_tokens` ≤ 100K, `timeout` ≤ 3600s
- Parameter bounds: `temperature` (0.0-2.0), `top_p` (0.0-1.0)
- URL formats: All endpoints must be HTTP/HTTPS
- Cache settings: All TTL values must be > 0
- Retry settings: Max retries ≤ 100, base delay > 0
- Thresholds: All similarity/quality thresholds between 0.0-1.0

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t niodoo-real-integrated:latest -f niodoo_real_integrated/Dockerfile .

# Run container
docker run -d \
  --name niodoo \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  -e VLLM_ENDPOINT=http://vllm:5001 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e OLLAMA_URL=http://ollama:11434 \
  niodoo-real-integrated:latest
```

### Health Checks
The Docker image includes a health check that verifies the process is running:
- Interval: 30s
- Timeout: 10s
- Start period: 40s
- Retries: 3

### Non-Root User
The container runs as user `niodoo` (UID 1000) for security.

## Monitoring

### Metrics
Prometheus metrics available at `/metrics` endpoint (when `svc` feature enabled):
- `niodoo_cycles_total`: Total pipeline cycles processed
- `niodoo_entropy`: Current entropy value
- `niodoo_latency_ms`: Pipeline latency in milliseconds
- `niodoo_rouge_score`: ROUGE-L score vs baseline
- `niodoo_threat_cycles`: Number of threat-detected cycles
- `niodoo_healing_cycles`: Number of healing-detected cycles

### Logs
- Application logs: Structured logging via `tracing` (configure via `RUST_LOG`)
- Security audit: `logs/security_audit.log` (all security events)
- Config audit: `logs/config_audit.log` (all config changes)

## Troubleshooting

### Common Issues

#### Rate Limit Violations
- **Symptom**: Prompts rejected with rate limit error
- **Solution**: Increase `SECURITY_PROMPT_RATE_LIMIT` or `SECURITY_PROMPT_RATE_WINDOW_SECS`
- **Check**: Review `logs/security_audit.log` for violation patterns

#### Configuration Validation Failures
- **Symptom**: Startup fails with validation error
- **Solution**: Check error message for specific parameter issue
- **Check**: Verify all environment variables are within valid ranges

#### Missing Audit Logs
- **Symptom**: Security/config audit logs not appearing
- **Solution**: Ensure `logs/` directory exists and is writable
- **Check**: Container must have write permissions for `/app/logs`

### Performance Tuning

#### Cache Configuration
- `EMBEDDING_CACHE_TTL_SECS`: Embedding cache TTL (default: 3600)
- `COLLAPSE_CACHE_TTL_SECS`: ERAG collapse cache TTL (default: 1800)
- `CACHE_CAPACITY`: LRU cache capacity (default: 256)

#### Generation Parameters
- `GENERATION_MAX_TOKENS`: Maximum tokens per generation (default: 2048)
- `GENERATION_TIMEOUT_SECS`: Generation timeout (default: 60)
- `TEMPERATURE`: Sampling temperature (default: 0.7, range: 0.0-2.0)
- `TOP_P`: Nucleus sampling parameter (default: 0.9, range: 0.0-1.0)

## Security Best Practices

1. **Audit Log Rotation**: Regularly rotate audit logs to prevent disk exhaustion
2. **Access Control**: Restrict access to audit logs (contain sensitive hashes)
3. **Rate Limiting**: Adjust rate limits based on actual traffic patterns
4. **Content Filtering**: Update banned patterns as new attack vectors emerge
5. **Configuration Security**: Never log sensitive values (API keys, passwords) - only hashes are logged

## Compliance

### Audit Trail
All security events are logged with:
- RFC3339 timestamps
- Blake3 hashes for tamper detection
- Event type and reason codes
- Character counts (not full content)

### Configuration Tracking
All configuration changes are logged with:
- Timestamp
- Configuration key
- Value hash (Blake3)
- Character count

This enables:
- Security forensics
- Compliance audits
- Configuration change tracking
- Tamper detection

## Support

For issues or questions:
1. Check `logs/security_audit.log` for security events
2. Check application logs for runtime errors
3. Review configuration validation errors on startup
4. Consult `CHANGELOG.md` for recent changes

