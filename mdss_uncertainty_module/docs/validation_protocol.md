# Validation Protocol

## Unit Tests

```bash
pytest tests/ -v
```

Coverage target: > 80%

## Integration Tests

```bash
pytest tests/test_api.py -v
```

## Performance

- Single image: < 2 seconds
- Batch: 1000 images/hour

## Compliance Checklist

- [ ] Risk levels correctly assigned
- [ ] Human review triggers work
- [ ] PDF reports generated
- [ ] API responds correctly
