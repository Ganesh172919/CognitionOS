# Billing Infrastructure - Quick Reference

## Quick Start

```python
from infrastructure.billing import (
    MockBillingProvider,
    EntitlementEnforcer,
    UsageTracker,
    require_entitlement,
)
```

## Billing Provider

### Create Customer
```python
provider = MockBillingProvider()
customer_id = await provider.create_customer(
    tenant_id=tenant.id,
    email="user@example.com",
    name="John Doe"
)
```

### Create Subscription
```python
subscription = await provider.create_subscription(
    customer_id=customer_id,
    price_id="price_pro_monthly",
    trial_days=14
)
```

### Update Subscription
```python
updated = await provider.update_subscription(
    subscription_id=sub_id,
    new_price_id="price_team_monthly"
)
```

### Cancel Subscription
```python
# Cancel at period end
await provider.cancel_subscription(sub_id, immediate=False)

# Cancel immediately
await provider.cancel_subscription(sub_id, immediate=True)
```

## Entitlement Enforcer

### Check and Record
```python
check = await enforcer.check_and_record(
    tenant_id=tenant_id,
    resource_type="executions",
    quantity=Decimal("1")
)

if not check.allowed:
    raise Exception(check.reason)
```

### Decorator Usage
```python
@require_entitlement("executions", quantity=Decimal("1"))
async def execute_workflow(tenant_id: UUID, enforcer=None):
    # Your logic here
    pass
```

### FastAPI Integration
```python
from fastapi import Depends

@app.post("/execute")
async def execute(
    enforcer: EntitlementEnforcer = Depends(get_entitlement_enforcer)
):
    check = await enforcer.check_and_record(tenant_id, "executions")
    check_entitlement_or_raise(check)
    # Your logic
```

## Usage Tracker

### Basic Tracking
```python
tracker = UsageTracker(usage_repo, batch_size=100, flush_interval_seconds=30)
tracker.start()

await tracker.track(tenant_id, "executions", Decimal("1"))

await tracker.stop()
```

### Context Manager
```python
async with UsageTracker(usage_repo) as tracker:
    await tracker.track_execution(tenant_id, workflow_id=wf_id)
    await tracker.track_tokens(tenant_id, tokens=1500, model="gpt-4")
    await tracker.track_api_call(tenant_id, endpoint="/api/v1/workflows")
    await tracker.track_storage(tenant_id, size_mb=Decimal("10.5"))
```

### Manual Flush
```python
flushed_count = await tracker.flush()
```

### Get Statistics
```python
stats = tracker.get_stats()
# Returns: tracked, flushed, errors, last_flush, buffer_size, etc.
```

## Resource Types

Common resource types for entitlement checks and usage tracking:

- `"executions"` - Workflow/agent executions
- `"tokens"` - LLM token usage
- `"api_calls"` - API request count
- `"storage_mb"` - Storage in megabytes
- `"agents"` - Number of agents
- `"workflows"` - Number of workflows

## Configuration

### Environment Variables
```bash
# Production Stripe
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Usage Tracker
USAGE_TRACKER_BATCH_SIZE=100
USAGE_TRACKER_FLUSH_INTERVAL=30
```

### Provider Selection
```python
import os

if os.getenv("ENVIRONMENT") == "production":
    provider = StripeBillingProvider(
        api_key=os.getenv("STRIPE_SECRET_KEY")
    )
else:
    provider = MockBillingProvider()
```

## Error Handling

```python
from infrastructure.billing import (
    BillingProviderError,
    EntitlementEnforcementError,
)

try:
    await provider.create_subscription(...)
except BillingProviderError as e:
    logger.error(f"Billing error: {e}")

try:
    await enforcer.check_and_record(...)
except EntitlementEnforcementError as e:
    logger.warning(f"Entitlement exceeded: {e}")
```

## Subscription Tiers

From `core/domain/billing/services.py`:

### FREE
- executions: 100
- tokens: 100,000
- storage_mb: 1,000
- api_calls: 1,000
- agents: 2
- workflows: 5

### PRO ($49/month)
- executions: 10,000
- tokens: 10,000,000
- storage_mb: 10,000
- api_calls: 100,000
- agents: 10
- workflows: 100

### TEAM ($199/month)
- executions: 100,000
- tokens: 100,000,000
- storage_mb: 100,000
- api_calls: 1,000,000
- agents: 50
- workflows: 500

### ENTERPRISE
- All unlimited

## Testing

### Unit Test with Mock Provider
```python
@pytest.mark.asyncio
async def test_create_customer():
    provider = MockBillingProvider()
    customer_id = await provider.create_customer(
        tenant_id=UUID("..."),
        email="test@example.com"
    )
    assert customer_id.startswith("cus_")
```

### Integration Test
```python
@pytest.mark.asyncio
async def test_entitlement_check(enforcer, test_tenant):
    check = await enforcer.check_and_record(
        tenant_id=test_tenant.id,
        resource_type="executions",
        quantity=Decimal("1")
    )
    assert check.allowed
```

## Common Patterns

### Check Before Action
```python
# Check entitlement
check = await enforcer.check_only(tenant_id, "executions")
if not check.allowed:
    return {"error": check.reason}

# Perform action
result = await execute_workflow(workflow_id)

# Record usage
await enforcer.record_usage_only(tenant_id, "executions", Decimal("1"))
```

### Estimate and Adjust
```python
# Estimate usage
estimated_tokens = len(prompt.split()) * 2
check = await enforcer.check_only(tenant_id, "tokens", Decimal(str(estimated_tokens)))
check_entitlement_or_raise(check)

# Execute
response = await llm.generate(prompt)

# Record actual usage
await tracker.track_tokens(tenant_id, tokens=response.tokens_used)
```

## Troubleshooting

### Imports fail
```bash
pip install fastapi sqlalchemy pydantic
```

### Stripe errors
- Verify API key: Check `STRIPE_SECRET_KEY`
- Check permissions: Ensure key has subscription management access

### Usage not tracked
- Ensure tracker is started: `tracker.start()`
- Check logs for errors
- Verify repository is configured

### Buffer not flushing
- Check `flush_interval_seconds` setting
- Manually flush: `await tracker.flush()`
- Check for errors in logs

## Performance Tips

1. **Batching**: Increase `batch_size` for high-volume systems
2. **Flush Interval**: Decrease for real-time requirements, increase for performance
3. **Caching**: Cache entitlement checks for frequently accessed tenants
4. **Async**: Always use async/await patterns

## Security Best Practices

1. Never commit API keys to version control
2. Always validate webhook signatures
3. Use environment variables for configuration
4. Implement rate limiting on billing endpoints
5. Enable audit logging for billing operations

## More Information

See `infrastructure/billing/README.md` for comprehensive documentation.
