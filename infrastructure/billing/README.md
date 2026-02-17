# Billing Infrastructure

This directory contains the billing infrastructure for CognitionOS, providing payment processing, entitlement enforcement, and usage tracking capabilities.

## Overview

The billing infrastructure consists of three main components:

1. **Provider** (`provider.py`) - Billing provider abstraction with Stripe implementation
2. **Entitlement Enforcer** (`entitlement_enforcer.py`) - Runtime entitlement enforcement
3. **Usage Tracker** (`usage_tracker.py`) - Asynchronous usage tracking with batching

## Components

### 1. Billing Provider (`provider.py`)

Abstract billing provider interface with implementations for external billing systems.

#### Features
- Abstract `BillingProvider` base class
- `StripeBillingProvider` - Production Stripe integration
- `MockBillingProvider` - Local development without external API calls

#### Usage

```python
from infrastructure.billing import StripeBillingProvider, MockBillingProvider

# Production with Stripe
stripe_provider = StripeBillingProvider(
    api_key="sk_live_...",
    webhook_secret="whsec_..."
)

# Development with mock
mock_provider = MockBillingProvider()

# Create customer
customer_id = await provider.create_customer(
    tenant_id=tenant.id,
    email="user@example.com",
    name="John Doe"
)

# Create subscription
subscription = await provider.create_subscription(
    customer_id=customer_id,
    price_id="price_123",
    trial_days=14
)

# Update subscription
updated = await provider.update_subscription(
    subscription_id=subscription["id"],
    new_price_id="price_456"
)

# Cancel subscription
canceled = await provider.cancel_subscription(
    subscription_id=subscription["id"],
    immediate=False  # Cancel at period end
)
```

#### Stripe Integration

The `StripeBillingProvider` requires the `stripe` package:

```bash
pip install stripe
```

Configure with your Stripe API keys:

```python
import os
from infrastructure.billing import StripeBillingProvider

provider = StripeBillingProvider(
    api_key=os.getenv("STRIPE_SECRET_KEY"),
    webhook_secret=os.getenv("STRIPE_WEBHOOK_SECRET")
)
```

### 2. Entitlement Enforcer (`entitlement_enforcer.py`)

Runtime entitlement enforcement with automatic usage recording.

#### Features
- Wraps `EntitlementService` from domain layer
- Decorator-based enforcement: `@require_entitlement()`
- FastAPI dependency injection support
- Automatic usage recording on successful checks

#### Usage

##### Basic Entitlement Check

```python
from decimal import Decimal
from infrastructure.billing import EntitlementEnforcer

# Check and record usage
check = await enforcer.check_and_record(
    tenant_id=tenant_id,
    resource_type="executions",
    quantity=Decimal("1")
)

if check.allowed:
    # Execute workflow
    pass
```

##### Decorator Usage

```python
from decimal import Decimal
from infrastructure.billing import require_entitlement

@require_entitlement("executions", quantity=Decimal("1"))
async def execute_workflow(
    tenant_id: UUID,
    workflow_id: UUID,
    enforcer: EntitlementEnforcer = Depends()
):
    # This only runs if tenant has available executions
    # Usage is automatically recorded
    return await run_workflow(workflow_id)

@require_entitlement("tokens", quantity=Decimal("1000"))
async def process_with_llm(
    tenant_id: UUID,
    prompt: str,
    enforcer: EntitlementEnforcer = Depends()
):
    # Only runs if tenant has 1000+ tokens available
    # Usage is automatically recorded
    return await llm.generate(prompt)
```

##### FastAPI Integration

```python
from fastapi import FastAPI, Depends, HTTPException
from infrastructure.billing import (
    get_entitlement_enforcer,
    EntitlementEnforcer,
    check_entitlement_or_raise
)

app = FastAPI()

@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: UUID,
    tenant_id: UUID,
    enforcer: EntitlementEnforcer = Depends(get_entitlement_enforcer),
):
    # Check entitlement
    check = await enforcer.check_and_record(
        tenant_id=tenant_id,
        resource_type="executions",
        quantity=Decimal("1"),
    )
    
    # Raises HTTPException if not allowed
    check_entitlement_or_raise(check)
    
    # Execute workflow
    result = await run_workflow(workflow_id)
    return result
```

### 3. Usage Tracker (`usage_tracker.py`)

High-performance usage tracking with background aggregation and batching.

#### Features
- Asynchronous batching for performance
- Background flush task
- Automatic aggregation
- Specialized tracking methods for common resource types

#### Usage

##### Basic Usage Tracking

```python
from infrastructure.billing import UsageTracker
from decimal import Decimal

# Initialize tracker
tracker = UsageTracker(
    usage_repository=usage_repo,
    batch_size=100,
    flush_interval_seconds=30.0
)

# Start background flush task
tracker.start()

# Track usage
await tracker.track(
    tenant_id=tenant_id,
    resource_type="executions",
    quantity=Decimal("1")
)

# Stop and flush remaining records
await tracker.stop()
```

##### Context Manager Usage

```python
async with UsageTracker(usage_repo) as tracker:
    await tracker.track_execution(
        tenant_id=tenant_id,
        workflow_id=workflow_id
    )
    
    await tracker.track_tokens(
        tenant_id=tenant_id,
        tokens=1500,
        model="gpt-4"
    )
    
    await tracker.track_api_call(
        tenant_id=tenant_id,
        endpoint="/api/v1/workflows",
        method="POST"
    )
    
    await tracker.track_storage(
        tenant_id=tenant_id,
        size_mb=Decimal("10.5"),
        storage_type="checkpoint"
    )
# Automatically stops and flushes on exit
```

##### Manual Flush

```python
# Flush immediately
flushed_count = await tracker.flush()
print(f"Flushed {flushed_count} records")

# Get tracker statistics
stats = tracker.get_stats()
print(f"Tracked: {stats['tracked']}, Flushed: {stats['flushed']}")
```

##### Usage Aggregation

```python
from infrastructure.billing import UsageAggregator
from datetime import datetime, timedelta

aggregator = UsageAggregator(usage_repo)

# Get usage trends
trends = await aggregator.get_usage_trends(
    tenant_id=tenant_id,
    resource_type="tokens",
    days=30
)

# Get top consumers
top_consumers = await aggregator.get_top_consumers(
    resource_type="executions",
    start_time=datetime.utcnow() - timedelta(days=7),
    end_time=datetime.utcnow(),
    limit=10
)
```

## Architecture

### Integration with Domain Layer

The billing infrastructure integrates with the domain layer services:

```
infrastructure/billing/
├── provider.py              # External billing system integration
├── entitlement_enforcer.py  # Wraps EntitlementService
└── usage_tracker.py         # Wraps UsageMeteringService

core/domain/billing/
├── entities.py              # Domain entities (Subscription, Invoice, etc.)
├── services.py              # Domain services (EntitlementService, etc.)
└── repositories.py          # Repository interfaces
```

### Async/Await Patterns

All components use async/await for non-blocking I/O:

```python
# Provider - async Stripe API calls
subscription = await provider.create_subscription(...)

# Enforcer - async entitlement checks
check = await enforcer.check_and_record(...)

# Tracker - async batched persistence
await tracker.track(...)
```

### Error Handling

Each component has specific error types:

```python
from infrastructure.billing import (
    BillingProviderError,
    EntitlementEnforcementError
)

try:
    await provider.create_subscription(...)
except BillingProviderError as e:
    logger.error(f"Billing provider error: {e}")

try:
    await enforcer.check_and_record(...)
except EntitlementEnforcementError as e:
    logger.warning(f"Entitlement check failed: {e}")
```

### Logging

Comprehensive structured logging throughout:

```python
logger.info(
    "Created subscription",
    extra={
        "tenant_id": str(tenant_id),
        "subscription_id": subscription_id,
        "tier": tier.value
    }
)
```

## Configuration

### Environment Variables

```bash
# Stripe Configuration (production)
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Usage Tracker Configuration
USAGE_TRACKER_BATCH_SIZE=100
USAGE_TRACKER_FLUSH_INTERVAL=30

# Entitlement Configuration
ENTITLEMENT_CHECK_ENABLED=true
```

### Application Setup

```python
from fastapi import FastAPI
from infrastructure.billing import (
    StripeBillingProvider,
    MockBillingProvider,
    UsageTracker
)
from infrastructure.persistence.billing_repository import (
    PostgreSQLUsageRecordRepository
)

app = FastAPI()

# Initialize based on environment
if os.getenv("ENVIRONMENT") == "production":
    billing_provider = StripeBillingProvider(
        api_key=os.getenv("STRIPE_SECRET_KEY")
    )
else:
    billing_provider = MockBillingProvider()

# Initialize usage tracker
usage_tracker = UsageTracker(
    usage_repository=PostgreSQLUsageRecordRepository(session),
    batch_size=int(os.getenv("USAGE_TRACKER_BATCH_SIZE", "100")),
    flush_interval_seconds=float(os.getenv("USAGE_TRACKER_FLUSH_INTERVAL", "30.0"))
)

@app.on_event("startup")
async def startup():
    usage_tracker.start()

@app.on_event("shutdown")
async def shutdown():
    await usage_tracker.stop()
```

## Testing

### Unit Tests

```python
import pytest
from infrastructure.billing import MockBillingProvider

@pytest.mark.asyncio
async def test_mock_provider_create_customer():
    provider = MockBillingProvider()
    
    customer_id = await provider.create_customer(
        tenant_id=UUID("..."),
        email="test@example.com"
    )
    
    assert customer_id.startswith("cus_")
    
    customer = await provider.get_customer(customer_id)
    assert customer["email"] == "test@example.com"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_entitlement_enforcer_check_and_record(
    enforcer: EntitlementEnforcer,
    test_tenant: Tenant
):
    check = await enforcer.check_and_record(
        tenant_id=test_tenant.id,
        resource_type="executions",
        quantity=Decimal("1")
    )
    
    assert check.allowed
    assert check.remaining > 0
```

## Performance Considerations

### Usage Tracker Batching

The usage tracker uses batching to minimize database writes:

- **Batch Size**: Controls memory usage (default: 100 records)
- **Flush Interval**: Controls latency (default: 30 seconds)
- **Buffer Lock**: Ensures thread-safe operations

### Entitlement Caching

Consider caching entitlement checks for frequently accessed tenants:

```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=1000)
def get_cached_entitlement(tenant_id: str, resource_type: str, timestamp: int):
    # Cache for 1 minute intervals
    pass
```

## Security Considerations

1. **API Keys**: Never commit Stripe keys to version control
2. **Webhook Validation**: Always validate Stripe webhooks
3. **Rate Limiting**: Implement rate limiting on billing endpoints
4. **Audit Logging**: Log all billing operations

## Migration Guide

### From Direct Stripe API

```python
# Before
import stripe
stripe.api_key = "sk_..."
customer = stripe.Customer.create(email="...")

# After
from infrastructure.billing import StripeBillingProvider
provider = StripeBillingProvider(api_key="sk_...")
customer_id = await provider.create_customer(
    tenant_id=tenant.id,
    email="..."
)
```

### From Manual Usage Tracking

```python
# Before
usage_record = UsageRecord.create(...)
await usage_repo.create(usage_record)

# After
from infrastructure.billing import UsageTracker
tracker = UsageTracker(usage_repo)
await tracker.track(tenant_id=tenant.id, resource_type="executions")
```

## Troubleshooting

### Common Issues

1. **Imports fail**: Ensure FastAPI and SQLAlchemy are installed
2. **Stripe errors**: Verify API key is correct and has proper permissions
3. **Usage not tracked**: Check that tracker background task is started
4. **Buffer not flushing**: Check flush interval and batch size settings

### Debug Logging

Enable debug logging to troubleshoot:

```python
import logging
logging.getLogger("infrastructure.billing").setLevel(logging.DEBUG)
```

## License

See main repository LICENSE file.
