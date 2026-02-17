"""
Example usage of billing infrastructure components.

This file demonstrates how to use the billing provider, entitlement enforcer,
and usage tracker in a FastAPI application.
"""

import asyncio
import os
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi import FastAPI, Depends, HTTPException

# Note: In production, import from your actual infrastructure
# from infrastructure.billing import (
#     MockBillingProvider,
#     StripeBillingProvider,
#     EntitlementEnforcer,
#     UsageTracker,
#     require_entitlement,
#     check_entitlement_or_raise,
# )

async def example_billing_provider_usage():
    """Example: Using billing providers."""
    from infrastructure.billing import MockBillingProvider
    
    print("\n=== Billing Provider Example ===\n")
    
    # Initialize mock provider (no external dependencies)
    provider = MockBillingProvider()
    
    # Create a customer
    tenant_id = uuid4()
    customer_id = await provider.create_customer(
        tenant_id=tenant_id,
        email="user@example.com",
        name="John Doe"
    )
    print(f"✓ Created customer: {customer_id}")
    
    # Create a subscription with trial
    subscription = await provider.create_subscription(
        customer_id=customer_id,
        price_id="price_pro_monthly",
        trial_days=14
    )
    print(f"✓ Created subscription: {subscription['id']}")
    print(f"  Status: {subscription['status']}")
    print(f"  Trial ends: {subscription['trial_end']}")
    
    # Update subscription
    updated = await provider.update_subscription(
        subscription_id=subscription["id"],
        new_price_id="price_team_monthly"
    )
    print(f"✓ Updated subscription to new price")
    
    # Create payment method
    payment_method = await provider.create_payment_method(
        customer_id=customer_id,
        payment_method_data={
            "type": "card",
            "last4": "4242",
            "brand": "visa"
        }
    )
    print(f"✓ Created payment method: {payment_method.id}")
    
    # Cancel subscription
    canceled = await provider.cancel_subscription(
        subscription_id=subscription["id"],
        immediate=False
    )
    print(f"✓ Canceled subscription (at period end)")


async def example_entitlement_enforcer_usage():
    """Example: Using entitlement enforcer."""
    print("\n=== Entitlement Enforcer Example ===\n")
    
    # Note: In production, you would inject these dependencies
    print("(Skipping - requires database setup)")
    
    # Example decorator usage:
    """
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
    """
    
    # Example manual check:
    """
    check = await enforcer.check_and_record(
        tenant_id=tenant_id,
        resource_type="executions",
        quantity=Decimal("1")
    )
    
    if not check.allowed:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "entitlement_exceeded",
                "message": check.reason,
                "limit": str(check.limit),
                "current_usage": str(check.current_usage),
            }
        )
    """


async def example_usage_tracker_usage():
    """Example: Using usage tracker."""
    print("\n=== Usage Tracker Example ===\n")
    
    # Note: In production, you would inject repository dependency
    print("(Skipping - requires database setup)")
    
    # Example usage:
    """
    from infrastructure.billing import UsageTracker
    
    # Initialize tracker
    tracker = UsageTracker(
        usage_repository=usage_repo,
        batch_size=100,
        flush_interval_seconds=30.0
    )
    
    # Context manager usage (recommended)
    async with tracker:
        # Track workflow execution
        await tracker.track_execution(
            tenant_id=tenant_id,
            workflow_id=workflow_id
        )
        
        # Track LLM tokens
        await tracker.track_tokens(
            tenant_id=tenant_id,
            tokens=1500,
            model="gpt-4",
            operation="completion"
        )
        
        # Track API call
        await tracker.track_api_call(
            tenant_id=tenant_id,
            endpoint="/api/v1/workflows/execute",
            method="POST"
        )
        
        # Track storage
        await tracker.track_storage(
            tenant_id=tenant_id,
            size_mb=Decimal("10.5"),
            storage_type="checkpoint"
        )
        
        # Manual flush (optional - happens automatically)
        flushed = await tracker.flush()
        print(f"Flushed {flushed} records")
    
    # Get statistics
    stats = tracker.get_stats()
    print(f"Tracked: {stats['tracked']}, Flushed: {stats['flushed']}")
    """


async def example_fastapi_integration():
    """Example: FastAPI application with billing integration."""
    print("\n=== FastAPI Integration Example ===\n")
    
    print("Example FastAPI routes with billing:")
    print("""
from fastapi import FastAPI, Depends
from infrastructure.billing import (
    get_entitlement_enforcer,
    EntitlementEnforcer,
    check_entitlement_or_raise,
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
    
    # Raises 402 if limit exceeded
    check_entitlement_or_raise(check)
    
    # Execute workflow
    result = await execute_workflow_logic(workflow_id)
    return result


@app.post("/agents/{agent_id}/chat")
async def chat_with_agent(
    agent_id: UUID,
    message: str,
    tenant_id: UUID,
    enforcer: EntitlementEnforcer = Depends(get_entitlement_enforcer),
):
    # Estimate tokens (simplified)
    estimated_tokens = len(message.split()) * 2
    
    # Check token entitlement
    check = await enforcer.check_only(
        tenant_id=tenant_id,
        resource_type="tokens",
        quantity=Decimal(str(estimated_tokens)),
    )
    
    check_entitlement_or_raise(check)
    
    # Process chat
    response = await process_chat(agent_id, message)
    
    # Record actual token usage
    await enforcer.record_usage_only(
        tenant_id=tenant_id,
        resource_type="tokens",
        quantity=Decimal(str(response.tokens_used)),
    )
    
    return response
""")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("BILLING INFRASTRUCTURE EXAMPLES")
    print("="*60)
    
    await example_billing_provider_usage()
    await example_entitlement_enforcer_usage()
    await example_usage_tracker_usage()
    await example_fastapi_integration()
    
    print("\n" + "="*60)
    print("✅ Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
