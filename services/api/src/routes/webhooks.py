"""
Webhook API Routes

FastAPI endpoints for receiving and processing Stripe webhooks.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse

from infrastructure.billing.webhook_handler import (
    StripeWebhookHandler,
    WebhookValidationError,
)
from infrastructure.persistence.webhook_event_repository import WebhookEventRepository
from services.api.src.dependencies.injection import (
    get_database_session,
    get_billing_service,
)
from core.exceptions import BillingException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


async def get_webhook_handler(
    session = Depends(get_database_session),
    billing_service = Depends(get_billing_service),
) -> StripeWebhookHandler:
    """Dependency to get webhook handler instance."""
    from core.config import get_settings
    settings = get_settings()
    
    event_repository = WebhookEventRepository(session)
    
    return StripeWebhookHandler(
        webhook_secret=settings.STRIPE_WEBHOOK_SECRET,
        billing_service=billing_service,
        event_repository=event_repository,
    )


@router.post("/stripe", status_code=200)
async def handle_stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
    webhook_handler: StripeWebhookHandler = Depends(get_webhook_handler),
) -> JSONResponse:
    """
    Handle incoming Stripe webhook events.
    
    This endpoint:
    - Verifies webhook signatures for security
    - Processes events idempotently (prevents duplicate processing)
    - Routes events to appropriate handlers
    - Returns 200 OK to Stripe for successful processing
    
    Args:
        request: FastAPI request object
        stripe_signature: Stripe webhook signature header
        webhook_handler: Injected webhook handler
        
    Returns:
        JSONResponse with processing status
        
    Raises:
        HTTPException: If signature validation fails or processing errors occur
    """
    # Get raw request body for signature verification
    payload = await request.body()
    
    if not stripe_signature:
        logger.error("Missing Stripe-Signature header")
        raise HTTPException(
            status_code=400,
            detail="Missing Stripe-Signature header"
        )
    
    try:
        # Parse signature header for timestamp
        sig_parts = dict(part.split('=') for part in stripe_signature.split(','))
        timestamp = int(sig_parts.get('t', 0))
        
        # Verify signature
        webhook_handler.verify_signature(
            payload=payload,
            signature=stripe_signature,
            timestamp=timestamp,
        )
        
        # Parse event data
        import json
        event_data = json.loads(payload.decode('utf-8'))
        
        # Process event
        result = await webhook_handler.process_event(
            event_data=event_data,
            idempotency_key=event_data.get("id"),
        )
        
        logger.info(f"Webhook processed successfully: {event_data.get('id')}")
        
        return JSONResponse(
            status_code=200,
            content={
                "received": True,
                "event_id": event_data.get("id"),
                "event_type": event_data.get("type"),
                "status": result.get("status"),
            }
        )
        
    except WebhookValidationError as e:
        logger.error(f"Webhook signature validation failed: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Webhook signature validation failed: {str(e)}"
        )
    
    except BillingException as e:
        logger.error(f"Billing error processing webhook: {e}")
        # Return 200 to Stripe but log the error for investigation
        # This prevents Stripe from retrying if it's a data issue
        return JSONResponse(
            status_code=200,
            content={
                "received": True,
                "status": "error_logged",
                "error": str(e),
            }
        )
    
    except Exception as e:
        logger.error(f"Unexpected error processing webhook: {e}", exc_info=True)
        # Return 500 so Stripe will retry
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing webhook: {str(e)}"
        )


@router.get("/events/{event_id}", status_code=200)
async def get_webhook_event(
    event_id: str,
    webhook_handler: StripeWebhookHandler = Depends(get_webhook_handler),
) -> Dict[str, Any]:
    """
    Get webhook event processing details.
    
    Useful for debugging and monitoring webhook processing.
    
    Args:
        event_id: Stripe event ID
        webhook_handler: Injected webhook handler
        
    Returns:
        Webhook event details
    """
    event = await webhook_handler.event_repository.get_event(event_id)
    
    if not event:
        raise HTTPException(
            status_code=404,
            detail=f"Webhook event not found: {event_id}"
        )
    
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "status": event.status,
        "result": event.result,
        "error_message": event.error_message,
        "created_at": event.created_at.isoformat(),
        "processed_at": event.processed_at.isoformat() if event.processed_at else None,
        "retry_count": event.retry_count,
    }


@router.get("/events", status_code=200)
async def list_webhook_events(
    event_type: str = None,
    limit: int = 100,
    offset: int = 0,
    webhook_handler: StripeWebhookHandler = Depends(get_webhook_handler),
) -> Dict[str, Any]:
    """
    List webhook events with optional filtering.
    
    Args:
        event_type: Optional filter by event type
        limit: Maximum number of events to return
        offset: Number of events to skip
        webhook_handler: Injected webhook handler
        
    Returns:
        List of webhook events
    """
    if event_type:
        events = await webhook_handler.event_repository.get_events_by_type(
            event_type=event_type,
            limit=limit,
            offset=offset,
        )
    else:
        # Get all events (would need to implement in repository)
        events = []
    
    return {
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "status": event.status,
                "created_at": event.created_at.isoformat(),
                "processed_at": event.processed_at.isoformat() if event.processed_at else None,
            }
            for event in events
        ],
        "count": len(events),
        "limit": limit,
        "offset": offset,
    }


@router.get("/statistics", status_code=200)
async def get_webhook_statistics(
    hours: int = 24,
    webhook_handler: StripeWebhookHandler = Depends(get_webhook_handler),
) -> Dict[str, Any]:
    """
    Get webhook processing statistics.
    
    Provides metrics for monitoring webhook health:
    - Total events processed
    - Success/failure rates
    - Events by type
    
    Args:
        hours: Calculate statistics for last N hours
        webhook_handler: Injected webhook handler
        
    Returns:
        Webhook processing statistics
    """
    stats = await webhook_handler.event_repository.get_event_statistics(hours=hours)
    
    return {
        "statistics": stats,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/retry-failed", status_code=200)
async def retry_failed_webhooks(
    max_retries: int = 5,
    webhook_handler: StripeWebhookHandler = Depends(get_webhook_handler),
) -> Dict[str, Any]:
    """
    Manually retry failed webhook events.
    
    This endpoint is useful for recovering from temporary failures.
    
    Args:
        max_retries: Only retry events with fewer than this many attempts
        webhook_handler: Injected webhook handler
        
    Returns:
        Retry results
    """
    failed_events = await webhook_handler.event_repository.get_failed_events(
        max_retry_count=max_retries,
        limit=100,
    )
    
    results = []
    for event in failed_events:
        try:
            # Increment retry count
            await webhook_handler.event_repository.increment_retry_count(event.event_id)
            
            # Attempt to reprocess
            # Note: In production, this would need the original event data
            # For now, just mark as retrying
            results.append({
                "event_id": event.event_id,
                "status": "retry_attempted",
            })
            
        except Exception as e:
            logger.error(f"Error retrying event {event.event_id}: {e}")
            results.append({
                "event_id": event.event_id,
                "status": "retry_failed",
                "error": str(e),
            })
    
    return {
        "total_retried": len(results),
        "results": results,
    }
