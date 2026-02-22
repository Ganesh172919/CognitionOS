"""
Payment Orchestration Engine - Multi-Provider Payment Management

Implements enterprise-grade payment orchestration with:
- Multi-gateway support (Stripe, PayPal, Adyen, Braintree)
- Intelligent routing and failover
- Payment method optimization
- Fraud detection integration
- Subscription payment automation
- Dunning management
- Payment analytics
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from decimal import Decimal
import hashlib
import json


class PaymentGateway(Enum):
    """Supported payment gateways"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    ADYEN = "adyen"
    BRAINTREE = "braintree"
    SQUARE = "square"


class PaymentMethod(Enum):
    """Payment method types"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    ACH = "ach"
    WIRE_TRANSFER = "wire_transfer"
    CRYPTO = "crypto"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"


class PaymentStatus(Enum):
    """Payment processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class FraudLevel(Enum):
    """Fraud risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PaymentTransaction:
    """Payment transaction record"""
    transaction_id: str
    customer_id: str
    amount: Decimal
    currency: str
    payment_method: PaymentMethod
    gateway: PaymentGateway
    status: PaymentStatus
    created_at: datetime
    processed_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)
    fraud_score: float = 0.0
    gateway_transaction_id: Optional[str] = None
    failure_reason: Optional[str] = None


@dataclass
class PaymentRoute:
    """Payment routing configuration"""
    gateway: PaymentGateway
    priority: int
    enabled: bool
    cost_percentage: float
    success_rate: float
    avg_processing_time: float
    supports_methods: List[PaymentMethod]
    geographic_restrictions: List[str] = field(default_factory=list)


@dataclass
class DunningConfig:
    """Dunning management configuration"""
    max_retry_attempts: int
    retry_schedule_days: List[int]  # e.g., [1, 3, 7, 14]
    email_templates: Dict[int, str]
    grace_period_days: int
    auto_cancel_after_days: int


class PaymentOrchestrationEngine:
    """
    Enterprise payment orchestration engine.

    Features:
    - Multi-gateway support with intelligent routing
    - Automatic failover and retry logic
    - Fraud detection and prevention
    - Payment method optimization
    - Dunning management
    - Real-time analytics
    """

    def __init__(self, dunning_config: Optional[DunningConfig] = None):
        self.gateways: Dict[PaymentGateway, Dict[str, Any]] = {}
        self.routes: List[PaymentRoute] = []
        self.transactions: Dict[str, PaymentTransaction] = {}
        self.failed_payments: Dict[str, List[PaymentTransaction]] = {}

        # Dunning configuration
        self.dunning_config = dunning_config or DunningConfig(
            max_retry_attempts=4,
            retry_schedule_days=[1, 3, 7, 14],
            email_templates={},
            grace_period_days=7,
            auto_cancel_after_days=30
        )

        # Analytics
        self.payment_metrics: Dict[str, Any] = {
            "total_volume": Decimal("0"),
            "successful_payments": 0,
            "failed_payments": 0,
            "fraud_blocked": 0,
            "by_gateway": {},
            "by_method": {}
        }

    def register_gateway(
        self,
        gateway: PaymentGateway,
        credentials: Dict[str, str],
        route_config: PaymentRoute
    ) -> None:
        """Register a payment gateway"""
        self.gateways[gateway] = {
            "credentials": credentials,
            "initialized": datetime.utcnow(),
            "status": "active"
        }
        self.routes.append(route_config)
        self.routes.sort(key=lambda x: x.priority)

    def process_payment(
        self,
        customer_id: str,
        amount: Decimal,
        currency: str,
        payment_method: PaymentMethod,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PaymentTransaction:
        """
        Process a payment with intelligent routing.

        Args:
            customer_id: Customer making the payment
            amount: Payment amount
            currency: Currency code (USD, EUR, etc.)
            payment_method: Payment method type
            metadata: Additional metadata

        Returns:
            PaymentTransaction with processing result
        """
        transaction_id = self._generate_transaction_id(customer_id, amount)

        # Step 1: Fraud detection
        fraud_score = self._assess_fraud_risk(customer_id, amount, payment_method)

        if fraud_score > 0.8:  # Critical fraud risk
            return PaymentTransaction(
                transaction_id=transaction_id,
                customer_id=customer_id,
                amount=amount,
                currency=currency,
                payment_method=payment_method,
                gateway=PaymentGateway.STRIPE,  # Default
                status=PaymentStatus.FAILED,
                created_at=datetime.utcnow(),
                processed_at=datetime.utcnow(),
                metadata=metadata or {},
                fraud_score=fraud_score,
                failure_reason="High fraud risk - payment blocked"
            )

        # Step 2: Select optimal gateway
        gateway = self._select_gateway(payment_method, amount, currency)

        # Step 3: Process payment
        transaction = self._execute_payment(
            transaction_id=transaction_id,
            customer_id=customer_id,
            amount=amount,
            currency=currency,
            payment_method=payment_method,
            gateway=gateway,
            fraud_score=fraud_score,
            metadata=metadata or {}
        )

        # Step 4: Handle result
        self.transactions[transaction_id] = transaction

        if transaction.status == PaymentStatus.SUCCEEDED:
            self._update_metrics(transaction, success=True)
        else:
            self._update_metrics(transaction, success=False)
            self._initiate_dunning(transaction)

        return transaction

    def retry_failed_payment(
        self,
        transaction_id: str,
        force_gateway: Optional[PaymentGateway] = None
    ) -> PaymentTransaction:
        """Retry a failed payment"""
        original = self.transactions.get(transaction_id)
        if not original:
            raise ValueError(f"Transaction {transaction_id} not found")

        if original.status == PaymentStatus.SUCCEEDED:
            raise ValueError("Cannot retry successful payment")

        # Try different gateway if not forced
        gateway = force_gateway or self._select_fallback_gateway(
            original.gateway, original.payment_method
        )

        # Create new transaction
        retry_transaction = self._execute_payment(
            transaction_id=f"{transaction_id}_retry_{datetime.utcnow().timestamp()}",
            customer_id=original.customer_id,
            amount=original.amount,
            currency=original.currency,
            payment_method=original.payment_method,
            gateway=gateway,
            fraud_score=original.fraud_score,
            metadata={**original.metadata, "original_transaction": transaction_id}
        )

        self.transactions[retry_transaction.transaction_id] = retry_transaction
        return retry_transaction

    def process_refund(
        self,
        transaction_id: str,
        amount: Optional[Decimal] = None,
        reason: str = ""
    ) -> PaymentTransaction:
        """
        Process a refund for a transaction.

        Args:
            transaction_id: Original transaction ID
            amount: Refund amount (None for full refund)
            reason: Refund reason

        Returns:
            New transaction for refund
        """
        original = self.transactions.get(transaction_id)
        if not original:
            raise ValueError(f"Transaction {transaction_id} not found")

        if original.status != PaymentStatus.SUCCEEDED:
            raise ValueError("Can only refund successful payments")

        refund_amount = amount or original.amount

        if refund_amount > original.amount:
            raise ValueError("Refund amount exceeds original payment")

        # Create refund transaction
        refund_id = f"refund_{transaction_id}_{datetime.utcnow().timestamp()}"

        refund_transaction = PaymentTransaction(
            transaction_id=refund_id,
            customer_id=original.customer_id,
            amount=-refund_amount,  # Negative for refund
            currency=original.currency,
            payment_method=original.payment_method,
            gateway=original.gateway,
            status=PaymentStatus.REFUNDED,
            created_at=datetime.utcnow(),
            processed_at=datetime.utcnow(),
            metadata={
                "original_transaction": transaction_id,
                "refund_reason": reason,
                "partial": refund_amount < original.amount
            }
        )

        # Update original transaction status
        if refund_amount == original.amount:
            original.status = PaymentStatus.REFUNDED
        else:
            original.status = PaymentStatus.PARTIALLY_REFUNDED

        self.transactions[refund_id] = refund_transaction
        return refund_transaction

    def get_payment_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get payment analytics for time period"""
        start = start_date or (datetime.utcnow() - timedelta(days=30))
        end = end_date or datetime.utcnow()

        filtered_txns = [
            txn for txn in self.transactions.values()
            if start <= txn.created_at <= end
        ]

        successful = [t for t in filtered_txns if t.status == PaymentStatus.SUCCEEDED]
        failed = [t for t in filtered_txns if t.status == PaymentStatus.FAILED]

        total_volume = sum(t.amount for t in successful)
        success_rate = (len(successful) / len(filtered_txns) * 100) if filtered_txns else 0

        # Gateway breakdown
        gateway_stats = {}
        for gateway in PaymentGateway:
            gateway_txns = [t for t in filtered_txns if t.gateway == gateway]
            gateway_successful = [t for t in gateway_txns if t.status == PaymentStatus.SUCCEEDED]

            if gateway_txns:
                gateway_stats[gateway.value] = {
                    "total_transactions": len(gateway_txns),
                    "successful": len(gateway_successful),
                    "success_rate": (len(gateway_successful) / len(gateway_txns)) * 100,
                    "volume": float(sum(t.amount for t in gateway_successful))
                }

        # Method breakdown
        method_stats = {}
        for method in PaymentMethod:
            method_txns = [t for t in filtered_txns if t.payment_method == method]
            if method_txns:
                method_successful = [t for t in method_txns if t.status == PaymentStatus.SUCCEEDED]
                method_stats[method.value] = {
                    "total_transactions": len(method_txns),
                    "successful": len(method_successful),
                    "success_rate": (len(method_successful) / len(method_txns)) * 100
                }

        return {
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat()
            },
            "summary": {
                "total_transactions": len(filtered_txns),
                "successful_transactions": len(successful),
                "failed_transactions": len(failed),
                "success_rate": success_rate,
                "total_volume": float(total_volume),
                "average_transaction": float(total_volume / len(successful)) if successful else 0
            },
            "by_gateway": gateway_stats,
            "by_method": method_stats,
            "fraud_blocked": sum(1 for t in filtered_txns if t.fraud_score > 0.8)
        }

    # Private helper methods

    def _generate_transaction_id(self, customer_id: str, amount: Decimal) -> str:
        """Generate unique transaction ID"""
        data = f"{customer_id}_{amount}_{datetime.utcnow().isoformat()}"
        return f"txn_{hashlib.sha256(data.encode()).hexdigest()[:16]}"

    def _assess_fraud_risk(
        self,
        customer_id: str,
        amount: Decimal,
        payment_method: PaymentMethod
    ) -> float:
        """Assess fraud risk (0.0-1.0)"""
        # Simplified fraud detection - would integrate with actual fraud service
        risk_score = 0.0

        # Check for suspicious amount
        if amount > Decimal("10000"):
            risk_score += 0.3

        # Check customer history
        customer_txns = [
            t for t in self.transactions.values()
            if t.customer_id == customer_id
        ]

        if not customer_txns:
            # New customer - moderate risk
            risk_score += 0.2
        else:
            # Check for recent failures
            recent_failures = sum(
                1 for t in customer_txns[-10:]
                if t.status == PaymentStatus.FAILED
            )
            if recent_failures > 3:
                risk_score += 0.4

        return min(1.0, risk_score)

    def _select_gateway(
        self,
        payment_method: PaymentMethod,
        amount: Decimal,
        currency: str
    ) -> PaymentGateway:
        """Select optimal gateway for transaction"""
        # Filter routes that support this payment method
        compatible_routes = [
            r for r in self.routes
            if r.enabled and payment_method in r.supports_methods
        ]

        if not compatible_routes:
            # Default to first available gateway
            return list(self.gateways.keys())[0]

        # Score routes
        scored_routes = []
        for route in compatible_routes:
            # Score based on success rate, cost, and processing time
            score = (
                route.success_rate * 0.5 +
                (1.0 - route.cost_percentage / 100) * 0.3 +
                (1.0 / (route.avg_processing_time + 1)) * 0.2
            )
            scored_routes.append((score, route))

        # Return highest scoring route
        scored_routes.sort(key=lambda x: x[0], reverse=True)
        return scored_routes[0][1].gateway

    def _select_fallback_gateway(
        self,
        failed_gateway: PaymentGateway,
        payment_method: PaymentMethod
    ) -> PaymentGateway:
        """Select fallback gateway after failure"""
        compatible_routes = [
            r for r in self.routes
            if r.enabled
            and payment_method in r.supports_methods
            and r.gateway != failed_gateway
        ]

        if compatible_routes:
            return compatible_routes[0].gateway

        # Return any available gateway
        available = [g for g in self.gateways.keys() if g != failed_gateway]
        return available[0] if available else failed_gateway

    def _execute_payment(
        self,
        transaction_id: str,
        customer_id: str,
        amount: Decimal,
        currency: str,
        payment_method: PaymentMethod,
        gateway: PaymentGateway,
        fraud_score: float,
        metadata: Dict[str, Any]
    ) -> PaymentTransaction:
        """Execute payment through gateway (simulated)"""
        # This would call actual gateway API
        # For now, simulate with 90% success rate

        import random
        success = random.random() > 0.1  # 90% success

        status = PaymentStatus.SUCCEEDED if success else PaymentStatus.FAILED
        failure_reason = None if success else "Payment declined by gateway"

        return PaymentTransaction(
            transaction_id=transaction_id,
            customer_id=customer_id,
            amount=amount,
            currency=currency,
            payment_method=payment_method,
            gateway=gateway,
            status=status,
            created_at=datetime.utcnow(),
            processed_at=datetime.utcnow(),
            metadata=metadata,
            fraud_score=fraud_score,
            gateway_transaction_id=f"{gateway.value}_{transaction_id}",
            failure_reason=failure_reason
        )

    def _update_metrics(self, transaction: PaymentTransaction, success: bool) -> None:
        """Update payment metrics"""
        if success:
            self.payment_metrics["successful_payments"] += 1
            self.payment_metrics["total_volume"] += transaction.amount
        else:
            self.payment_metrics["failed_payments"] += 1

        # Update gateway metrics
        gateway_key = transaction.gateway.value
        if gateway_key not in self.payment_metrics["by_gateway"]:
            self.payment_metrics["by_gateway"][gateway_key] = {
                "total": 0,
                "successful": 0,
                "volume": Decimal("0")
            }

        self.payment_metrics["by_gateway"][gateway_key]["total"] += 1
        if success:
            self.payment_metrics["by_gateway"][gateway_key]["successful"] += 1
            self.payment_metrics["by_gateway"][gateway_key]["volume"] += transaction.amount

    def _initiate_dunning(self, transaction: PaymentTransaction) -> None:
        """Initiate dunning process for failed payment"""
        customer_id = transaction.customer_id

        if customer_id not in self.failed_payments:
            self.failed_payments[customer_id] = []

        self.failed_payments[customer_id].append(transaction)

        # Schedule retries based on dunning config
        retry_count = len(self.failed_payments[customer_id])

        if retry_count <= self.dunning_config.max_retry_attempts:
            # Would schedule retry based on retry_schedule_days
            pass  # Implementation would use task queue
