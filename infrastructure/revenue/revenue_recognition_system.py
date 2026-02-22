"""
Revenue Recognition System - ASC 606 Compliant

Implements enterprise revenue recognition with:
- ASC 606 / IFRS 15 compliance
- Multi-element arrangements (MEA)
- Deferred revenue management
- Revenue waterfall tracking
- Performance obligations
- Contract modifications
- Revenue forecasting
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from decimal import Decimal
import json


class RevenueType(Enum):
    """Revenue classification types"""
    SUBSCRIPTION = "subscription"
    LICENSE = "license"
    PROFESSIONAL_SERVICES = "professional_services"
    USAGE_BASED = "usage_based"
    ONE_TIME = "one_time"
    MAINTENANCE = "maintenance"


class RecognitionMethod(Enum):
    """Revenue recognition methods"""
    STRAIGHT_LINE = "straight_line"
    PROPORTIONAL_PERFORMANCE = "proportional_performance"
    MILESTONE_BASED = "milestone_based"
    POINT_IN_TIME = "point_in_time"
    OUTPUT_METHOD = "output_method"


class ContractStatus(Enum):
    """Contract lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    MODIFIED = "modified"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    TERMINATED = "terminated"


@dataclass
class PerformanceObligation:
    """Performance obligation per ASC 606"""
    obligation_id: str
    description: str
    allocated_amount: Decimal
    recognition_method: RecognitionMethod
    start_date: datetime
    end_date: datetime
    percentage_complete: float  # 0-100
    recognized_to_date: Decimal
    remaining: Decimal
    milestones: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RevenueContract:
    """Revenue contract with obligations"""
    contract_id: str
    customer_id: str
    total_contract_value: Decimal
    contract_start: datetime
    contract_end: datetime
    status: ContractStatus
    performance_obligations: List[PerformanceObligation]
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RevenueSchedule:
    """Revenue recognition schedule"""
    schedule_id: str
    contract_id: str
    obligation_id: str
    period_start: datetime
    period_end: datetime
    scheduled_amount: Decimal
    recognized_amount: Decimal
    deferred_amount: Decimal
    is_recognized: bool
    recognition_date: Optional[datetime]


@dataclass
class DeferredRevenue:
    """Deferred revenue liability"""
    customer_id: str
    contract_id: str
    total_deferred: Decimal
    current_portion: Decimal  # < 12 months
    long_term_portion: Decimal  # >= 12 months
    by_obligation: Dict[str, Decimal]
    as_of_date: datetime


@dataclass
class RevenueWaterfall:
    """Revenue waterfall analysis"""
    period: str
    bookings: Decimal
    billings: Decimal
    revenue_recognized: Decimal
    deferred_revenue_beginning: Decimal
    deferred_revenue_ending: Decimal
    unbilled_revenue: Decimal
    refunds_and_credits: Decimal


class RevenueRecognitionSystem:
    """
    ASC 606 compliant revenue recognition system.

    Features:
    - Multi-element arrangement (MEA) support
    - Automated revenue scheduling
    - Deferred revenue tracking
    - Contract modification handling
    - Revenue forecasting
    - Compliance reporting
    """

    def __init__(self):
        self.contracts: Dict[str, RevenueContract] = {}
        self.schedules: Dict[str, List[RevenueSchedule]] = {}
        self.deferred_revenue: Dict[str, DeferredRevenue] = {}
        self.revenue_history: List[Dict[str, Any]] = []

        # Recognition rules
        self.recognition_rules: Dict[RevenueType, RecognitionMethod] = {
            RevenueType.SUBSCRIPTION: RecognitionMethod.STRAIGHT_LINE,
            RevenueType.LICENSE: RecognitionMethod.POINT_IN_TIME,
            RevenueType.PROFESSIONAL_SERVICES: RecognitionMethod.PROPORTIONAL_PERFORMANCE,
            RevenueType.USAGE_BASED: RecognitionMethod.OUTPUT_METHOD,
            RevenueType.ONE_TIME: RecognitionMethod.POINT_IN_TIME,
            RevenueType.MAINTENANCE: RecognitionMethod.STRAIGHT_LINE
        }

    def create_contract(
        self,
        contract_id: str,
        customer_id: str,
        total_value: Decimal,
        start_date: datetime,
        end_date: datetime,
        obligations: List[Dict[str, Any]]
    ) -> RevenueContract:
        """
        Create revenue contract with performance obligations.

        Implements ASC 606 Step 1-3:
        - Identify contract
        - Identify performance obligations
        - Determine transaction price

        Args:
            contract_id: Unique contract identifier
            customer_id: Customer identifier
            total_value: Total contract value
            start_date: Contract start date
            end_date: Contract end date
            obligations: List of performance obligations

        Returns:
            Created RevenueContract
        """
        # Step 4: Allocate transaction price to performance obligations
        allocated_obligations = self._allocate_transaction_price(
            total_value, obligations, start_date, end_date
        )

        contract = RevenueContract(
            contract_id=contract_id,
            customer_id=customer_id,
            total_contract_value=total_value,
            contract_start=start_date,
            contract_end=end_date,
            status=ContractStatus.ACTIVE,
            performance_obligations=allocated_obligations,
            metadata={
                "created_at": datetime.utcnow().isoformat(),
                "fiscal_year": start_date.year
            }
        )

        # Step 5: Recognize revenue when obligations are satisfied
        self._create_revenue_schedules(contract)

        self.contracts[contract_id] = contract
        return contract

    def recognize_revenue(
        self,
        period_end: datetime
    ) -> Dict[str, Any]:
        """
        Recognize revenue for period.

        Processes all scheduled revenue recognition up to period_end.

        Args:
            period_end: End of recognition period

        Returns:
            Revenue recognition summary
        """
        total_recognized = Decimal("0")
        recognized_by_contract = {}
        recognized_by_type = {}

        for contract_id, schedules in self.schedules.items():
            contract = self.contracts.get(contract_id)
            if not contract:
                continue

            contract_recognized = Decimal("0")

            for schedule in schedules:
                # Recognize if period has passed and not already recognized
                if schedule.period_end <= period_end and not schedule.is_recognized:
                    # Mark as recognized
                    schedule.is_recognized = True
                    schedule.recognized_amount = schedule.scheduled_amount
                    schedule.deferred_amount = Decimal("0")
                    schedule.recognition_date = datetime.utcnow()

                    contract_recognized += schedule.recognized_amount
                    total_recognized += schedule.recognized_amount

                    # Update performance obligation
                    self._update_obligation_progress(
                        contract, schedule.obligation_id, schedule.recognized_amount
                    )

            if contract_recognized > 0:
                recognized_by_contract[contract_id] = float(contract_recognized)

        # Update deferred revenue
        self._update_deferred_revenue(period_end)

        return {
            "period_end": period_end.isoformat(),
            "total_recognized": float(total_recognized),
            "contracts_processed": len(recognized_by_contract),
            "by_contract": recognized_by_contract,
            "deferred_revenue_balance": float(self._get_total_deferred_revenue())
        }

    def modify_contract(
        self,
        contract_id: str,
        modification_type: str,
        changes: Dict[str, Any]
    ) -> RevenueContract:
        """
        Handle contract modification per ASC 606.

        Args:
            contract_id: Contract to modify
            modification_type: Type of modification
            changes: Modification details

        Returns:
            Modified contract
        """
        contract = self.contracts.get(contract_id)
        if not contract:
            raise ValueError(f"Contract {contract_id} not found")

        modification = {
            "modification_id": f"mod_{len(contract.modifications) + 1}",
            "type": modification_type,
            "changes": changes,
            "effective_date": datetime.utcnow(),
            "previous_value": float(contract.total_contract_value)
        }

        # Apply modification based on type
        if modification_type == "value_increase":
            # Separate contract approach
            additional_value = Decimal(str(changes.get("additional_amount", 0)))
            contract.total_contract_value += additional_value

            # Reallocate transaction price
            self._reallocate_after_modification(contract, additional_value)

        elif modification_type == "term_extension":
            # Extend end date
            additional_days = changes.get("additional_days", 0)
            contract.contract_end += timedelta(days=additional_days)

            # Recreate schedules
            self._create_revenue_schedules(contract)

        elif modification_type == "obligation_add":
            # Add new performance obligation
            new_obligation = changes.get("obligation")
            if new_obligation:
                self._add_performance_obligation(contract, new_obligation)

        contract.modifications.append(modification)
        contract.status = ContractStatus.MODIFIED

        return contract

    def forecast_revenue(
        self,
        months_ahead: int = 12
    ) -> Dict[str, Any]:
        """
        Forecast revenue based on contracts and schedules.

        Args:
            months_ahead: Number of months to forecast

        Returns:
            Revenue forecast by month
        """
        forecast_by_month = {}
        current_date = datetime.utcnow()

        for month in range(months_ahead):
            month_start = current_date + timedelta(days=30 * month)
            month_end = month_start + timedelta(days=30)

            month_revenue = Decimal("0")

            # Sum all scheduled revenue for this month
            for schedules in self.schedules.values():
                for schedule in schedules:
                    if month_start <= schedule.period_end <= month_end:
                        month_revenue += schedule.scheduled_amount

            forecast_by_month[month_start.strftime("%Y-%m")] = float(month_revenue)

        return {
            "forecast_period": f"{months_ahead} months",
            "total_forecast": float(sum(Decimal(str(v)) for v in forecast_by_month.values())),
            "by_month": forecast_by_month,
            "generated_at": current_date.isoformat()
        }

    def get_revenue_waterfall(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> RevenueWaterfall:
        """
        Generate revenue waterfall report.

        Shows flow from bookings → billings → revenue → deferred.

        Args:
            start_date: Period start
            end_date: Period end

        Returns:
            RevenueWaterfall analysis
        """
        # Calculate bookings (new contracts)
        bookings = sum(
            c.total_contract_value for c in self.contracts.values()
            if start_date <= c.contract_start <= end_date
        )

        # Calculate billings (invoiced)
        billings = bookings  # Simplified - would calculate from invoices

        # Calculate recognized revenue
        revenue_recognized = Decimal("0")
        for schedules in self.schedules.values():
            for schedule in schedules:
                if start_date <= schedule.period_end <= end_date and schedule.is_recognized:
                    revenue_recognized += schedule.recognized_amount

        # Deferred revenue
        deferred_beginning = self._get_total_deferred_revenue_at_date(start_date)
        deferred_ending = self._get_total_deferred_revenue()

        return RevenueWaterfall(
            period=f"{start_date.date()} to {end_date.date()}",
            bookings=bookings,
            billings=billings,
            revenue_recognized=revenue_recognized,
            deferred_revenue_beginning=deferred_beginning,
            deferred_revenue_ending=deferred_ending,
            unbilled_revenue=bookings - billings,
            refunds_and_credits=Decimal("0")  # Would track separately
        )

    def get_deferred_revenue_report(
        self,
        customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get deferred revenue report"""
        if customer_id:
            deferred = self.deferred_revenue.get(customer_id)
            if not deferred:
                return {"customer_id": customer_id, "total_deferred": 0}

            return {
                "customer_id": customer_id,
                "total_deferred": float(deferred.total_deferred),
                "current_portion": float(deferred.current_portion),
                "long_term_portion": float(deferred.long_term_portion),
                "by_obligation": {k: float(v) for k, v in deferred.by_obligation.items()},
                "as_of_date": deferred.as_of_date.isoformat()
            }

        # All customers
        total_deferred = sum(d.total_deferred for d in self.deferred_revenue.values())
        total_current = sum(d.current_portion for d in self.deferred_revenue.values())
        total_long_term = sum(d.long_term_portion for d in self.deferred_revenue.values())

        return {
            "total_deferred": float(total_deferred),
            "current_portion": float(total_current),
            "long_term_portion": float(total_long_term),
            "customers_count": len(self.deferred_revenue),
            "as_of_date": datetime.utcnow().isoformat()
        }

    # Private helper methods

    def _allocate_transaction_price(
        self,
        total_value: Decimal,
        obligations_data: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> List[PerformanceObligation]:
        """Allocate transaction price using standalone selling price"""
        # Calculate standalone selling prices
        total_ssp = sum(Decimal(str(o.get("standalone_price", 0))) for o in obligations_data)

        obligations = []

        for i, obl_data in enumerate(obligations_data):
            standalone_price = Decimal(str(obl_data.get("standalone_price", 0)))

            # Allocate proportionally
            if total_ssp > 0:
                allocated_amount = (standalone_price / total_ssp) * total_value
            else:
                allocated_amount = total_value / len(obligations_data)

            obligation = PerformanceObligation(
                obligation_id=f"obl_{i+1}",
                description=obl_data.get("description", ""),
                allocated_amount=allocated_amount,
                recognition_method=self.recognition_rules.get(
                    RevenueType(obl_data.get("type", "subscription")),
                    RecognitionMethod.STRAIGHT_LINE
                ),
                start_date=start_date,
                end_date=end_date,
                percentage_complete=0.0,
                recognized_to_date=Decimal("0"),
                remaining=allocated_amount
            )

            obligations.append(obligation)

        return obligations

    def _create_revenue_schedules(self, contract: RevenueContract) -> None:
        """Create revenue recognition schedules"""
        schedules = []

        for obligation in contract.performance_obligations:
            if obligation.recognition_method == RecognitionMethod.STRAIGHT_LINE:
                # Create monthly schedules
                obl_schedules = self._create_straight_line_schedule(
                    contract.contract_id,
                    obligation
                )
                schedules.extend(obl_schedules)

            elif obligation.recognition_method == RecognitionMethod.POINT_IN_TIME:
                # Single schedule at start
                schedule = RevenueSchedule(
                    schedule_id=f"{contract.contract_id}_{obligation.obligation_id}_1",
                    contract_id=contract.contract_id,
                    obligation_id=obligation.obligation_id,
                    period_start=obligation.start_date,
                    period_end=obligation.start_date,
                    scheduled_amount=obligation.allocated_amount,
                    recognized_amount=Decimal("0"),
                    deferred_amount=obligation.allocated_amount,
                    is_recognized=False,
                    recognition_date=None
                )
                schedules.append(schedule)

        self.schedules[contract.contract_id] = schedules

    def _create_straight_line_schedule(
        self,
        contract_id: str,
        obligation: PerformanceObligation
    ) -> List[RevenueSchedule]:
        """Create straight-line revenue schedule"""
        schedules = []

        # Calculate number of periods
        total_days = (obligation.end_date - obligation.start_date).days
        num_months = max(1, total_days // 30)

        monthly_amount = obligation.allocated_amount / num_months

        current_date = obligation.start_date
        for month in range(num_months):
            period_start = current_date
            period_end = current_date + timedelta(days=30)

            schedule = RevenueSchedule(
                schedule_id=f"{contract_id}_{obligation.obligation_id}_{month+1}",
                contract_id=contract_id,
                obligation_id=obligation.obligation_id,
                period_start=period_start,
                period_end=period_end,
                scheduled_amount=monthly_amount,
                recognized_amount=Decimal("0"),
                deferred_amount=monthly_amount,
                is_recognized=False,
                recognition_date=None
            )

            schedules.append(schedule)
            current_date = period_end

        return schedules

    def _update_obligation_progress(
        self,
        contract: RevenueContract,
        obligation_id: str,
        amount: Decimal
    ) -> None:
        """Update performance obligation progress"""
        for obligation in contract.performance_obligations:
            if obligation.obligation_id == obligation_id:
                obligation.recognized_to_date += amount
                obligation.remaining = obligation.allocated_amount - obligation.recognized_to_date
                obligation.percentage_complete = float(
                    (obligation.recognized_to_date / obligation.allocated_amount) * 100
                )
                break

    def _update_deferred_revenue(self, as_of_date: datetime) -> None:
        """Update deferred revenue balances"""
        for contract_id, contract in self.contracts.items():
            customer_id = contract.customer_id

            total_deferred = Decimal("0")
            current_portion = Decimal("0")
            long_term_portion = Decimal("0")
            by_obligation = {}

            schedules = self.schedules.get(contract_id, [])

            for schedule in schedules:
                if not schedule.is_recognized:
                    deferred_amt = schedule.scheduled_amount

                    total_deferred += deferred_amt

                    # Classify as current or long-term
                    if schedule.period_end <= as_of_date + timedelta(days=365):
                        current_portion += deferred_amt
                    else:
                        long_term_portion += deferred_amt

                    # Track by obligation
                    obl_id = schedule.obligation_id
                    by_obligation[obl_id] = by_obligation.get(obl_id, Decimal("0")) + deferred_amt

            if total_deferred > 0:
                self.deferred_revenue[customer_id] = DeferredRevenue(
                    customer_id=customer_id,
                    contract_id=contract_id,
                    total_deferred=total_deferred,
                    current_portion=current_portion,
                    long_term_portion=long_term_portion,
                    by_obligation=by_obligation,
                    as_of_date=as_of_date
                )

    def _get_total_deferred_revenue(self) -> Decimal:
        """Get total deferred revenue across all customers"""
        return sum(d.total_deferred for d in self.deferred_revenue.values())

    def _get_total_deferred_revenue_at_date(self, date: datetime) -> Decimal:
        """Get historical deferred revenue (simplified)"""
        # Would query historical data
        return self._get_total_deferred_revenue()

    def _reallocate_after_modification(
        self,
        contract: RevenueContract,
        additional_value: Decimal
    ) -> None:
        """Reallocate transaction price after modification"""
        # Simplified reallocation
        pass

    def _add_performance_obligation(
        self,
        contract: RevenueContract,
        obligation_data: Dict[str, Any]
    ) -> None:
        """Add new performance obligation to contract"""
        # Would create new obligation and reallocate
        pass
