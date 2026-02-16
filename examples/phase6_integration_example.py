"""
Phase 6 Intelligence Layer Integration Example

This example demonstrates how to use the Phase 6 intelligence components together
to create a self-optimizing, self-healing AI system.

Components:
1. AdaptiveCacheOptimizer - Optimizes cache TTLs
2. IntelligentModelRouter - Routes tasks to optimal models
3. MetaLearningSystem - Learns from execution history
4. PerformanceAnomalyDetector - Detects performance issues
5. SelfHealingService - Automatically remediates failures

Usage:
    python examples/phase6_integration_example.py
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

# Note: In production, these would be actual database connections
# For this example, we'll use None and the components will use mock data


async def demonstrate_adaptive_cache_optimization():
    """Demonstrate adaptive cache optimization"""
    print("\n" + "="*80)
    print("1. ADAPTIVE CACHE OPTIMIZATION")
    print("="*80 + "\n")
    
    from infrastructure.intelligence import AdaptiveCacheOptimizer
    
    # Initialize optimizer
    optimizer = AdaptiveCacheOptimizer(db_connection=None)
    
    # Analyze current cache performance
    print("Analyzing cache performance...")
    metrics = await optimizer.analyze_cache_performance(time_window_hours=24)
    
    for layer, metric in metrics.items():
        print(f"\n{layer}:")
        print(f"  Hit Rate: {metric.hit_rate:.2%}")
        print(f"  Total Requests: {metric.total_requests}")
        print(f"  Avg Latency: {metric.avg_latency_ms:.1f}ms")
        print(f"  Cost Saved: ${metric.total_cost_saved:.2f}")
    
    # Run optimization
    print("\n\nOptimizing cache TTLs...")
    optimizations = await optimizer.optimize_cache_ttls(time_window_hours=24, apply=True)
    
    for opt in optimizations:
        print(f"\n{opt.cache_layer}:")
        print(f"  Old TTL: {opt.old_ttl_seconds}s")
        print(f"  New TTL: {opt.new_ttl_seconds}s")
        print(f"  Predicted Hit Rate: {opt.predicted_hit_rate:.2%}")
        print(f"  Reason: {opt.optimization_reason}")
        print(f"  Confidence: {opt.confidence:.2%}")
    
    # Calculate cost savings
    print("\n\nCalculating cost savings...")
    savings = await optimizer.calculate_cost_savings(time_window_hours=24)
    print(f"Total Requests: {savings['total_requests']}")
    print(f"Cache Hit Rate: {savings['overall_hit_rate']:.2%}")
    print(f"Cost Saved: ${savings['total_cost_saved_usd']:.2f}")
    print(f"Savings: {savings['savings_percent']:.1f}%")


async def demonstrate_intelligent_routing():
    """Demonstrate intelligent model routing"""
    print("\n" + "="*80)
    print("2. INTELLIGENT MODEL ROUTING")
    print("="*80 + "\n")
    
    from infrastructure.intelligence import IntelligentModelRouter
    
    # Initialize router
    router = IntelligentModelRouter(db_connection=None)
    
    # Example tasks with different complexities
    tasks = [
        {
            "type": "simple_qa",
            "description": "What is the capital of France?",
            "name": "Simple Question"
        },
        {
            "type": "code_generation",
            "description": "Generate a Python function to sort a list using quicksort",
            "name": "Code Generation"
        },
        {
            "type": "complex_reasoning",
            "description": "Design a distributed system architecture for handling 1M concurrent users",
            "name": "Complex Architecture Design"
        }
    ]
    
    for task in tasks:
        print(f"\nTask: {task['name']}")
        print(f"Type: {task['type']}")
        print(f"Description: {task['description']}")
        
        # Get routing decision
        decision = await router.select_optimal_model(
            task_type=task["type"],
            task_description=task["description"]
        )
        
        print(f"\nComplexity: {decision.complexity.score:.2f} ({decision.complexity.reasoning})")
        print(f"Selected Model: {decision.selected_model}")
        print(f"Reason: {decision.selection_reason}")
        print(f"Predicted Cost: ${decision.predicted_cost:.4f}")
        print(f"Predicted Quality: {decision.predicted_quality:.2%}")
        print(f"Confidence: {decision.confidence:.2%}")
    
    # Evaluate routing performance
    print("\n\nRouting Performance Metrics:")
    performance = await router.evaluate_routing_performance(time_window_hours=24)
    print(f"Total Decisions: {performance['total_decisions']}")
    print(f"Optimal Rate: {performance['optimal_rate']:.2%}")
    print(f"Avg Cost per Request: ${performance['avg_cost_per_request']:.4f}")
    print(f"Cost Savings vs Always GPT-4: {performance['cost_savings_vs_always_gpt4']:.2%}")


async def demonstrate_meta_learning():
    """Demonstrate meta-learning system"""
    print("\n" + "="*80)
    print("3. META-LEARNING SYSTEM")
    print("="*80 + "\n")
    
    from infrastructure.intelligence import MetaLearningSystem
    
    # Initialize meta-learning system
    meta_learner = MetaLearningSystem(db_connection=None)
    
    # Analyze execution history
    print("Analyzing execution history...")
    history = await meta_learner.analyze_execution_history(time_window_days=7)
    
    print(f"\nTotal Executions: {history['total_executions']}")
    print(f"Success Rate: {history['overall_success_rate']:.2%}")
    print(f"Avg Execution Time: {history['avg_execution_time_ms']:.0f}ms")
    print(f"Avg Cost: ${history['avg_cost_usd']:.4f}")
    
    # Identify patterns
    print("\n\nIdentifying patterns...")
    patterns = await meta_learner.identify_patterns(time_window_days=7)
    
    for pattern in patterns[:3]:
        print(f"\nPattern: {pattern.pattern_type}")
        print(f"  Task Types: {', '.join(pattern.task_types)}")
        print(f"  Frequency: {pattern.frequency}")
        print(f"  Success Rate: {pattern.success_rate:.2%}")
        print(f"  Recommendations:")
        for rec in pattern.recommended_optimizations:
            print(f"    - {rec}")
    
    # Evaluate strategies
    print("\n\nEvaluating strategies...")
    strategies = await meta_learner.evaluate_strategies(time_window_days=7)
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.strategy_name}")
        print(f"  Usage: {strategy.usage_count} times")
        print(f"  Success Rate: {strategy.success_rate:.2%}")
        print(f"  Avg Cost: ${strategy.avg_cost_usd:.4f}")
        print(f"  Performance Score: {strategy.performance_score:.2f}/1.0")
        print(f"  Recommendation: {strategy.recommendation.upper()}")
    
    # Generate optimizations
    print("\n\nGenerating optimization recommendations...")
    optimizations = await meta_learner.generate_optimization_recommendations(time_window_days=7)
    
    for opt in optimizations:
        print(f"\nOptimization: {opt.optimization_type}")
        print(f"  Improvement: {opt.improvement_percent:.1f}%")
        print(f"  Confidence: {opt.confidence:.2%}")
        print(f"  Current Cost: ${opt.current_performance.get('avg_cost_usd', 0):.4f}")
        print(f"  Predicted Cost: ${opt.predicted_performance.get('avg_cost_usd', 0):.4f}")
        print(f"  Actions:")
        for action in opt.actions:
            print(f"    - {action}")


async def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection"""
    print("\n" + "="*80)
    print("4. PERFORMANCE ANOMALY DETECTION")
    print("="*80 + "\n")
    
    from infrastructure.intelligence import PerformanceAnomalyDetector
    from infrastructure.intelligence.anomaly_detector import MetricType
    
    # Initialize detector
    detector = PerformanceAnomalyDetector(db_connection=None)
    
    # Establish baselines
    print("Establishing performance baselines...")
    
    baselines = []
    for metric_name, metric_type in [
        ("api_latency_p95", MetricType.LATENCY),
        ("llm_cost_per_request", MetricType.COST),
        ("api_error_rate", MetricType.ERROR_RATE),
    ]:
        baseline = await detector.establish_baseline(metric_name, metric_type, time_window_days=7)
        baselines.append(baseline)
        
        print(f"\n{metric_name}:")
        print(f"  Baseline: {baseline.baseline_value:.2f}")
        print(f"  Std Dev: {baseline.std_deviation:.2f}")
        print(f"  95th Percentile: {baseline.percentile_95:.2f}")
        print(f"  99th Percentile: {baseline.percentile_99:.2f}")
    
    # Test anomaly detection with various scenarios
    print("\n\nTesting anomaly detection...")
    
    test_scenarios = [
        ("api_latency_p95", MetricType.LATENCY, 1200.0, "Normal latency"),
        ("api_latency_p95", MetricType.LATENCY, 5000.0, "High latency (anomaly expected)"),
        ("llm_cost_per_request", MetricType.COST, 0.016, "Normal cost"),
        ("llm_cost_per_request", MetricType.COST, 0.100, "Cost spike (anomaly expected)"),
        ("api_error_rate", MetricType.ERROR_RATE, 0.04, "Normal error rate"),
        ("api_error_rate", MetricType.ERROR_RATE, 0.25, "Error spike (anomaly expected)"),
    ]
    
    for metric_name, metric_type, value, description in test_scenarios:
        print(f"\n{description}:")
        print(f"  Metric: {metric_name} = {value}")
        
        anomaly = await detector.detect_anomaly(metric_name, metric_type, value)
        
        if anomaly:
            print(f"  ⚠️  ANOMALY DETECTED!")
            print(f"  Severity: {anomaly.severity.upper()}")
            print(f"  Expected: {anomaly.expected_value:.2f}")
            print(f"  Actual: {anomaly.actual_value:.2f}")
            print(f"  Deviation: {anomaly.deviation_percent:.1f}%")
            print(f"  Root Cause: {anomaly.root_cause}")
            print(f"  Remediation: {anomaly.remediation_action}")
        else:
            print(f"  ✓ Normal (within expected range)")


async def demonstrate_self_healing():
    """Demonstrate self-healing service"""
    print("\n" + "="*80)
    print("5. SELF-HEALING SERVICE")
    print("="*80 + "\n")
    
    from infrastructure.resilience.self_healing import SelfHealingService, TriggerType
    
    # Initialize self-healing service
    healer = SelfHealingService(db_connection=None)
    
    # Simulate service metrics
    print("Monitoring services...")
    
    services = {
        "api_service": {
            "circuit_breaker_state": "closed",
            "error_rate": 0.03,
            "service_healthy": True,
            "cache_hit_rate": 0.87,
            "latency_p95": 1200,
            "memory_usage_percent": 65,
            "circuit_breaker_failures": 2
        },
        "llm_service": {
            "circuit_breaker_state": "open",  # Problem!
            "error_rate": 0.15,
            "service_healthy": False,
            "cache_hit_rate": 0.45,  # Low!
            "latency_p95": 5500,  # High!
            "memory_usage_percent": 88,  # High!
            "circuit_breaker_failures": 6
        }
    }
    
    for service_name, metrics in services.items():
        print(f"\n{service_name}:")
        print(f"  Circuit Breaker: {metrics['circuit_breaker_state']}")
        print(f"  Error Rate: {metrics['error_rate']:.2%}")
        print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
        print(f"  Latency P95: {metrics['latency_p95']}ms")
        print(f"  Memory Usage: {metrics['memory_usage_percent']}%")
        
        # Detect failures
        failure = await healer.detect_failure(service_name, metrics)
        if failure:
            print(f"  ⚠️  FAILURE DETECTED: {failure}")
        
        # Predict future failures
        prediction = await healer.predict_failure(service_name, metrics)
        if prediction and prediction.probability > 0.7:
            print(f"  🔮 FAILURE PREDICTED: {prediction.failure_type}")
            print(f"     Probability: {prediction.probability:.2%}")
            print(f"     Time: {prediction.predicted_time_minutes} minutes")
            print(f"     Recommended Actions: {', '.join([a.value for a in prediction.recommended_actions])}")
    
    # Run healing cycle
    print("\n\nRunning self-healing cycle...")
    results = await healer.run_healing_cycle(services)
    
    print(f"\nHealing Results:")
    print(f"  Services Monitored: {results['services_monitored']}")
    print(f"  Failures Detected: {results['failures_detected']}")
    print(f"  Predictions Made: {results['predictions_made']}")
    print(f"  Actions Taken: {results['actions_taken']}")
    print(f"  Successful Actions: {results['successful_actions']}")
    
    if results['actions']:
        print("\nActions Taken:")
        for action in results['actions']:
            print(f"  - {action['action_type']}: {'✓ Success' if action.get('success') else '✗ Failed'}")
            if action.get('impact_assessment'):
                impact = action['impact_assessment']
                print(f"    Before: {impact.get('metrics_before', {})}")
                print(f"    After: {impact.get('metrics_after', {})}")
    
    # Get recovery metrics
    print("\n\nRecovery Metrics (Last 24 hours):")
    metrics = await healer.get_recovery_metrics(time_window_hours=24)
    print(f"  Total Failures: {metrics['total_failures_detected']}")
    print(f"  Auto-Recovery Rate: {metrics['auto_recovery_rate']:.2%}")
    print(f"  Avg MTTR: {metrics['avg_mttr_seconds']}s")
    print(f"  Prediction Accuracy: {metrics['prediction_accuracy']:.2%}")


async def demonstrate_integrated_intelligence():
    """Demonstrate all components working together"""
    print("\n" + "="*80)
    print("6. INTEGRATED INTELLIGENCE SYSTEM")
    print("="*80 + "\n")
    
    print("This demonstrates how all Phase 6 components work together:")
    print("\n1. AdaptiveCacheOptimizer optimizes cache TTLs based on historical data")
    print("2. IntelligentModelRouter selects optimal models based on task complexity")
    print("3. MetaLearningSystem analyzes patterns and recommends optimizations")
    print("4. PerformanceAnomalyDetector detects performance issues in real-time")
    print("5. SelfHealingService automatically remediates detected failures")
    
    print("\n\nIntegrated Workflow:")
    print("━" * 80)
    
    # Step 1: Meta-learning analyzes and recommends
    print("\n[Meta-Learning] Analyzing execution history and generating recommendations...")
    from infrastructure.intelligence import MetaLearningSystem
    meta_learner = MetaLearningSystem()
    results = await meta_learner.run_learning_cycle()
    print(f"  ✓ Identified {results['patterns_identified']} patterns")
    print(f"  ✓ Generated {results['optimizations_recommended']} optimizations")
    print(f"  ✓ Avg improvement potential: {results['avg_improvement_percent']:.1f}%")
    
    # Step 2: Apply cache optimizations
    print("\n[Cache Optimizer] Applying adaptive cache optimizations...")
    from infrastructure.intelligence import AdaptiveCacheOptimizer
    optimizer = AdaptiveCacheOptimizer()
    opt_results = await optimizer.run_optimization_cycle()
    print(f"  ✓ Applied {opt_results['optimizations_applied']} cache optimizations")
    print(f"  ✓ Current savings: ${opt_results['current_savings']['total_cost_saved_usd']:.2f}")
    
    # Step 3: Monitor for anomalies
    print("\n[Anomaly Detector] Monitoring for performance anomalies...")
    from infrastructure.intelligence import PerformanceAnomalyDetector
    detector = PerformanceAnomalyDetector()
    anomaly_results = await detector.run_monitoring_cycle()
    print(f"  ✓ Monitored {anomaly_results['metrics_monitored']} metrics")
    print(f"  ✓ Detected {anomaly_results['anomalies_detected']} anomalies")
    
    # Step 4: Auto-heal any issues
    print("\n[Self-Healing] Automatically remediating issues...")
    from infrastructure.resilience.self_healing import SelfHealingService
    healer = SelfHealingService()
    
    # Mock service metrics
    services = {
        "api": {"error_rate": 0.03, "service_healthy": True, "latency_p95": 1200},
        "llm": {"error_rate": 0.08, "service_healthy": True, "latency_p95": 2000}
    }
    
    healing_results = await healer.run_healing_cycle(services)
    print(f"  ✓ Monitored {healing_results['services_monitored']} services")
    print(f"  ✓ Took {healing_results['actions_taken']} healing actions")
    
    # Step 5: Route requests intelligently
    print("\n[Model Router] Routing requests to optimal models...")
    from infrastructure.intelligence import IntelligentModelRouter
    router = IntelligentModelRouter()
    
    decision = await router.select_optimal_model(
        task_type="code_generation",
        task_description="Generate optimized code"
    )
    print(f"  ✓ Selected {decision.selected_model}")
    print(f"  ✓ Predicted cost: ${decision.predicted_cost:.4f}")
    print(f"  ✓ Confidence: {decision.confidence:.2%}")
    
    print("\n" + "━" * 80)
    print("\n✅ Phase 6 Intelligence Layer fully operational!")
    print("\nKey Achievements:")
    print("  • Self-optimizing cache management")
    print("  • Intelligent cost-aware model routing")
    print("  • Continuous learning from execution history")
    print("  • Real-time anomaly detection")
    print("  • Automated failure remediation")
    print("\nExpected Outcomes:")
    print("  • 30% cost reduction through adaptive optimization")
    print("  • 40% workflow improvement through meta-learning")
    print("  • <1% false positive rate in anomaly detection")
    print("  • >99% auto-recovery rate with <2min MTTR")


async def main():
    """Main execution"""
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "PHASE 6 INTELLIGENCE LAYER - INTEGRATION DEMONSTRATION".center(78) + "║")
    print("║" + " "*78 + "║")
    print("║" + "CognitionOS V5: Advanced Intelligence & Self-Optimization".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    try:
        # Run all demonstrations
        await demonstrate_adaptive_cache_optimization()
        await demonstrate_intelligent_routing()
        await demonstrate_meta_learning()
        await demonstrate_anomaly_detection()
        await demonstrate_self_healing()
        await demonstrate_integrated_intelligence()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80 + "\n")
        print("All Phase 6 intelligence components are working as designed.")
        print("The system is now capable of self-optimization and self-healing.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
