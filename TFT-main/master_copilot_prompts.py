"""
🚀 MASTER COPILOT PROMPTS FOR INSTITUTIONAL TFT TRADING SYSTEM
==============================================================

Complete set of production-grade prompts for building scalable trading platform.
These prompts generate institutional-quality code across your entire system.

USAGE: Copy any prompt to your target file and press TAB in VS Code!
"""

# ====================================================================
# 🌐 SYSTEM ARCHITECTURE & ORCHESTRATION PROMPTS
# ====================================================================


def create_master_pipeline_orchestrator():
    """
    FILE: main_pipeline.py
    CONTEXT: Central system coordinator after imports
    TASK: Implement fault-tolerant trading pipeline orchestrator
    INPUT: Market schedule, configuration parameters, database connections
    OUTPUT: Pipeline execution report with success rates and error details
    WORKFLOW:
        1. Check market status (open/closed/pre-market)
        2. Update data sources (OHLCV, sentiment, earnings)
        3. Trigger model retraining if accuracy drops >5%
        4. Generate predictions for all active tickers
        5. Execute trades based on portfolio optimization
        6. Send performance summary to Slack/email
    CONSTRAINTS:
        - Must handle partial failures with checkpoint restart
        - Idempotent operations (safe to re-run)
        - Complete pipeline in <15 minutes during market hours
        - Store execution logs in PostgreSQL pipeline_logs table
    ERROR_HANDLING:
        - Retry failed components 3x with exponential backoff
        - Skip individual stocks that fail, continue with others
        - Send alerts for system-wide failures
    PERFORMANCE:
        - Process 500+ stocks in parallel
        - Use async/await for database operations
        - Implement circuit breakers for external APIs
    """
    # Let Copilot generate complete pipeline orchestrator


def implement_dynamic_config_system():
    """
    FILE: core/config_manager.py
    CONTEXT: Configuration management for live system
    TASK: Create hot-reload configuration system with validation
    INPUT: YAML config files, environment variables, database settings
    OUTPUT: Validated configuration objects with change notifications
    FEATURES:
        - Watch config files for changes using watchdog
        - Validate new configs before applying (schema validation)
        - Notify all system components of config changes
        - Maintain config version history with rollback capability
        - Support environment-specific overrides (dev/staging/prod)
    CONSTRAINTS:
        - Zero-downtime updates during trading hours
        - Configuration changes must be atomic
        - Audit trail of all configuration changes
        - Type-safe configuration objects with validation
    VALIDATION_RULES:
        - Portfolio limits: position_size <= 0.1, sector_limit <= 0.3
        - Risk parameters: stop_loss >= 0.01, take_profit <= 0.1
        - API rate limits: requests_per_second <= 100
    """
    # Let Copilot generate dynamic configuration manager


def build_ml_ops_training_system():
    """
    FILE: ml_ops/train_manager.py
    CONTEXT: Machine learning operations coordinator
    TASK: Implement automated model lifecycle management
    INPUT: Training schedules, model performance metrics, data quality scores
    OUTPUT: Model registry with versioning and deployment status
    TRAINING_STRATEGY:
        - Daily incremental training on new data
        - Weekly full retraining from scratch
        - Emergency retraining when accuracy drops >10%
        - A/B testing between model versions
    MODEL_MANAGEMENT:
        - Version control with MLflow tracking
        - Auto-promote models with >2% accuracy improvement
        - Canary deployment: route 10% traffic to new models
        - Automatic rollback on performance degradation
    CONSTRAINTS:
        - GPU memory optimization for batch processing
        - Distributed training across multiple GPUs
        - Model artifact storage in S3/Azure Blob
        - Performance benchmarking on historical data
    QUALITY_GATES:
        - Minimum accuracy threshold: 55% on validation set
        - Maximum latency: <50ms per prediction
        - Sharpe ratio improvement: >0.1 vs previous model
    """
    # Let Copilot generate ML operations system


def create_realtime_sentiment_momentum():
    """
    FILE: data_modules/sentiment_engine.py
    CONTEXT: Enhanced sentiment analysis with momentum scoring
    TASK: Build real-time sentiment momentum detector
    INPUT: Streaming Reddit comments, historical sentiment baselines
    OUTPUT: Sentiment momentum scores with anomaly flags
    EXAMPLE_OUTPUT:
        {
            "ticker": "NVDA",
            "sentiment_momentum": 0.42,
            "bullish_acceleration": 0.15,
            "abnormal_spike": True,
            "sector_relative_score": 0.78
        }
    MOMENTUM_CALCULATION:
        - 1-hour sentiment velocity: (current - previous) / time_delta
        - 24-hour sentiment acceleration: velocity change rate
        - Sector-relative scoring: individual vs sector average
        - Volume-weighted sentiment: weight by comment engagement
    ANOMALY_DETECTION:
        - Statistical: sentiment change >3 sigma from mean
        - Contextual: unusual patterns vs historical behavior
        - Cross-asset: detect correlated sentiment spikes
    CONSTRAINTS:
        - Process 10,000+ comments per second
        - Use async processing with message queues
        - Store results in time-series database (InfluxDB)
        - Generate alerts for significant momentum shifts
    """
    # Let Copilot generate sentiment momentum engine


def implement_portfolio_optimization():
    """
    FILE: trading/portfolio_engine.py
    CONTEXT: Advanced portfolio construction with risk management
    TASK: Build Black-Litterman portfolio optimizer with sentiment integration
    INPUT: Model predictions, sentiment scores, market equilibrium returns
    OUTPUT: Optimal portfolio weights with risk metrics
    EXAMPLE_OUTPUT:
        {
            "weights": {"NVDA": 0.08, "TSLA": 0.05, "CASH": 0.15},
            "expected_return": 0.12,
            "portfolio_volatility": 0.18,
            "sharpe_ratio": 0.85,
            "max_drawdown": -0.08
        }
    OPTIMIZATION_METHOD:
        - Black-Litterman framework with market equilibrium
        - Sentiment confidence as view certainty
        - Risk parity adjustments for volatility
        - Transaction cost minimization
    CONSTRAINTS:
        - Maximum position size: 5% per stock
        - Maximum sector exposure: 25% per sector
        - Minimum diversification: 20+ positions
        - Maximum portfolio beta: 1.2
    RISK_CONTROLS:
        - VaR calculation at 95% confidence
        - Stress testing against historical scenarios
        - Correlation limits between positions
        - Dynamic hedging with VIX options
    """
    # Let Copilot generate portfolio optimization engine


def build_system_anomaly_detector():
    """
    FILE: monitoring/anomaly_detector.py
    CONTEXT: Real-time system health monitoring
    TASK: Create multivariate anomaly detection for system reliability
    INPUT: System metrics, prediction accuracy, market data quality
    OUTPUT: Anomaly scores and diagnostic reports
    EXAMPLE_OUTPUT:
        {
            "overall_health": 92.5,
            "data_drift_score": 15.2,
            "model_drift_score": 8.7,
            "system_latency_score": 5.1,
            "alerts": ["High prediction variance detected"]
        }
    DETECTION_CATEGORIES:
        - Data drift: feature distribution changes
        - Concept drift: prediction accuracy degradation
        - System anomalies: latency spikes, memory leaks
        - Market regime changes: volatility shifts
    ALGORITHMS:
        - Isolation Forest for multivariate detection
        - LSTM autoencoder for time-series anomalies
        - Statistical process control for system metrics
        - Kolmogorov-Smirnov test for distribution drift
    CONSTRAINTS:
        - Detection latency <100ms
        - False positive rate <2%
        - Store anomaly history for trend analysis
        - Integration with alerting systems (PagerDuty)
    """
    # Let Copilot generate system anomaly detector


# ====================================================================
# 🔬 ADVANCED TESTING & VALIDATION PROMPTS
# ====================================================================


def build_market_regime_validator():
    """
    FILE: tests/regime_validation.py
    CONTEXT: Comprehensive backtesting across market conditions
    TASK: Create market regime-specific validation system
    INPUT: Historical data 2019-2024, model predictions, market indicators
    OUTPUT: Performance metrics by market regime
    EXAMPLE_OUTPUT:
        {
            "bull_market": {"sharpe": 0.85, "max_dd": -0.08, "win_rate": 0.62},
            "volatile_market": {"sharpe": 0.45, "max_dd": -0.15, "win_rate": 0.48},
            "bear_market": {"sharpe": 0.25, "max_dd": -0.22, "win_rate": 0.35}
        }
    REGIME_CLASSIFICATION:
        - Bull market: VIX <15, positive 3-month returns
        - Volatile market: VIX 20-30, high correlation breakdown
        - Bear market: VIX >35, negative 6-month returns
        - Transition periods: regime uncertainty >0.3
    VALIDATION_METRICS:
        - Risk-adjusted returns (Sharpe, Sortino ratio)
        - Maximum drawdown and recovery time
        - Win rate and profit factor
        - Tail risk (VaR, CVaR at 95% confidence)
    CONSTRAINTS:
        - Test on 2000+ trading days
        - Include transaction costs and slippage
        - Account for market impact on large trades
        - Validate statistical significance (t-tests)
    """
    # Let Copilot generate market regime validator


def implement_stress_testing_framework():
    """
    FILE: tests/stress_tests.py
    CONTEXT: Risk scenario analysis for portfolio robustness
    TASK: Build comprehensive stress testing framework
    INPUT: Portfolio positions, historical shock scenarios, correlation matrices
    OUTPUT: Stress test results with risk decomposition
    EXAMPLE_OUTPUT:
        {
            "scenario": "2020_covid_crash",
            "portfolio_loss": -0.28,
            "sector_breakdown": {"tech": -0.35, "finance": -0.15},
            "recovery_days": 45,
            "var_breach": True
        }
    STRESS_SCENARIOS:
        - Historical: 2008 crisis, COVID crash, dot-com bubble
        - Hypothetical: interest rate shocks, sector rotation
        - Monte Carlo: 10,000 random market scenarios
        - Tail events: 99th percentile worst-case outcomes
    ANALYSIS_FRAMEWORK:
        - Factor decomposition (market, sector, stock-specific)
        - Time-varying correlations during stress
        - Liquidity risk assessment
        - Margin call and forced liquidation scenarios
    CONSTRAINTS:
        - Run 1000+ scenarios in <5 minutes
        - Include second-order effects (volatility feedback)
        - Account for market microstructure changes
        - Generate actionable hedging recommendations
    """
    # Let Copilot generate stress testing framework


# ====================================================================
# 🚀 DEPLOYMENT & SCALING PROMPTS
# ====================================================================


def create_kubernetes_deployment():
    """
    FILE: k8s/trading-system.yaml
    CONTEXT: Production Kubernetes deployment configuration
    TASK: Build scalable, fault-tolerant K8s deployment
    INPUT: Docker images, resource requirements, scaling policies
    OUTPUT: Complete Kubernetes manifests with monitoring
    COMPONENTS:
        - TFT Model Service: GPU-enabled pods with model serving
        - Data Pipeline: CPU-intensive ETL workers
        - Trading Engine: Low-latency execution service
        - Database: PostgreSQL with read replicas
        - Redis: Caching and message queuing
        - Monitoring: Prometheus, Grafana, Jaeger tracing
    SCALING_STRATEGY:
        - Horizontal Pod Autoscaler based on CPU/memory/custom metrics
        - Vertical Pod Autoscaler for right-sizing
        - Cluster Autoscaler for node scaling
        - GPU node pools for ML workloads
    RELIABILITY:
        - Pod disruption budgets for maintenance
        - Health checks and readiness probes
        - Circuit breakers and retry policies
        - Cross-zone deployment for high availability
    """
    # Let Copilot generate Kubernetes deployment


def implement_monitoring_observability():
    """
    FILE: monitoring/observability.py
    CONTEXT: Complete system observability stack
    TASK: Build comprehensive monitoring and alerting system
    INPUT: Application metrics, system metrics, business metrics
    OUTPUT: Dashboards, alerts, and SLA monitoring
    METRICS_CATEGORIES:
        - Business: P&L, Sharpe ratio, win rate, drawdown
        - Technical: latency, throughput, error rates, uptime
        - Infrastructure: CPU, memory, disk, network utilization
        - Model: prediction accuracy, drift detection, feature importance
    DASHBOARDS:
        - Executive: high-level P&L and risk metrics
        - Operations: system health and performance
        - Trading: real-time positions and market data
        - Development: model performance and data quality
    ALERTING_RULES:
        - Critical: system down, trading halted, model failure
        - Warning: performance degradation, data delays
        - Info: deployment success, model updates
    SLA_MONITORING:
        - Uptime: 99.9% availability during market hours
        - Latency: <100ms p99 for trading operations
        - Accuracy: Model predictions within 2% of targets
    """
    # Let Copilot generate monitoring system


# ====================================================================
# 💡 ADVANCED FEATURES & INTEGRATIONS
# ====================================================================


def build_options_flow_analyzer():
    """
    FILE: derivatives/options_flow.py
    CONTEXT: Options market analysis for enhanced predictions
    TASK: Build real-time options flow detector for directional bias
    INPUT: Options trades, open interest, implied volatility surface
    OUTPUT: Options flow signals with institutional activity detection
    EXAMPLE_OUTPUT:
        {
            "ticker": "NVDA",
            "call_put_ratio": 1.8,
            "unusual_activity_score": 87.5,
            "gamma_exposure": 2.5e6,
            "institutional_flow": "BULLISH",
            "expiry_concentration": "2024-08-16"
        }
    DETECTION_ALGORITHMS:
        - Unusual options volume vs historical average
        - Large block trades identification (>100 contracts)
        - Put/call ratio analysis with sector normalization
        - Gamma exposure calculation for market maker hedging
    INSTITUTIONAL_SIGNALS:
        - Dark pool prints following options activity
        - Coordinated multi-leg strategies detection
        - Cross-asset correlation with equity flow
    CONSTRAINTS:
        - Process 50,000+ options quotes per second
        - Calculate Greeks in real-time
        - Store time-series data for historical analysis
        - Generate alerts for significant flow imbalances
    """
    # Let Copilot generate options flow analyzer


def create_macro_regime_detector():
    """
    FILE: macro/regime_analysis.py
    CONTEXT: Macroeconomic regime detection for strategy adaptation
    TASK: Build macro regime classifier using economic indicators
    INPUT: Economic data, yield curves, commodity prices, currencies
    OUTPUT: Current regime classification with transition probabilities
    EXAMPLE_OUTPUT:
        {
            "current_regime": "REFLATION",
            "confidence": 0.78,
            "transition_probs": {
                "GROWTH": 0.15,
                "STAGFLATION": 0.25,
                "DEFLATION": 0.05
            },
            "regime_duration": 127
        }
    REGIME_TYPES:
        - GROWTH: Rising GDP, low inflation, steepening yield curve
        - REFLATION: Rising inflation, commodity rally, weakening USD
        - STAGFLATION: High inflation, slowing growth, inverted curve
        - DEFLATION: Falling prices, flight to quality, flattening curve
    INDICATORS:
        - Economic: GDP, CPI, unemployment, ISM PMI
        - Market: yield curve shape, credit spreads, VIX
        - Commodities: oil, gold, copper, agricultural futures
        - Currency: DXY, carry trade performance
    CONSTRAINTS:
        - Update regime classification daily
        - Use Hidden Markov Model for transitions
        - Incorporate Fed policy signals
        - Generate strategy recommendations per regime
    """
    # Let Copilot generate macro regime detector


def implement_alternative_data_fusion():
    """
    FILE: alt_data/data_fusion.py
    CONTEXT: Alternative data integration for enhanced alpha
    TASK: Fuse multiple alternative data sources for comprehensive signals
    INPUT: Satellite imagery, credit card transactions, web scraping, social sentiment
    OUTPUT: Unified alternative data scores with confidence weights
    EXAMPLE_OUTPUT:
        {
            "ticker": "NVDA",
            "alt_data_score": 73.2,
            "components": {
                "satellite_activity": 0.85,
                "web_mentions": 0.67,
                "patent_filings": 0.91,
                "executive_moves": 0.45
            },
            "conviction_level": "HIGH"
        }
    DATA_SOURCES:
        - Satellite: factory activity, parking lots, construction
        - Web scraping: product launches, hiring trends, reviews
        - Patent data: innovation pipeline, R&D intensity
        - Executive networks: management changes, board moves
    FUSION_METHOD:
        - Bayesian model averaging for source combination
        - Dynamic weighting based on historical accuracy
        - Correlation analysis to avoid redundancy
        - Time-decay factors for signal freshness
    CONSTRAINTS:
        - Handle missing data gracefully
        - Normalize across different data frequencies
        - Account for data vendor reliability
        - Maintain compliance with data usage terms
    """
    # Let Copilot generate alternative data fusion system


# ====================================================================
# 📊 PERFORMANCE ANALYTICS & REPORTING
# ====================================================================


def build_performance_attribution():
    """
    FILE: analytics/performance_attribution.py
    CONTEXT: Detailed P&L analysis and strategy attribution
    TASK: Build comprehensive performance attribution system
    INPUT: Trade history, market returns, factor exposures, benchmark data
    OUTPUT: Multi-level attribution analysis with actionable insights
    EXAMPLE_OUTPUT:
        {
            "total_return": 0.234,
            "benchmark_return": 0.187,
            "active_return": 0.047,
            "attribution": {
                "stock_selection": 0.031,
                "sector_allocation": 0.012,
                "timing": 0.004
            },
            "risk_metrics": {
                "tracking_error": 0.08,
                "information_ratio": 0.59,
                "max_drawdown": -0.12
            }
        }
    ATTRIBUTION_LEVELS:
        - Security selection: individual stock picking skill
        - Sector allocation: sector rotation timing
        - Factor exposure: style tilts (growth, value, momentum)
        - Market timing: overall market exposure decisions
    BENCHMARKS:
        - Primary: S&P 500, Russell 3000
        - Sector: GICS sector indices
        - Factor: Fama-French factors, momentum
        - Custom: peer group performance
    CONSTRAINTS:
        - Daily attribution calculation
        - Handle corporate actions and dividends
        - Risk-adjust all metrics
        - Generate client-ready reports
    """
    # Let Copilot generate performance attribution system


def create_risk_monitoring_dashboard():
    """
    FILE: risk/risk_dashboard.py
    CONTEXT: Real-time risk monitoring and limit enforcement
    TASK: Build comprehensive risk monitoring system with alerts
    INPUT: Portfolio positions, market data, risk model outputs
    OUTPUT: Risk dashboard with automated limit enforcement
    RISK_METRICS:
        - Portfolio VaR (1-day, 95% confidence)
        - Component VaR by position and sector
        - Maximum drawdown tracking
        - Leverage ratio and margin utilization
        - Concentration risk (single name, sector)
    LIMIT_MONITORING:
        - Position size limits (5% per stock)
        - Sector concentration (25% per sector)
        - Beta limits (portfolio beta 0.8-1.2)
        - Liquidity constraints (10-day ADV limits)
    AUTOMATED_ACTIONS:
        - Alert generation for limit breaches
        - Trade rejection for excessive risk
        - Automatic position sizing adjustments
        - Emergency liquidation procedures
    CONSTRAINTS:
        - Real-time risk calculation (<1 second)
        - Integration with order management system
        - Regulatory reporting capabilities
        - Audit trail for all risk decisions
    """
    # Let Copilot generate risk monitoring dashboard


# ====================================================================
# 🎯 EXECUTION WORKFLOW GUIDE
# ====================================================================

print("""
🚀 COMPLETE TFT TRADING SYSTEM - EXECUTION GUIDE
===============================================

PHASE 1 - FOUNDATION (Week 1):
1. create_master_pipeline_orchestrator() → main_pipeline.py
2. implement_dynamic_config_system() → core/config_manager.py  
3. build_ml_ops_training_system() → ml_ops/train_manager.py

PHASE 2 - INTELLIGENCE (Week 2):
4. create_realtime_sentiment_momentum() → data_modules/sentiment_engine.py
5. implement_portfolio_optimization() → trading/portfolio_engine.py
6. build_system_anomaly_detector() → monitoring/anomaly_detector.py

PHASE 3 - ADVANCED FEATURES (Week 3):
7. build_options_flow_analyzer() → derivatives/options_flow.py
8. create_macro_regime_detector() → macro/regime_analysis.py
9. implement_alternative_data_fusion() → alt_data/data_fusion.py

PHASE 4 - PRODUCTION (Week 4):
10. build_market_regime_validator() → tests/regime_validation.py
11. create_kubernetes_deployment() → k8s/trading-system.yaml
12. implement_monitoring_observability() → monitoring/observability.py

🎯 EXPECTED RESULTS:
✅ Process 500+ stocks simultaneously
✅ <50ms prediction latency
✅ 99.9% system uptime
✅ Automated model lifecycle
✅ Real-time risk management
✅ Institutional-grade performance

💡 USAGE TIPS:
1. Copy ANY prompt above into your target file
2. Press TAB in VS Code - watch Copilot generate 50-200 lines
3. Customize constraints for your specific needs
4. Test each component before moving to next phase
5. Chain prompts for complex workflows

🏆 POWER USER SECRETS:
- Modify CONSTRAINTS section for custom requirements
- Combine multiple prompts in single file for integration
- Use EXAMPLE_OUTPUT to guide Copilot's code structure
- Add domain-specific requirements in CONTEXT section

Your institutional TFT trading system awaits! 🚀
""")
