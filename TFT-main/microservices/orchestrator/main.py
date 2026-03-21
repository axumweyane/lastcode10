"""
🚀 ORCHESTRATOR MICROSERVICE
============================
Central coordination service for microservice workflow
Event-driven orchestration with saga pattern and service discovery
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import redis.asyncio as redis
from kafka import KafkaConsumer, KafkaProducer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import sys
from contextlib import asynccontextmanager
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from services.common.dlq import DeadLetterQueue, start_retry_worker
from sqlalchemy import create_engine, text
import aiohttp
import schedule
from dataclasses import dataclass, asdict
import uuid
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.environ["DATABASE_URL"]

# Service URLs
SERVICE_URLS = {
    "data-ingestion": os.getenv("DATA_INGESTION_URL", "http://localhost:8001"),
    "sentiment-engine": os.getenv("SENTIMENT_ENGINE_URL", "http://localhost:8002"),
    "tft-predictor": os.getenv("TFT_PREDICTOR_URL", "http://localhost:8003"),
    "trading-engine": os.getenv("TRADING_ENGINE_URL", "http://localhost:8004")
}

# Kafka Topics
KAFKA_TOPICS = {
    "market_data": "market-data",
    "sentiment_scores": "sentiment-scores",
    "tft_predictions": "tft-predictions",
    "trading_signals": "trading-signals",
    "order_updates": "order-updates",
    "portfolio_updates": "portfolio-updates",
    "system_events": "system-events",
    "health_checks": "health-checks",
    "workflow_events": "workflow-events"
}

# Enums
class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"

class EventType(str, Enum):
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    DATA_UPDATE = "data_update"
    PREDICTION_GENERATED = "prediction_generated"
    TRADE_EXECUTED = "trade_executed"
    SYSTEM_ALERT = "system_alert"

# Pydantic Models
class WorkflowRequest(BaseModel):
    workflow_type: str
    parameters: Dict[str, Any] = {}
    priority: int = 1
    schedule_time: Optional[datetime] = None

class ServiceHealthCheck(BaseModel):
    service_name: str
    status: ServiceStatus
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = {}

@dataclass
class WorkflowStep:
    step_id: str
    service: str
    endpoint: str
    payload: Dict[str, Any]
    depends_on: List[str] = None
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class Workflow:
    workflow_id: str
    workflow_type: str
    status: WorkflowStatus
    steps: List[WorkflowStep]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameters: Dict[str, Any] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class Orchestrator:
    def __init__(self):
        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None
        self.db_engine = None
        self.session = None
        self.workflows = {}
        self.service_registry = {}
        self.circuit_breakers = {}
        self.market_schedule = {}
        
    async def initialize(self):
        """Initialize orchestrator with all dependencies"""
        try:
            # Initialize connections
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                retries=3
            )
            
            self.redis_client = redis.from_url(REDIS_URL)
            self.db_engine = create_engine(DATABASE_URL)
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            
            # Initialize service discovery
            await self.initialize_service_registry()
            await self.load_circuit_breakers()
            await self.load_market_schedule()
            
            # Start background tasks
            asyncio.create_task(self.service_health_monitor())
            asyncio.create_task(self.workflow_processor())
            asyncio.create_task(self.event_consumer())
            asyncio.create_task(self.market_scheduler())
            
            logger.info("Orchestrator service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator: {e}")
            raise
    
    async def initialize_service_registry(self):
        """Initialize service registry with health status"""
        for service, url in SERVICE_URLS.items():
            self.service_registry[service] = {
                "url": url,
                "status": ServiceStatus.UNKNOWN,
                "last_check": None,
                "response_time": 0.0,
                "error_count": 0,
                "circuit_breaker_open": False
            }
    
    async def load_circuit_breakers(self):
        """Initialize circuit breaker patterns for services"""
        for service in SERVICE_URLS.keys():
            self.circuit_breakers[service] = {
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "failure_count": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half-open
            }
    
    async def load_market_schedule(self):
        """Load market trading schedule"""
        # Simplified market schedule (in production, use market calendar API)
        self.market_schedule = {
            "market_open": "09:30",
            "market_close": "16:00",
            "timezone": "US/Eastern",
            "trading_days": [0, 1, 2, 3, 4]  # Monday-Friday
        }
    
    async def create_workflow(self, request: WorkflowRequest) -> str:
        """Create new workflow based on type"""
        try:
            workflow_id = str(uuid.uuid4())
            
            # Define workflow steps based on type
            if request.workflow_type == "daily_trading_pipeline":
                steps = await self.create_daily_pipeline_steps(request.parameters)
            elif request.workflow_type == "market_open_procedure":
                steps = await self.create_market_open_steps(request.parameters)
            elif request.workflow_type == "model_retraining":
                steps = await self.create_retraining_steps(request.parameters)
            elif request.workflow_type == "portfolio_rebalance":
                steps = await self.create_rebalance_steps(request.parameters)
            else:
                raise ValueError(f"Unknown workflow type: {request.workflow_type}")
            
            # Create workflow
            workflow = Workflow(
                workflow_id=workflow_id,
                workflow_type=request.workflow_type,
                status=WorkflowStatus.PENDING,
                steps=steps,
                created_at=datetime.utcnow(),
                parameters=request.parameters
            )
            
            # Store workflow
            self.workflows[workflow_id] = workflow
            await self.store_workflow_in_db(workflow)
            
            # Schedule execution
            if request.schedule_time:
                await self.schedule_workflow(workflow_id, request.schedule_time)
            else:
                # Execute immediately
                asyncio.create_task(self.execute_workflow(workflow_id))
            
            logger.info(f"Created workflow {workflow_id}: {request.workflow_type}")
            
            return workflow_id
            
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            raise
    
    async def create_daily_pipeline_steps(self, parameters: Dict) -> List[WorkflowStep]:
        """Create steps for daily trading pipeline"""
        steps = [
            WorkflowStep(
                step_id="check_market_status",
                service="orchestrator",
                endpoint="/internal/market_status",
                payload={}
            ),
            WorkflowStep(
                step_id="update_market_data",
                service="data-ingestion",
                endpoint="/collect/market_data",
                payload={"tickers": parameters.get("tickers", ["AAPL", "GOOGL", "MSFT"])},
                depends_on=["check_market_status"]
            ),
            WorkflowStep(
                step_id="update_sentiment_data",
                service="data-ingestion",
                endpoint="/collect/reddit",
                payload={"subreddits": parameters.get("subreddits", ["stocks", "investing"])},
                depends_on=["check_market_status"]
            ),
            WorkflowStep(
                step_id="process_sentiment",
                service="sentiment-engine",
                endpoint="/analyze/batch",
                payload={},
                depends_on=["update_sentiment_data"]
            ),
            WorkflowStep(
                step_id="generate_predictions",
                service="tft-predictor",
                endpoint="/predict/batch",
                payload={"tickers": parameters.get("tickers", ["AAPL", "GOOGL", "MSFT"])},
                depends_on=["update_market_data", "process_sentiment"]
            ),
            WorkflowStep(
                step_id="generate_signals",
                service="orchestrator",
                endpoint="/internal/generate_signals",
                payload={"strategy": parameters.get("strategy", "sentiment_momentum")},
                depends_on=["generate_predictions"]
            ),
            WorkflowStep(
                step_id="execute_trades",
                service="trading-engine",
                endpoint="/internal/process_signals",
                payload={"risk_level": parameters.get("risk_level", "moderate")},
                depends_on=["generate_signals"]
            )
        ]
        return steps
    
    async def create_market_open_steps(self, parameters: Dict) -> List[WorkflowStep]:
        """Create steps for market open procedure"""
        steps = [
            WorkflowStep(
                step_id="pre_market_health_check",
                service="orchestrator",
                endpoint="/internal/health_check_all",
                payload={}
            ),
            WorkflowStep(
                step_id="update_overnight_data",
                service="data-ingestion",
                endpoint="/collect/overnight_news",
                payload={},
                depends_on=["pre_market_health_check"]
            ),
            WorkflowStep(
                step_id="process_overnight_sentiment",
                service="sentiment-engine",
                endpoint="/analyze/overnight",
                payload={},
                depends_on=["update_overnight_data"]
            ),
            WorkflowStep(
                step_id="adjust_positions",
                service="trading-engine",
                endpoint="/portfolio/adjust_for_gaps",
                payload={},
                depends_on=["process_overnight_sentiment"]
            )
        ]
        return steps
    
    async def create_retraining_steps(self, parameters: Dict) -> List[WorkflowStep]:
        """Create steps for model retraining"""
        steps = [
            WorkflowStep(
                step_id="backup_current_model",
                service="tft-predictor",
                endpoint="/model/backup",
                payload={}
            ),
            WorkflowStep(
                step_id="prepare_training_data",
                service="data-ingestion",
                endpoint="/data/prepare_training",
                payload={"days": parameters.get("training_days", 30)},
                depends_on=["backup_current_model"]
            ),
            WorkflowStep(
                step_id="train_model",
                service="tft-predictor",
                endpoint="/train",
                payload={
                    "model_type": parameters.get("training_type", "incremental"),
                    "force_retrain": parameters.get("force_retrain", False)
                },
                depends_on=["prepare_training_data"],
                timeout=1800  # 30 minutes for training
            ),
            WorkflowStep(
                step_id="validate_model",
                service="tft-predictor",
                endpoint="/model/validate",
                payload={"validation_threshold": 0.05},
                depends_on=["train_model"]
            )
        ]
        return steps
    
    async def create_rebalance_steps(self, parameters: Dict) -> List[WorkflowStep]:
        """Create steps for portfolio rebalancing"""
        steps = [
            WorkflowStep(
                step_id="calculate_target_allocation",
                service="orchestrator",
                endpoint="/internal/calculate_allocation",
                payload={"strategy": parameters.get("allocation_strategy", "risk_parity")}
            ),
            WorkflowStep(
                step_id="check_rebalance_threshold",
                service="trading-engine",
                endpoint="/portfolio/check_drift",
                payload={"threshold": parameters.get("drift_threshold", 0.05)},
                depends_on=["calculate_target_allocation"]
            ),
            WorkflowStep(
                step_id="execute_rebalance",
                service="trading-engine",
                endpoint="/portfolio/rebalance",
                payload={"tax_optimize": parameters.get("tax_optimize", True)},
                depends_on=["check_rebalance_threshold"]
            )
        ]
        return steps
    
    async def execute_workflow(self, workflow_id: str):
        """Execute workflow with saga pattern"""
        try:
            if workflow_id not in self.workflows:
                logger.error(f"Workflow {workflow_id} not found")
                return
            
            workflow = self.workflows[workflow_id]
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.utcnow()
            
            logger.info(f"Starting workflow execution: {workflow_id}")
            
            # Track completed steps for dependency resolution
            completed_steps = set()
            
            # Execute steps in dependency order
            while True:
                # Find next executable steps
                ready_steps = []
                for step in workflow.steps:
                    if (step.status == WorkflowStatus.PENDING and 
                        (not step.depends_on or all(dep in completed_steps for dep in step.depends_on))):
                        ready_steps.append(step)
                
                if not ready_steps:
                    # Check if all steps completed
                    if all(step.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] for step in workflow.steps):
                        break
                    else:
                        # Wait for running steps to complete
                        await asyncio.sleep(1)
                        continue
                
                # Execute ready steps in parallel
                tasks = []
                for step in ready_steps:
                    tasks.append(self.execute_step(workflow_id, step))
                
                # Wait for current batch to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    step = ready_steps[i]
                    if isinstance(result, Exception):
                        step.status = WorkflowStatus.FAILED
                        step.error = str(result)
                        logger.error(f"Step {step.step_id} failed: {result}")
                    else:
                        step.status = WorkflowStatus.COMPLETED
                        step.result = result
                        completed_steps.add(step.step_id)
                        logger.info(f"Step {step.step_id} completed successfully")
            
            # Determine final workflow status
            failed_steps = [step for step in workflow.steps if step.status == WorkflowStatus.FAILED]
            if failed_steps:
                workflow.status = WorkflowStatus.FAILED
                workflow.error = f"Failed steps: {[step.step_id for step in failed_steps]}"
                
                # Trigger compensation if needed
                await self.compensate_workflow(workflow)
            else:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.result = {"completed_steps": len(completed_steps)}
            
            workflow.completed_at = datetime.utcnow()
            
            # Update database
            await self.update_workflow_in_db(workflow)
            
            # Publish completion event
            await self.publish_workflow_event(workflow, "workflow_completed")
            
            logger.info(f"Workflow {workflow_id} completed with status: {workflow.status}")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow = self.workflows.get(workflow_id)
            if workflow:
                workflow.status = WorkflowStatus.FAILED
                workflow.error = str(e)
                workflow.completed_at = datetime.utcnow()
    
    async def execute_step(self, workflow_id: str, step: WorkflowStep) -> Dict[str, Any]:
        """Execute individual workflow step"""
        try:
            step.status = WorkflowStatus.RUNNING
            step.started_at = datetime.utcnow()
            
            # Check circuit breaker
            if step.service != "orchestrator":
                if await self.is_circuit_breaker_open(step.service):
                    raise Exception(f"Circuit breaker open for {step.service}")
            
            # Execute step
            if step.service == "orchestrator":
                # Internal orchestrator operation
                result = await self.execute_internal_step(step)
            else:
                # External service call
                result = await self.call_service(step.service, step.endpoint, step.payload, step.timeout)
            
            step.completed_at = datetime.utcnow()
            return result
            
        except Exception as e:
            step.error = str(e)
            await self.handle_step_failure(workflow_id, step)
            raise
    
    async def execute_internal_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute internal orchestrator operations"""
        try:
            if step.endpoint == "/internal/market_status":
                return await self.check_market_status()
            elif step.endpoint == "/internal/health_check_all":
                return await self.health_check_all_services()
            elif step.endpoint == "/internal/generate_signals":
                return await self.generate_trading_signals(step.payload)
            elif step.endpoint == "/internal/calculate_allocation":
                return await self.calculate_portfolio_allocation(step.payload)
            else:
                raise ValueError(f"Unknown internal endpoint: {step.endpoint}")
                
        except Exception as e:
            logger.error(f"Internal step execution failed: {e}")
            raise
    
    async def call_service(self, service: str, endpoint: str, payload: Dict, timeout: int = 300) -> Dict[str, Any]:
        """Make HTTP call to microservice"""
        try:
            if service not in self.service_registry:
                raise ValueError(f"Service {service} not registered")
            
            service_info = self.service_registry[service]
            url = f"{service_info['url']}{endpoint}"
            
            start_time = time.time()
            
            async with self.session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response_time = time.time() - start_time
                
                # Update service metrics
                service_info["response_time"] = response_time
                service_info["last_check"] = datetime.utcnow()
                
                if response.status == 200:
                    service_info["status"] = ServiceStatus.HEALTHY
                    service_info["error_count"] = 0
                    await self.close_circuit_breaker(service)
                    
                    result = await response.json()
                    return result
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            # Update error metrics
            if service in self.service_registry:
                self.service_registry[service]["error_count"] += 1
                self.service_registry[service]["status"] = ServiceStatus.DEGRADED
            
            await self.handle_service_error(service, str(e))
            raise
    
    async def handle_step_failure(self, workflow_id: str, step: WorkflowStep):
        """Handle step failure with retry logic"""
        try:
            step.retry_count += 1
            
            if step.retry_count <= step.max_retries:
                logger.info(f"Retrying step {step.step_id}, attempt {step.retry_count}")
                
                # Exponential backoff
                delay = min(2 ** step.retry_count, 60)
                await asyncio.sleep(delay)
                
                # Reset status for retry
                step.status = WorkflowStatus.PENDING
                step.error = None
            else:
                logger.error(f"Step {step.step_id} failed after {step.max_retries} retries")
                step.status = WorkflowStatus.FAILED
                
        except Exception as e:
            logger.error(f"Error handling step failure: {e}")
    
    async def compensate_workflow(self, workflow: Workflow):
        """Execute compensation actions for failed workflow"""
        try:
            logger.info(f"Starting compensation for workflow {workflow.workflow_id}")
            
            # Define compensation actions based on workflow type
            if workflow.workflow_type == "daily_trading_pipeline":
                # Cancel any pending orders
                await self.call_service("trading-engine", "/orders/cancel_all", {})
                
            elif workflow.workflow_type == "model_retraining":
                # Rollback to previous model
                await self.call_service("tft-predictor", "/model/rollback", {})
                
            # Add more compensation logic as needed
            
        except Exception as e:
            logger.error(f"Compensation failed: {e}")
    
    async def is_circuit_breaker_open(self, service: str) -> bool:
        """Check if circuit breaker is open for service"""
        if service not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[service]
        
        if breaker["state"] == "open":
            # Check if recovery timeout has passed
            if (breaker["last_failure"] and 
                datetime.utcnow() - breaker["last_failure"] > timedelta(seconds=breaker["recovery_timeout"])):
                breaker["state"] = "half-open"
                logger.info(f"Circuit breaker for {service} moved to half-open")
                return False
            return True
        
        return False
    
    async def handle_service_error(self, service: str, error: str):
        """Handle service error and update circuit breaker"""
        if service not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[service]
        breaker["failure_count"] += 1
        breaker["last_failure"] = datetime.utcnow()
        
        # Open circuit breaker if threshold exceeded
        if breaker["failure_count"] >= breaker["failure_threshold"]:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for {service} after {breaker['failure_count']} failures")
            
            # Publish alert
            await self.publish_system_alert(f"Circuit breaker opened for {service}", "high", {
                "service": service,
                "error_count": breaker["failure_count"],
                "last_error": error
            })
    
    async def close_circuit_breaker(self, service: str):
        """Close circuit breaker after successful call"""
        if service in self.circuit_breakers:
            breaker = self.circuit_breakers[service]
            if breaker["state"] in ["half-open", "open"]:
                breaker["state"] = "closed"
                breaker["failure_count"] = 0
                logger.info(f"Circuit breaker closed for {service}")
    
    async def check_market_status(self) -> Dict[str, Any]:
        """Check if market is open"""
        try:
            now = datetime.now()
            weekday = now.weekday()
            current_time = now.strftime("%H:%M")
            
            is_trading_day = weekday in self.market_schedule["trading_days"]
            is_trading_hours = (self.market_schedule["market_open"] <= current_time <= self.market_schedule["market_close"])
            is_market_open = is_trading_day and is_trading_hours
            
            return {
                "is_open": is_market_open,
                "is_trading_day": is_trading_day,
                "current_time": current_time,
                "market_open": self.market_schedule["market_open"],
                "market_close": self.market_schedule["market_close"]
            }
            
        except Exception as e:
            logger.error(f"Market status check failed: {e}")
            return {"is_open": False, "error": str(e)}
    
    async def health_check_all_services(self) -> Dict[str, Any]:
        """Health check all registered services"""
        try:
            results = {}
            
            for service, info in self.service_registry.items():
                try:
                    start_time = time.time()
                    async with self.session.get(f"{info['url']}/health") as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            info["status"] = ServiceStatus.HEALTHY
                            info["response_time"] = response_time
                            results[service] = {"status": "healthy", "response_time": response_time}
                        else:
                            info["status"] = ServiceStatus.DEGRADED
                            results[service] = {"status": "degraded", "response_code": response.status}
                            
                except Exception as e:
                    info["status"] = ServiceStatus.DOWN
                    results[service] = {"status": "down", "error": str(e)}
            
            return {"services": results, "timestamp": datetime.utcnow().isoformat()}
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"error": str(e)}
    
    async def generate_trading_signals(self, payload: Dict) -> Dict[str, Any]:
        """Generate trading signals based on predictions and sentiment"""
        try:
            strategy = payload.get("strategy", "sentiment_momentum")
            
            # Get latest predictions from cache
            predictions_key = "latest_predictions"
            predictions_data = await self.redis_client.get(predictions_key)
            
            if not predictions_data:
                raise Exception("No predictions available")
            
            predictions = json.loads(predictions_data)
            
            # Generate signals based on strategy
            signals = []
            for ticker, pred in predictions.items():
                signal = self.calculate_signal(ticker, pred, strategy)
                if signal:
                    signals.append(signal)
            
            # Publish signals to Kafka
            for signal in signals:
                self.kafka_producer.send(KAFKA_TOPICS["trading_signals"], signal)
            
            return {
                "signals_generated": len(signals),
                "strategy": strategy,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise
    
    def calculate_signal(self, ticker: str, prediction_data: Dict, strategy: str) -> Optional[Dict]:
        """Calculate trading signal for ticker"""
        try:
            if strategy == "sentiment_momentum":
                # Simple sentiment momentum strategy
                sentiment_score = prediction_data.get("sentiment_score", 0)
                predicted_return = prediction_data.get("predicted_return", 0)
                confidence = prediction_data.get("confidence", 0)
                
                if confidence < 0.6:
                    return None
                
                if sentiment_score > 0.6 and predicted_return > 0.02:
                    action = "STRONG_BUY"
                    allocation = 0.08
                elif sentiment_score > 0.3 and predicted_return > 0.01:
                    action = "BUY"
                    allocation = 0.05
                elif sentiment_score < -0.3 and predicted_return < -0.01:
                    action = "SELL"
                    allocation = 0.05
                elif sentiment_score < -0.6 and predicted_return < -0.02:
                    action = "STRONG_SELL"
                    allocation = 0.08
                else:
                    return None
                
                return {
                    "ticker": ticker,
                    "action": action,
                    "allocation": allocation,
                    "confidence": confidence,
                    "sentiment_score": sentiment_score,
                    "predicted_return": predicted_return,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Signal calculation failed for {ticker}: {e}")
            return None
    
    async def calculate_portfolio_allocation(self, payload: Dict) -> Dict[str, Any]:
        """Calculate optimal portfolio allocation"""
        try:
            strategy = payload.get("strategy", "risk_parity")
            
            # Simplified allocation calculation
            if strategy == "risk_parity":
                # Equal risk contribution
                allocations = {
                    "AAPL": 0.15,
                    "GOOGL": 0.15,
                    "MSFT": 0.15,
                    "TSLA": 0.10,
                    "NVDA": 0.10,
                    "AMZN": 0.10,
                    "META": 0.10,
                    "CASH": 0.15
                }
            else:
                # Default equal weight
                allocations = {"CASH": 0.2}
                tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
                weight = 0.8 / len(tickers)
                for ticker in tickers:
                    allocations[ticker] = weight
            
            return {
                "strategy": strategy,
                "allocations": allocations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio allocation calculation failed: {e}")
            raise
    
    async def service_health_monitor(self):
        """Monitor service health continuously"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for service, info in self.service_registry.items():
                    try:
                        start_time = time.time()
                        async with self.session.get(f"{info['url']}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                            response_time = time.time() - start_time
                            
                            if response.status == 200:
                                if info["status"] != ServiceStatus.HEALTHY:
                                    logger.info(f"Service {service} recovered")
                                    await self.close_circuit_breaker(service)
                                
                                info["status"] = ServiceStatus.HEALTHY
                                info["response_time"] = response_time
                                info["last_check"] = datetime.utcnow()
                            else:
                                info["status"] = ServiceStatus.DEGRADED
                                
                    except Exception as e:
                        if info["status"] != ServiceStatus.DOWN:
                            logger.warning(f"Service {service} is down: {e}")
                            await self.publish_system_alert(f"Service {service} is down", "high", {"service": service, "error": str(e)})
                        
                        info["status"] = ServiceStatus.DOWN
                        info["last_check"] = datetime.utcnow()
                        await self.handle_service_error(service, str(e))
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def workflow_processor(self):
        """Process scheduled workflows"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check for scheduled workflows
                current_time = datetime.utcnow()
                
                for workflow_id, workflow in list(self.workflows.items()):
                    if (workflow.status == WorkflowStatus.PENDING and 
                        hasattr(workflow, 'schedule_time') and 
                        workflow.schedule_time and 
                        workflow.schedule_time <= current_time):
                        
                        logger.info(f"Executing scheduled workflow: {workflow_id}")
                        asyncio.create_task(self.execute_workflow(workflow_id))
                
            except Exception as e:
                logger.error(f"Workflow processor error: {e}")
    
    async def event_consumer(self):
        """Consume system events from Kafka"""
        dlq = None
        try:
            dlq = DeadLetterQueue(db_url=DATABASE_URL, service_name="orchestrator")
        except Exception as e:
            logger.warning("DLQ init failed: %s", e)

        try:
            consumer = KafkaConsumer(
                KAFKA_TOPICS["system_events"],
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id="orchestrator-events-group",
                enable_auto_commit=False,
            )

            for message in consumer:
                try:
                    event = message.value
                    await self.process_system_event(event)
                    self.kafka_producer.flush()
                    consumer.commit()

                except Exception as e:
                    logger.error(f"Error processing system event: {e}")
                    if dlq:
                        dlq.persist(
                            topic=message.topic,
                            key=message.key.decode() if message.key else None,
                            value=event,
                            error=str(e),
                        )
                    consumer.commit()

        except Exception as e:
            logger.error(f"Event consumer error: {e}")
    
    async def process_system_event(self, event: Dict):
        """Process incoming system events"""
        try:
            event_type = event.get("type")
            
            if event_type == "market_open":
                # Trigger market open workflow
                await self.create_workflow(WorkflowRequest(
                    workflow_type="market_open_procedure",
                    parameters={}
                ))
            elif event_type == "model_accuracy_degraded":
                # Trigger model retraining
                await self.create_workflow(WorkflowRequest(
                    workflow_type="model_retraining",
                    parameters={"force_retrain": True}
                ))
            elif event_type == "high_volatility_detected":
                # Adjust risk parameters
                await self.adjust_risk_parameters("conservative")
                
        except Exception as e:
            logger.error(f"System event processing failed: {e}")
    
    async def market_scheduler(self):
        """Schedule market-related workflows"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = datetime.now()
                current_time = now.strftime("%H:%M")
                
                # Market open trigger
                if current_time == self.market_schedule["market_open"]:
                    if now.weekday() in self.market_schedule["trading_days"]:
                        logger.info("Market opening - triggering daily pipeline")
                        await self.create_workflow(WorkflowRequest(
                            workflow_type="daily_trading_pipeline",
                            parameters={"tickers": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]}
                        ))
                
                # Market close trigger
                elif current_time == self.market_schedule["market_close"]:
                    if now.weekday() in self.market_schedule["trading_days"]:
                        logger.info("Market closing - triggering end of day procedures")
                        await self.create_workflow(WorkflowRequest(
                            workflow_type="end_of_day_procedure",
                            parameters={}
                        ))
                
            except Exception as e:
                logger.error(f"Market scheduler error: {e}")
    
    async def adjust_risk_parameters(self, risk_level: str):
        """Adjust system risk parameters"""
        try:
            if risk_level == "conservative":
                # Reduce position sizes and increase cash allocation
                risk_params = {
                    "max_position_size": 0.03,
                    "max_sector_allocation": 0.20,
                    "min_cash_reserve": 0.20
                }
            elif risk_level == "aggressive":
                risk_params = {
                    "max_position_size": 0.08,
                    "max_sector_allocation": 0.35,
                    "min_cash_reserve": 0.05
                }
            else:  # moderate
                risk_params = {
                    "max_position_size": 0.05,
                    "max_sector_allocation": 0.25,
                    "min_cash_reserve": 0.10
                }
            
            # Update risk parameters in trading engine
            await self.call_service("trading-engine", "/risk/update_parameters", risk_params)
            
            logger.info(f"Updated risk parameters to {risk_level} level")
            
        except Exception as e:
            logger.error(f"Risk parameter adjustment failed: {e}")
    
    async def publish_workflow_event(self, workflow: Workflow, event_type: str):
        """Publish workflow event to Kafka"""
        try:
            message = {
                "event_type": event_type,
                "workflow_id": workflow.workflow_id,
                "workflow_type": workflow.workflow_type,
                "status": workflow.status.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.kafka_producer.send(KAFKA_TOPICS["workflow_events"], message)
            
        except Exception as e:
            logger.error(f"Failed to publish workflow event: {e}")
    
    async def publish_system_alert(self, message: str, severity: str, details: Dict):
        """Publish system alert"""
        try:
            alert = {
                "type": "system_alert",
                "message": message,
                "severity": severity,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.kafka_producer.send(KAFKA_TOPICS["system_events"], alert)
            logger.warning(f"System alert: {message}")
            
        except Exception as e:
            logger.error(f"Failed to publish system alert: {e}")
    
    async def store_workflow_in_db(self, workflow: Workflow):
        """Store workflow in database"""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                INSERT INTO workflows (
                    workflow_id, workflow_type, status, parameters, created_at
                ) VALUES (
                    :workflow_id, :workflow_type, :status, :parameters, :created_at
                )
                """)
                
                conn.execute(query, {
                    'workflow_id': workflow.workflow_id,
                    'workflow_type': workflow.workflow_type,
                    'status': workflow.status.value,
                    'parameters': json.dumps(workflow.parameters),
                    'created_at': workflow.created_at
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store workflow in database: {e}")
    
    async def update_workflow_in_db(self, workflow: Workflow):
        """Update workflow in database"""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                UPDATE workflows SET 
                    status = :status,
                    result = :result,
                    error = :error,
                    completed_at = :completed_at
                WHERE workflow_id = :workflow_id
                """)
                
                conn.execute(query, {
                    'workflow_id': workflow.workflow_id,
                    'status': workflow.status.value,
                    'result': json.dumps(workflow.result),
                    'error': workflow.error,
                    'completed_at': workflow.completed_at
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update workflow in database: {e}")
    
    async def schedule_workflow(self, workflow_id: str, schedule_time: datetime):
        """Schedule workflow for future execution"""
        try:
            # Store scheduled time
            workflow = self.workflows[workflow_id]
            workflow.schedule_time = schedule_time
            
            logger.info(f"Scheduled workflow {workflow_id} for {schedule_time}")
            
        except Exception as e:
            logger.error(f"Workflow scheduling failed: {e}")
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get workflow status"""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "parameters": workflow.parameters,
            "result": workflow.result,
            "error": workflow.error,
            "steps": [
                {
                    "step_id": step.step_id,
                    "service": step.service,
                    "status": step.status.value,
                    "retry_count": step.retry_count,
                    "error": step.error
                }
                for step in workflow.steps
            ]
        }
    
    async def get_service_registry(self) -> Dict:
        """Get service registry status"""
        return {
            service: {
                "url": info["url"],
                "status": info["status"].value,
                "response_time": info["response_time"],
                "last_check": info["last_check"].isoformat() if info["last_check"] else None,
                "error_count": info["error_count"],
                "circuit_breaker_open": info["circuit_breaker_open"]
            }
            for service, info in self.service_registry.items()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.session:
            await self.session.close()

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global orchestrator
    orchestrator = Orchestrator()
    await orchestrator.initialize()
    # Start DLQ retry worker
    try:
        _dlq = DeadLetterQueue(db_url=DATABASE_URL, service_name="orchestrator")

        def _reprocess(topic, key, value):
            import asyncio as _aio
            loop = _aio.new_event_loop()
            try:
                loop.run_until_complete(orchestrator.process_system_event(value))
            finally:
                loop.close()

        start_retry_worker(_dlq, _reprocess)
        logger.info("DLQ retry worker started for orchestrator")
    except Exception as e:
        logger.warning("DLQ retry worker not started: %s", e)
    yield
    # Shutdown
    await orchestrator.cleanup()

app = FastAPI(title="Orchestrator Service", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "orchestrator", "timestamp": datetime.utcnow().isoformat()}

@app.post("/workflows")
async def create_workflow_endpoint(request: WorkflowRequest):
    """Create new workflow"""
    try:
        workflow_id = await orchestrator.create_workflow(request)
        return {"workflow_id": workflow_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get workflow status"""
    try:
        status = await orchestrator.get_workflow_status(workflow_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/services")
async def get_services():
    """Get service registry"""
    try:
        services = await orchestrator.get_service_registry()
        return {"services": services}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/services/health")
async def health_check_all():
    """Health check all services"""
    try:
        result = await orchestrator.health_check_all_services()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/status")
async def market_status():
    """Get market status"""
    try:
        status = await orchestrator.check_market_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
