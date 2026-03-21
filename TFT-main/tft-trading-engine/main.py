"""
🚀 TRADING ENGINE MICROSERVICE
==============================
Institutional-grade order execution and risk management service
Multi-broker integration with advanced risk controls
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from enum import Enum
import redis.asyncio as redis
from kafka import KafkaConsumer, KafkaProducer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
from contextlib import asynccontextmanager
import psycopg2
from sqlalchemy import create_engine, text
import aiohttp
import asyncio
from dataclasses import dataclass
import uuid
from decimal import Decimal, ROUND_HALF_UP
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.environ["DATABASE_URL"]
ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Risk Parameters
MAX_POSITION_SIZE = 0.05  # 5% max per position
MAX_SECTOR_ALLOCATION = 0.25  # 25% max per sector
MAX_PORTFOLIO_BETA = 1.5
MIN_CASH_RESERVE = 0.1  # 10% minimum cash

# Kafka Topics
KAFKA_TOPICS = {
    "trading_signals": "trading-signals",
    "order_updates": "order-updates",
    "portfolio_updates": "portfolio-updates",
    "risk_alerts": "risk-alerts",
    "system_health": "system-health"
}

# Enums
class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    BRACKET = "bracket"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"

# Pydantic Models
class OrderRequest(BaseModel):
    ticker: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

class BracketOrderRequest(BaseModel):
    ticker: str
    side: OrderSide
    quantity: int
    limit_price: float
    take_profit_price: float
    stop_loss_price: float
    time_in_force: TimeInForce = TimeInForce.DAY

class RiskCheckRequest(BaseModel):
    ticker: str
    side: OrderSide
    quantity: int
    price: float

@dataclass
class Position:
    ticker: str
    quantity: int
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    day_pnl: float
    side: str

@dataclass
class Order:
    id: str
    ticker: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    status: OrderStatus
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_qty: int = 0
    filled_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class TradingEngine:
    def __init__(self):
        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None
        self.db_engine = None
        self.alpaca_session = None
        self.portfolio_value = 0.0
        self.buying_power = 0.0
        self.positions = {}
        self.open_orders = {}
        self.risk_metrics = {}
        
    async def initialize(self):
        """Initialize connections and load portfolio state"""
        try:
            # Initialize connections
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                retries=3
            )
            
            self.redis_client = redis.from_url(REDIS_URL)
            self.db_engine = create_engine(DATABASE_URL)
            self.alpaca_session = aiohttp.ClientSession()
            
            # Load portfolio state
            await self.load_portfolio_state()
            await self.load_risk_parameters()
            
            # Start background tasks
            asyncio.create_task(self.monitor_orders())
            asyncio.create_task(self.update_portfolio_metrics())
            asyncio.create_task(self.consume_trading_signals())
            
            logger.info("Trading Engine service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Trading Engine: {e}")
            raise
    
    async def load_portfolio_state(self):
        """Load current portfolio positions from broker"""
        try:
            # Get account info from Alpaca
            account_data = await self.alpaca_api_call("GET", "/v2/account")
            if account_data:
                self.portfolio_value = float(account_data.get('portfolio_value', 0))
                self.buying_power = float(account_data.get('buying_power', 0))
                
            # Get current positions
            positions_data = await self.alpaca_api_call("GET", "/v2/positions")
            if positions_data:
                self.positions = {}
                for pos in positions_data:
                    ticker = pos['symbol']
                    self.positions[ticker] = Position(
                        ticker=ticker,
                        quantity=int(pos['qty']),
                        market_value=float(pos['market_value']),
                        cost_basis=float(pos['cost_basis']),
                        unrealized_pnl=float(pos['unrealized_pl']),
                        day_pnl=float(pos['unrealized_intraday_pl']),
                        side=pos['side']
                    )
            
            # Get open orders
            orders_data = await self.alpaca_api_call("GET", "/v2/orders")
            if orders_data:
                self.open_orders = {}
                for order in orders_data:
                    order_id = order['id']
                    self.open_orders[order_id] = Order(
                        id=order_id,
                        ticker=order['symbol'],
                        side=OrderSide(order['side']),
                        quantity=int(order['qty']),
                        order_type=OrderType(order['order_type']),
                        status=OrderStatus(order['status']),
                        limit_price=float(order.get('limit_price', 0)) if order.get('limit_price') else None,
                        stop_price=float(order.get('stop_price', 0)) if order.get('stop_price') else None,
                        filled_qty=int(order.get('filled_qty', 0)),
                        created_at=datetime.fromisoformat(order['created_at'].replace('Z', '+00:00'))
                    )
            
            logger.info(f"Portfolio loaded: ${self.portfolio_value:,.2f}, {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
    
    async def load_risk_parameters(self):
        """Load risk parameters from database/config"""
        try:
            # Default risk parameters
            self.risk_metrics = {
                'max_position_size': MAX_POSITION_SIZE,
                'max_sector_allocation': MAX_SECTOR_ALLOCATION,
                'max_portfolio_beta': MAX_PORTFOLIO_BETA,
                'min_cash_reserve': MIN_CASH_RESERVE,
                'daily_loss_limit': 0.05,  # 5% daily loss limit
                'position_concentration': 0.1,  # 10% max in single position
                'sector_limits': {
                    'technology': 0.3,
                    'healthcare': 0.2,
                    'financials': 0.2,
                    'consumer': 0.15,
                    'other': 0.15
                }
            }
            
            # Cache risk parameters
            await self.redis_client.set("risk_params", json.dumps(self.risk_metrics), ex=3600)
            
        except Exception as e:
            logger.error(f"Risk parameters initialization failed: {e}")
    
    async def alpaca_api_call(self, method: str, endpoint: str, data: Dict = None) -> Optional[Dict]:
        """Make authenticated API call to Alpaca"""
        try:
            url = f"{ALPACA_BASE_URL}{endpoint}"
            headers = {
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
                "Content-Type": "application/json"
            }
            
            async with self.alpaca_session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if data else None
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Alpaca API error: {response.status} - {await response.text()}")
                    return None
                    
        except Exception as e:
            logger.error(f"Alpaca API call failed: {e}")
            return None
    
    async def place_order(self, request: OrderRequest) -> Dict:
        """Place order with comprehensive risk checks"""
        try:
            # Pre-trade risk validation
            risk_check = await self.validate_risk(RiskCheckRequest(
                ticker=request.ticker,
                side=request.side,
                quantity=request.quantity,
                price=request.limit_price or 0.0
            ))
            
            if not risk_check['approved']:
                return {
                    "status": "rejected",
                    "reason": risk_check['reason'],
                    "risk_violations": risk_check.get('violations', [])
                }
            
            # Prepare order data for Alpaca
            order_data = {
                "symbol": request.ticker,
                "qty": request.quantity,
                "side": request.side.value,
                "type": request.order_type.value,
                "time_in_force": request.time_in_force.value
            }
            
            if request.limit_price:
                order_data["limit_price"] = str(request.limit_price)
            if request.stop_price:
                order_data["stop_price"] = str(request.stop_price)
            
            # Submit order to broker
            response = await self.alpaca_api_call("POST", "/v2/orders", order_data)
            
            if response:
                order_id = response['id']
                
                # Create order object
                order = Order(
                    id=order_id,
                    ticker=request.ticker,
                    side=request.side,
                    quantity=request.quantity,
                    order_type=request.order_type,
                    status=OrderStatus(response['status']),
                    limit_price=request.limit_price,
                    stop_price=request.stop_price,
                    created_at=datetime.utcnow()
                )
                
                # Store in local cache
                self.open_orders[order_id] = order
                
                # Store in database
                await self.store_order_in_db(order)
                
                # Publish order update
                await self.publish_order_update(order, "order_placed")
                
                return {
                    "status": "success",
                    "order_id": order_id,
                    "ticker": request.ticker,
                    "side": request.side.value,
                    "quantity": request.quantity,
                    "order_type": request.order_type.value
                }
            else:
                return {"status": "error", "reason": "Failed to submit order to broker"}
                
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return {"status": "error", "reason": str(e)}
    
    async def place_bracket_order(self, request: BracketOrderRequest) -> Dict:
        """Place bracket order with take profit and stop loss"""
        try:
            # Risk validation
            risk_check = await self.validate_risk(RiskCheckRequest(
                ticker=request.ticker,
                side=request.side,
                quantity=request.quantity,
                price=request.limit_price
            ))
            
            if not risk_check['approved']:
                return {"status": "rejected", "reason": risk_check['reason']}
            
            # Bracket order data
            order_data = {
                "symbol": request.ticker,
                "qty": request.quantity,
                "side": request.side.value,
                "type": "limit",
                "time_in_force": request.time_in_force.value,
                "limit_price": str(request.limit_price),
                "order_class": "bracket",
                "take_profit": {
                    "limit_price": str(request.take_profit_price)
                },
                "stop_loss": {
                    "stop_price": str(request.stop_loss_price)
                }
            }
            
            response = await self.alpaca_api_call("POST", "/v2/orders", order_data)
            
            if response:
                return {
                    "status": "success",
                    "order_id": response['id'],
                    "legs": response.get('legs', [])
                }
            else:
                return {"status": "error", "reason": "Failed to submit bracket order"}
                
        except Exception as e:
            logger.error(f"Bracket order failed: {e}")
            return {"status": "error", "reason": str(e)}
    
    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel existing order"""
        try:
            response = await self.alpaca_api_call("DELETE", f"/v2/orders/{order_id}")
            
            if response:
                # Update local state
                if order_id in self.open_orders:
                    self.open_orders[order_id].status = OrderStatus.CANCELLED
                    await self.publish_order_update(self.open_orders[order_id], "order_cancelled")
                
                return {"status": "success", "order_id": order_id}
            else:
                return {"status": "error", "reason": "Failed to cancel order"}
                
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return {"status": "error", "reason": str(e)}
    
    async def validate_risk(self, request: RiskCheckRequest) -> Dict:
        """Comprehensive pre-trade risk validation"""
        try:
            violations = []
            
            # Position size check
            position_value = request.quantity * request.price
            position_pct = position_value / self.portfolio_value if self.portfolio_value > 0 else 1.0
            
            if position_pct > self.risk_metrics['max_position_size']:
                violations.append(f"Position size {position_pct:.2%} exceeds limit {self.risk_metrics['max_position_size']:.2%}")
            
            # Concentration check
            current_position = self.positions.get(request.ticker)
            if current_position:
                new_quantity = current_position.quantity + (request.quantity if request.side == OrderSide.BUY else -request.quantity)
                new_value = new_quantity * request.price
                concentration = new_value / self.portfolio_value if self.portfolio_value > 0 else 1.0
                
                if concentration > self.risk_metrics['position_concentration']:
                    violations.append(f"Position concentration {concentration:.2%} exceeds limit")
            
            # Buying power check
            if request.side == OrderSide.BUY:
                required_capital = request.quantity * request.price
                if required_capital > self.buying_power:
                    violations.append(f"Insufficient buying power: ${required_capital:,.2f} required, ${self.buying_power:,.2f} available")
            
            # Sector concentration check
            sector = await self.get_sector(request.ticker)
            if sector:
                current_sector_value = sum(
                    pos.market_value for pos in self.positions.values()
                    if await self.get_sector(pos.ticker) == sector
                )
                new_sector_value = current_sector_value + position_value
                sector_pct = new_sector_value / self.portfolio_value if self.portfolio_value > 0 else 1.0
                sector_limit = self.risk_metrics['sector_limits'].get(sector.lower(), 0.15)
                
                if sector_pct > sector_limit:
                    violations.append(f"Sector allocation {sector_pct:.2%} exceeds {sector} limit {sector_limit:.2%}")
            
            # Daily loss limit check
            daily_pnl = sum(pos.day_pnl for pos in self.positions.values())
            daily_loss_pct = abs(daily_pnl) / self.portfolio_value if daily_pnl < 0 and self.portfolio_value > 0 else 0.0
            
            if daily_loss_pct > self.risk_metrics['daily_loss_limit']:
                violations.append(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
            
            # Cash reserve check
            if request.side == OrderSide.BUY:
                cash_after_trade = self.buying_power - (request.quantity * request.price)
                cash_reserve_pct = cash_after_trade / self.portfolio_value if self.portfolio_value > 0 else 0.0
                
                if cash_reserve_pct < self.risk_metrics['min_cash_reserve']:
                    violations.append(f"Cash reserve would fall below minimum {self.risk_metrics['min_cash_reserve']:.2%}")
            
            approved = len(violations) == 0
            
            return {
                "approved": approved,
                "reason": "Risk checks passed" if approved else "Risk violations detected",
                "violations": violations,
                "position_size_pct": position_pct,
                "buying_power_required": request.quantity * request.price if request.side == OrderSide.BUY else 0
            }
            
        except Exception as e:
            logger.error(f"Risk validation failed: {e}")
            return {
                "approved": False,
                "reason": f"Risk validation error: {str(e)}",
                "violations": ["System error during risk validation"]
            }
    
    async def get_sector(self, ticker: str) -> Optional[str]:
        """Get sector for ticker (cached)"""
        try:
            # Check cache first
            sector = await self.redis_client.get(f"sector:{ticker}")
            if sector:
                return sector.decode('utf-8')
            
            # Simplified sector mapping (in production, use financial data API)
            sector_mapping = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology',
                'TSLA': 'Consumer', 'AMZN': 'Consumer', 'META': 'Technology',
                'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
                'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
            }
            
            sector = sector_mapping.get(ticker, 'Other')
            
            # Cache for future use
            await self.redis_client.set(f"sector:{ticker}", sector, ex=86400)  # 24 hours
            
            return sector
            
        except Exception as e:
            logger.error(f"Sector lookup failed for {ticker}: {e}")
            return 'Other'
    
    async def store_order_in_db(self, order: Order):
        """Store order in database"""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                INSERT INTO trade_executions (
                    order_id, ticker, side, quantity, order_type, status,
                    limit_price, stop_price, created_at
                ) VALUES (
                    :order_id, :ticker, :side, :quantity, :order_type, :status,
                    :limit_price, :stop_price, :created_at
                )
                """)
                
                conn.execute(query, {
                    'order_id': order.id,
                    'ticker': order.ticker,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'order_type': order.order_type.value,
                    'status': order.status.value,
                    'limit_price': order.limit_price,
                    'stop_price': order.stop_price,
                    'created_at': order.created_at
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store order in database: {e}")
    
    async def publish_order_update(self, order: Order, event_type: str):
        """Publish order update to Kafka"""
        try:
            message = {
                "event_type": event_type,
                "order_id": order.id,
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "status": order.status.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.kafka_producer.send(KAFKA_TOPICS["order_updates"], message)
            logger.info(f"Published order update: {event_type} for {order.ticker}")
            
        except Exception as e:
            logger.error(f"Failed to publish order update: {e}")
    
    async def monitor_orders(self):
        """Monitor order status and update local state"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Get order updates from Alpaca
                orders_data = await self.alpaca_api_call("GET", "/v2/orders")
                if orders_data:
                    for order_data in orders_data:
                        order_id = order_data['id']
                        
                        if order_id in self.open_orders:
                            old_status = self.open_orders[order_id].status
                            new_status = OrderStatus(order_data['status'])
                            
                            # Update if status changed
                            if old_status != new_status:
                                self.open_orders[order_id].status = new_status
                                self.open_orders[order_id].filled_qty = int(order_data.get('filled_qty', 0))
                                
                                if order_data.get('filled_avg_price'):
                                    self.open_orders[order_id].filled_price = float(order_data['filled_avg_price'])
                                
                                # Publish update
                                await self.publish_order_update(
                                    self.open_orders[order_id], 
                                    f"order_{new_status.value}"
                                )
                                
                                # Remove from open orders if filled/cancelled
                                if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                                    del self.open_orders[order_id]
                
            except Exception as e:
                logger.error(f"Order monitoring error: {e}")
    
    async def update_portfolio_metrics(self):
        """Update portfolio metrics and risk calculations"""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Reload portfolio state
                await self.load_portfolio_state()
                
                # Calculate portfolio metrics
                metrics = await self.calculate_portfolio_metrics()
                
                # Cache metrics
                await self.redis_client.set("portfolio_metrics", json.dumps(metrics), ex=300)
                
                # Publish portfolio update
                await self.publish_portfolio_update(metrics)
                
            except Exception as e:
                logger.error(f"Portfolio metrics update error: {e}")
    
    async def calculate_portfolio_metrics(self) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        try:
            total_value = sum(pos.market_value for pos in self.positions.values())
            total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            daily_pnl = sum(pos.day_pnl for pos in self.positions.values())
            
            # Position concentrations
            position_concentrations = {}
            for ticker, position in self.positions.items():
                concentration = position.market_value / total_value if total_value > 0 else 0
                if concentration > 0.01:  # Only include positions >1%
                    position_concentrations[ticker] = concentration
            
            # Sector allocations
            sector_allocations = {}
            for ticker, position in self.positions.items():
                sector = await self.get_sector(ticker)
                if sector not in sector_allocations:
                    sector_allocations[sector] = 0
                sector_allocations[sector] += position.market_value
            
            # Convert to percentages
            for sector in sector_allocations:
                sector_allocations[sector] = sector_allocations[sector] / total_value if total_value > 0 else 0
            
            return {
                "portfolio_value": self.portfolio_value,
                "buying_power": self.buying_power,
                "total_positions": len(self.positions),
                "total_pnl": total_pnl,
                "daily_pnl": daily_pnl,
                "cash_percentage": self.buying_power / self.portfolio_value if self.portfolio_value > 0 else 1.0,
                "position_concentrations": position_concentrations,
                "sector_allocations": sector_allocations,
                "open_orders": len(self.open_orders),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {}
    
    async def publish_portfolio_update(self, metrics: Dict):
        """Publish portfolio metrics to Kafka"""
        try:
            message = {
                "type": "portfolio_update",
                "service": "trading-engine",
                "data": metrics
            }
            self.kafka_producer.send(KAFKA_TOPICS["portfolio_updates"], message)
            
        except Exception as e:
            logger.error(f"Failed to publish portfolio update: {e}")
    
    async def consume_trading_signals(self):
        """Consume trading signals and execute trades"""
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPICS["trading_signals"],
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id="trading-engine-group"
            )
            
            for message in consumer:
                try:
                    signal = message.value
                    await self.process_trading_signal(signal)
                    
                except Exception as e:
                    logger.error(f"Error processing trading signal: {e}")
                    
        except Exception as e:
            logger.error(f"Trading signal consumer error: {e}")
    
    async def process_trading_signal(self, signal: Dict):
        """Process incoming trading signal"""
        try:
            ticker = signal.get('ticker')
            action = signal.get('action')  # BUY, SELL, STRONG_BUY, STRONG_SELL
            confidence = signal.get('confidence', 0.5)
            allocation = signal.get('allocation', 0.05)
            
            if not ticker or not action:
                logger.warning("Invalid trading signal received")
                return
            
            # Calculate position size
            target_value = self.portfolio_value * allocation
            current_price = signal.get('current_price', 100.0)  # Would get from market data
            quantity = int(target_value / current_price)
            
            if quantity == 0:
                logger.info(f"Quantity too small for {ticker}, skipping")
                return
            
            # Determine order side
            side = OrderSide.BUY if action in ['BUY', 'STRONG_BUY'] else OrderSide.SELL
            
            # Create order request
            order_request = OrderRequest(
                ticker=ticker,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET if confidence > 0.8 else OrderType.LIMIT,
                limit_price=current_price * 1.005 if side == OrderSide.BUY else current_price * 0.995  # 0.5% slippage
            )
            
            # Execute order
            result = await self.place_order(order_request)
            logger.info(f"Signal processed for {ticker}: {result.get('status')}")
            
        except Exception as e:
            logger.error(f"Trading signal processing failed: {e}")
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        return [
            {
                "ticker": pos.ticker,
                "quantity": pos.quantity,
                "market_value": pos.market_value,
                "cost_basis": pos.cost_basis,
                "unrealized_pnl": pos.unrealized_pnl,
                "day_pnl": pos.day_pnl,
                "side": pos.side
            }
            for pos in self.positions.values()
        ]
    
    async def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """Get orders with optional status filter"""
        orders = []
        for order in self.open_orders.values():
            if status is None or order.status.value == status:
                orders.append({
                    "id": order.id,
                    "ticker": order.ticker,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "order_type": order.order_type.value,
                    "status": order.status.value,
                    "limit_price": order.limit_price,
                    "stop_price": order.stop_price,
                    "filled_qty": order.filled_qty,
                    "created_at": order.created_at.isoformat() if order.created_at else None
                })
        return orders
    
    async def get_pnl_summary(self) -> Dict:
        """Get P&L summary"""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_day_pnl = sum(pos.day_pnl for pos in self.positions.values())
        
        return {
            "portfolio_value": self.portfolio_value,
            "buying_power": self.buying_power,
            "total_unrealized_pnl": total_unrealized,
            "total_day_pnl": total_day_pnl,
            "positions_count": len(self.positions),
            "open_orders_count": len(self.open_orders),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.alpaca_session:
            await self.alpaca_session.close()

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global trading_engine
    trading_engine = TradingEngine()
    await trading_engine.initialize()
    yield
    # Shutdown
    await trading_engine.cleanup()

app = FastAPI(title="Trading Engine Service", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trading-engine", "timestamp": datetime.utcnow().isoformat()}

@app.post("/orders")
async def place_order_endpoint(request: OrderRequest):
    """Place new order"""
    try:
        result = await trading_engine.place_order(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/orders/bracket")
async def place_bracket_order_endpoint(request: BracketOrderRequest):
    """Place bracket order"""
    try:
        result = await trading_engine.place_bracket_order(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/orders/{order_id}")
async def cancel_order_endpoint(order_id: str):
    """Cancel order"""
    try:
        result = await trading_engine.cancel_order(order_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions")
async def get_positions():
    """Get current positions"""
    try:
        positions = await trading_engine.get_positions()
        return {"positions": positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders")
async def get_orders(status: Optional[str] = None):
    """Get orders"""
    try:
        orders = await trading_engine.get_orders(status)
        return {"orders": orders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk/check")
async def risk_check_endpoint(request: RiskCheckRequest):
    """Pre-trade risk validation"""
    try:
        result = await trading_engine.validate_risk(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pnl")
async def get_pnl():
    """Get P&L summary"""
    try:
        pnl = await trading_engine.get_pnl_summary()
        return pnl
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
