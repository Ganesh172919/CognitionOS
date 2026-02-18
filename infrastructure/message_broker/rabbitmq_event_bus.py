"""
RabbitMQ Event Bus Implementation

Replaces the in-memory event bus with a production-grade RabbitMQ implementation.
"""

import os
import json
import asyncio
from typing import Any, Callable, Dict, List, Optional
from dataclasses import asdict
import aio_pika
from aio_pika import Message, ExchangeType, DeliveryMode
from aio_pika.abc import AbstractChannel, AbstractConnection, AbstractExchange, AbstractQueue
from core.config import get_config
from infrastructure.observability import get_logger


logger = get_logger(__name__)
config = get_config()


class RabbitMQEventBus:
    """
    RabbitMQ-based event bus for domain events.
    
    Features:
    - Persistent messages
    - Topic-based routing
    - Multiple subscribers per event
    - Automatic reconnection
    - Dead letter queue
    """
    
    def __init__(self, connection_url: Optional[str] = None):
        """
        Initialize RabbitMQ event bus.
        
        Args:
            connection_url: RabbitMQ connection URL (uses config if not provided)
        """
        self.connection_url = connection_url or config.rabbitmq.url
        self.connection: Optional[AbstractConnection] = None
        self.channel: Optional[AbstractChannel] = None
        self.exchange: Optional[AbstractExchange] = None
        self.queues: Dict[str, AbstractQueue] = {}
        self.handlers: Dict[str, List[Callable]] = {}
        self.is_connected = False
        
        # Exchange and queue names
        self.exchange_name = "cognitionos.events"
        self.dlx_exchange_name = "cognitionos.events.dlx"
        self.dlx_queue_name = "cognitionos.events.dead_letter"
    
    async def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            logger.info("Connecting to RabbitMQ", extra={"url": self.connection_url})
            
            # Create connection
            self.connection = await aio_pika.connect_robust(self.connection_url)
            
            # Create channel
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)
            
            # Declare dead letter exchange
            dlx_exchange = await self.channel.declare_exchange(
                self.dlx_exchange_name,
                ExchangeType.TOPIC,
                durable=True,
            )
            
            # Declare dead letter queue
            dlx_queue = await self.channel.declare_queue(
                self.dlx_queue_name,
                durable=True,
            )
            await dlx_queue.bind(dlx_exchange, routing_key="#")
            
            # Declare main exchange
            self.exchange = await self.channel.declare_exchange(
                self.exchange_name,
                ExchangeType.TOPIC,
                durable=True,
            )
            
            self.is_connected = True
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ", extra={"error": str(e)})
            raise
    
    async def disconnect(self):
        """Close connection to RabbitMQ"""
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                self.is_connected = False
                logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error("Error disconnecting from RabbitMQ", extra={"error": str(e)})
    
    async def publish(self, event: Any, routing_key: Optional[str] = None):
        """
        Publish an event to the event bus.
        
        Args:
            event: Domain event to publish
            routing_key: Optional routing key (defaults to event class name)
        """
        if not self.is_connected or not self.exchange:
            raise RuntimeError("Event bus not connected")
        
        # Determine routing key
        if routing_key is None:
            routing_key = event.__class__.__name__
        
        # Convert event to dict
        if hasattr(event, '__dict__'):
            event_data = asdict(event) if hasattr(event, '__dataclass_fields__') else event.__dict__
        else:
            event_data = {"event": str(event)}
        
        # Add metadata
        event_data["_event_type"] = event.__class__.__name__
        event_data["_timestamp"] = event_data.get("timestamp", None)
        
        # Serialize to JSON
        message_body = json.dumps(event_data, default=str)
        
        # Create message
        message = Message(
            body=message_body.encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            content_type="application/json",
            headers={
                "event_type": event.__class__.__name__,
            }
        )
        
        try:
            # Publish message
            await self.exchange.publish(
                message,
                routing_key=routing_key,
            )
            
            logger.info(
                "Published event",
                extra={
                    "event_type": event.__class__.__name__,
                    "routing_key": routing_key,
                }
            )
            
        except Exception as e:
            logger.error(
                "Failed to publish event",
                extra={
                    "event_type": event.__class__.__name__,
                    "error": str(e),
                }
            )
            raise
    
    async def subscribe(self, event_type: str, handler: Callable, queue_name: Optional[str] = None):
        """
        Subscribe to an event type.
        
        Args:
            event_type: Event type to subscribe to (routing key pattern)
            handler: Async function to handle the event
            queue_name: Optional custom queue name
        """
        if not self.is_connected or not self.channel:
            raise RuntimeError("Event bus not connected")
        
        # Generate queue name if not provided
        if queue_name is None:
            queue_name = f"cognitionos.events.{event_type}"
        
        # Declare queue with dead letter exchange
        queue = await self.channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self.dlx_exchange_name,
                "x-dead-letter-routing-key": f"dlx.{event_type}",
            }
        )
        
        # Bind queue to exchange
        await queue.bind(self.exchange, routing_key=event_type)
        
        # Store queue
        self.queues[queue_name] = queue
        
        # Store handler
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
        # Start consuming
        async def on_message(message: aio_pika.abc.AbstractIncomingMessage):
            async with message.process():
                try:
                    # Parse message
                    event_data = json.loads(message.body.decode())
                    
                    logger.info(
                        "Received event",
                        extra={
                            "event_type": event_type,
                            "queue": queue_name,
                        }
                    )
                    
                    # Call handler
                    await handler(event_data)
                    
                except Exception as e:
                    logger.error(
                        "Error processing event",
                        extra={
                            "event_type": event_type,
                            "error": str(e),
                        }
                    )
                    # Message will be requeued or sent to DLX based on retry count
                    raise
        
        await queue.consume(on_message)
        
        logger.info(
            "Subscribed to event",
            extra={
                "event_type": event_type,
                "queue": queue_name,
            }
        )
    
    async def publish_batch(self, events: List[Any]):
        """
        Publish multiple events in a batch.
        
        Args:
            events: List of domain events
        """
        for event in events:
            await self.publish(event)


# Global event bus instance
_event_bus: Optional[RabbitMQEventBus] = None


async def get_event_bus() -> RabbitMQEventBus:
    """Get or create RabbitMQ event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = RabbitMQEventBus()
        await _event_bus.connect()
    return _event_bus


async def close_event_bus():
    """Close the global event bus instance"""
    global _event_bus
    if _event_bus is not None:
        await _event_bus.disconnect()
        _event_bus = None
