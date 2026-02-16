"""
etcd Leader Election and Distributed Coordination
Phase 5.5: Scalability Foundation
"""

import asyncio
import uuid
from typing import Optional, Callable
from datetime import datetime
import etcd3

from infrastructure.observability import get_logger


logger = get_logger(__name__)


class LeaderElection:
    """Leader election using etcd lease mechanism"""
    
    def __init__(self, etcd_client: etcd3.Etcd3Client, election_name: str, ttl: int = 10):
        self.etcd = etcd_client
        self.election_name = election_name
        self.ttl = ttl
        self.instance_id = str(uuid.uuid4())
        self.lease = None
        self.is_leader = False
        self._running = False
        
    async def campaign(self, on_elected: Optional[Callable] = None):
        """Campaign for leadership"""
        self._running = True
        key = f"/cognitionos/leader/{self.election_name}"
        
        while self._running:
            try:
                self.lease = self.etcd.lease(self.ttl)
                success, _ = self.etcd.transaction(
                    compare=[self.etcd.transactions.version(key) == 0],
                    success=[self.etcd.transactions.put(key, self.instance_id, lease=self.lease)],
                    failure=[]
                )
                
                if success:
                    self.is_leader = True
                    logger.info(f"Elected as leader for {self.election_name}")
                    if on_elected:
                        if asyncio.iscoroutinefunction(on_elected):
                            await on_elected()
                        else:
                            on_elected()
                    await self._keep_lease_alive()
                else:
                    await asyncio.sleep(self.ttl / 2)
            except Exception as e:
                logger.error(f"Leader election error: {str(e)}")
                await asyncio.sleep(self.ttl)
    
    async def _keep_lease_alive(self):
        """Keep the lease alive while leader"""
        while self._running and self.is_leader:
            try:
                self.etcd.refresh_lease(self.lease)
                await asyncio.sleep(self.ttl / 3)
            except Exception:
                self.is_leader = False
                break


class DistributedLock:
    """Distributed lock using etcd"""
    
    def __init__(self, etcd_client: etcd3.Etcd3Client, lock_name: str, ttl: int = 30):
        self.etcd = etcd_client
        self.lock_name = lock_name
        self.ttl = ttl
        self.instance_id = str(uuid.uuid4())
        self.lease = None
        self.acquired = False
        
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
        
    async def acquire(self, timeout: Optional[int] = None) -> bool:
        """Acquire the distributed lock"""
        key = f"/cognitionos/locks/{self.lock_name}"
        start_time = datetime.utcnow()
        
        while True:
            try:
                self.lease = self.etcd.lease(self.ttl)
                success, _ = self.etcd.transaction(
                    compare=[self.etcd.transactions.version(key) == 0],
                    success=[self.etcd.transactions.put(key, self.instance_id, lease=self.lease)],
                    failure=[]
                )
                
                if success:
                    self.acquired = True
                    return True
                
                if timeout and (datetime.utcnow() - start_time).total_seconds() >= timeout:
                    return False
                
                await asyncio.sleep(0.5)
            except Exception:
                return False
    
    async def release(self):
        """Release the distributed lock"""
        if self.acquired:
            try:
                key = f"/cognitionos/locks/{self.lock_name}"
                self.etcd.delete(key)
                if self.lease:
                    self.etcd.revoke_lease(self.lease)
                self.acquired = False
            except Exception as e:
                logger.error(f"Lock release error: {str(e)}")


class ServiceDiscovery:
    """Service discovery using etcd"""
    
    def __init__(self, etcd_client: etcd3.Etcd3Client, service_name: str, service_endpoint: str, ttl: int = 30):
        self.etcd = etcd_client
        self.service_name = service_name
        self.service_endpoint = service_endpoint
        self.ttl = ttl
        self.instance_id = str(uuid.uuid4())
        self.lease = None
        self._registered = False
        
    async def register(self):
        """Register this service instance"""
        try:
            self.lease = self.etcd.lease(self.ttl)
            key = f"/cognitionos/services/{self.service_name}/{self.instance_id}"
            self.etcd.put(key, self.service_endpoint, lease=self.lease)
            self._registered = True
            logger.info(f"Service registered: {self.service_name}")
            asyncio.create_task(self._keep_registration_alive())
        except Exception as e:
            logger.error(f"Service registration error: {str(e)}")
    
    async def _keep_registration_alive(self):
        """Keep service registration alive"""
        while self._registered and self.lease:
            try:
                self.etcd.refresh_lease(self.lease)
                await asyncio.sleep(self.ttl / 3)
            except Exception:
                break
    
    def discover(self, service_name: str) -> list:
        """Discover all instances of a service"""
        try:
            prefix = f"/cognitionos/services/{service_name}/"
            results = self.etcd.get_prefix(prefix)
            return [value.decode('utf-8') for value, _ in results]
        except Exception:
            return []
