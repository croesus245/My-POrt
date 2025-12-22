"""
Permission and Access Control

Handles:
- User authentication/authorization
- Tenant isolation
- Role-based access control (RBAC)
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Role(Enum):
    """User roles with increasing permissions"""
    VIEWER = "viewer"      # Read-only, limited docs
    USER = "user"          # Standard access
    ADMIN = "admin"        # Full access
    SUPER_ADMIN = "super"  # Cross-tenant (internal only)


@dataclass
class User:
    """User with permissions"""
    user_id: str
    tenant_id: str
    role: str
    permissions: list[str]
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions or self.role == "admin"
    
    def can_access_tenant(self, tenant_id: str) -> bool:
        """Check if user can access a tenant's data"""
        if self.role == "super":
            return True
        return self.tenant_id == tenant_id


class PermissionChecker:
    """
    Manages user permissions and access control.
    
    In production, this would integrate with:
    - OAuth/OIDC provider
    - Database for user/role storage
    - ABAC/RBAC policy engine
    """
    
    def __init__(self):
        # Demo users (in production, fetch from DB)
        self.users = {
            ("user_123", "company_a"): User(
                user_id="user_123",
                tenant_id="company_a",
                role="user",
                permissions=["read", "query"]
            ),
            ("admin_456", "company_a"): User(
                user_id="admin_456",
                tenant_id="company_a",
                role="admin",
                permissions=["read", "write", "query", "admin"]
            ),
            ("user_789", "company_b"): User(
                user_id="user_789",
                tenant_id="company_b",
                role="user",
                permissions=["read", "query"]
            ),
            ("test_user", "test_tenant"): User(
                user_id="test_user",
                tenant_id="test_tenant",
                role="user",
                permissions=["read", "query"]
            ),
        }
        
        # Role -> base permissions mapping
        self.role_permissions = {
            "viewer": ["read"],
            "user": ["read", "query"],
            "admin": ["read", "write", "query", "admin", "delete"],
            "super": ["read", "write", "query", "admin", "delete", "cross_tenant"]
        }
    
    def get_user(self, user_id: str, tenant_id: str) -> Optional[User]:
        """
        Get user by ID and tenant.
        
        Returns None if user doesn't exist or doesn't belong to tenant.
        """
        # Check explicit user
        user = self.users.get((user_id, tenant_id))
        if user:
            return user
        
        # Auto-create demo user for testing
        if user_id.startswith("demo_"):
            return User(
                user_id=user_id,
                tenant_id=tenant_id,
                role="user",
                permissions=["read", "query"]
            )
        
        return None
    
    def create_user(
        self,
        user_id: str,
        tenant_id: str,
        role: str = "user"
    ) -> User:
        """Create a new user"""
        permissions = self.role_permissions.get(role, ["read"])
        
        user = User(
            user_id=user_id,
            tenant_id=tenant_id,
            role=role,
            permissions=permissions
        )
        
        self.users[(user_id, tenant_id)] = user
        return user
    
    def check_document_access(
        self,
        user: User,
        doc_metadata: dict
    ) -> bool:
        """
        Check if user can access a specific document.
        
        Checks:
        1. Tenant match (critical)
        2. Access level requirements
        3. Specific permissions
        """
        # Tenant isolation (non-negotiable)
        doc_tenant = doc_metadata.get("tenant_id")
        if doc_tenant and not user.can_access_tenant(doc_tenant):
            return False
        
        # Access level check
        doc_access_level = doc_metadata.get("access_level", "user")
        
        access_hierarchy = {
            "viewer": 0,
            "user": 1,
            "admin": 2,
            "super": 3
        }
        
        user_level = access_hierarchy.get(user.role, 0)
        required_level = access_hierarchy.get(doc_access_level, 1)
        
        if user_level < required_level:
            return False
        
        # Check specific permission requirements
        required_permissions = doc_metadata.get("required_permissions", [])
        for perm in required_permissions:
            if not user.has_permission(perm):
                return False
        
        return True
    
    def check_action_permission(
        self,
        user: User,
        action: str
    ) -> bool:
        """Check if user can perform an action"""
        return user.has_permission(action)


class RateLimiter:
    """
    Simple rate limiter for API protection.
    
    In production, use Redis-based distributed rate limiting.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 500
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.request_counts: dict[str, list[float]] = {}
    
    def check(self, user_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed.
        
        Returns: (allowed, reason_if_blocked)
        """
        import time
        now = time.time()
        
        if user_id not in self.request_counts:
            self.request_counts[user_id] = []
        
        # Clean old entries
        self.request_counts[user_id] = [
            t for t in self.request_counts[user_id]
            if now - t < 3600  # Keep last hour
        ]
        
        requests = self.request_counts[user_id]
        
        # Check per-minute limit
        last_minute = [t for t in requests if now - t < 60]
        if len(last_minute) >= self.rpm:
            return False, f"Rate limit exceeded: {self.rpm} requests per minute"
        
        # Check per-hour limit
        if len(requests) >= self.rph:
            return False, f"Rate limit exceeded: {self.rph} requests per hour"
        
        # Record request
        self.request_counts[user_id].append(now)
        
        return True, None
