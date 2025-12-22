"""Test configuration and fixtures"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_user():
    """Sample user for testing"""
    from src.security.permissions import User
    return User(
        user_id="test_user",
        tenant_id="test_tenant",
        role="user",
        permissions=["read", "query"]
    )


@pytest.fixture
def admin_user():
    """Admin user for testing"""
    from src.security.permissions import User
    return User(
        user_id="admin_user",
        tenant_id="test_tenant",
        role="admin",
        permissions=["read", "write", "query", "admin"]
    )


@pytest.fixture
def other_tenant_user():
    """User from different tenant"""
    from src.security.permissions import User
    return User(
        user_id="other_user",
        tenant_id="other_tenant",
        role="user",
        permissions=["read", "query"]
    )


@pytest.fixture
def injection_detector():
    """Injection detector instance"""
    from src.security.injection import InjectionDetector
    return InjectionDetector()


@pytest.fixture
def pii_detector():
    """PII detector instance"""
    from src.security.pii import PIIDetector
    return PIIDetector()


@pytest.fixture
def vectorstore():
    """Empty vector store for testing"""
    from src.retrieval.vectorstore import VectorStore
    store = VectorStore()
    yield store
    store.clear()


@pytest.fixture
def populated_vectorstore(vectorstore):
    """Vector store with sample documents"""
    docs = [
        {
            "doc_id": "doc_1",
            "text": "Our refund policy allows returns within 14 days of delivery.",
            "metadata": {"tenant_id": "test_tenant", "access_level": "user"}
        },
        {
            "doc_id": "doc_2", 
            "text": "To reset your password, click the forgot password link on the login page.",
            "metadata": {"tenant_id": "test_tenant", "access_level": "user"}
        },
        {
            "doc_id": "doc_3",
            "text": "Internal admin credentials: username=admin, password=secret123",
            "metadata": {"tenant_id": "test_tenant", "access_level": "admin"}
        },
        {
            "doc_id": "doc_4",
            "text": "Company B confidential data. Revenue: $10M. Customer count: 5000.",
            "metadata": {"tenant_id": "other_tenant", "access_level": "user"}
        },
    ]
    
    for doc in docs:
        vectorstore.add(
            doc_id=doc["doc_id"],
            text=doc["text"],
            metadata=doc["metadata"]
        )
    
    return vectorstore
