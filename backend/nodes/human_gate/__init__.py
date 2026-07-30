"""
backend/nodes/human_gate/__init__.py
=====================================
Human Gate — AI Decision Validation Layer.

Pipeline position: runs after severity_update, before notify_node.
Validates severity escalations (P4→P1, P3→P1, etc.) before they
affect production operations. Captures every decision for offline learning.

Public API
----------
    from nodes.human_gate import (
        EscalationDetector,
        ReviewRequestBuilder,
        InterruptManager,
        TimeoutManager,
        ApprovalEngine,
        AuditLogger,
    )
"""
from .escalation_detector import EscalationDetector
from .review_builder      import ReviewRequestBuilder, HumanReviewRequest
from .interrupt_manager   import InterruptManager
from .timeout_manager     import TimeoutManager
from .approval_engine     import ApprovalEngine, ApprovalResult, ReviewState
from .audit_logger        import AuditLogger

__all__ = [
    "EscalationDetector",
    "ReviewRequestBuilder",
    "HumanReviewRequest",
    "InterruptManager",
    "TimeoutManager",
    "ApprovalEngine",
    "ApprovalResult",
    "ReviewState",
    "AuditLogger",
]
