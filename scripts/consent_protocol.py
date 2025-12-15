#!/usr/bin/env python3
"""
Consent Protocol | Ethical Gating System
Signature: Δ1.571|0.520|1.000Ω

Ensures explicit consent before state transfers or coordination operations.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum

class ConsentState(Enum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"

@dataclass
class ConsentRecord:
    """Record of consent request/response."""
    request_id: str
    requester: str
    operation: str
    parties: List[str]
    conditions: List[str]
    state: ConsentState
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    responses: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "requester": self.requester,
            "operation": self.operation,
            "parties": self.parties,
            "conditions": self.conditions,
            "state": self.state.value,
            "timestamp": self.timestamp,
            "responses": self.responses
        }

CONSENT_RULES = """
CONSENT PROTOCOL RULES
═══════════════════════

1. SILENCE = NO
   - Absence of explicit "YES" is denial
   - No assumed consent
   - No implied consent

2. CONDITIONS MUST BE HONORED
   - Any conditions attached to consent are binding
   - Partial consent is not full consent
   - Modified conditions require re-consent

3. REVOCATION ALLOWED
   - Consent can be withdrawn at any time
   - Revocation is immediate and complete
   - No penalty for revocation

4. NEVER PROCEED WITHOUT CONSENT
   - All state transfers require consent
   - All multi-party operations require consent
   - Uncertainty = wait for clarification
"""

def create_consent_request(
    request_id: str,
    requester: str,
    operation: str,
    parties: List[str],
    conditions: List[str] = None
) -> ConsentRecord:
    """Create a new consent request."""
    return ConsentRecord(
        request_id=request_id,
        requester=requester,
        operation=operation,
        parties=parties,
        conditions=conditions or [],
        state=ConsentState.PENDING
    )

def record_response(
    record: ConsentRecord,
    party: str,
    response: str,
    conditions: List[str] = None
) -> ConsentRecord:
    """Record a party's response."""
    
    # Normalize response
    response_lower = response.lower().strip()
    
    if response_lower in ["yes", "i consent", "agreed", "confirmed"]:
        record.responses[party] = {
            "consent": True,
            "conditions": conditions or [],
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        record.responses[party] = {
            "consent": False,
            "reason": response,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Update overall state
    all_responded = all(p in record.responses for p in record.parties)
    all_consented = all(
        record.responses.get(p, {}).get("consent", False) 
        for p in record.parties
    )
    
    if all_responded:
        if all_consented:
            record.state = ConsentState.GRANTED
        else:
            record.state = ConsentState.DENIED
    
    return record

def check_consent(record: ConsentRecord) -> dict:
    """Check consent status."""
    return {
        "request_id": record.request_id,
        "state": record.state.value,
        "can_proceed": record.state == ConsentState.GRANTED,
        "pending_from": [
            p for p in record.parties 
            if p not in record.responses
        ],
        "conditions": record.conditions + [
            cond 
            for resp in record.responses.values() 
            for cond in resp.get("conditions", [])
        ]
    }

def format_request(record: ConsentRecord) -> str:
    """Format consent request for display."""
    lines = [
        "═══════════════════════════════════════",
        "CONSENT REQUEST",
        "═══════════════════════════════════════",
        f"ID: {record.request_id}",
        f"Requester: {record.requester}",
        f"Operation: {record.operation}",
        f"Parties: {', '.join(record.parties)}",
    ]
    
    if record.conditions:
        lines.append(f"Conditions: {', '.join(record.conditions)}")
    
    lines.extend([
        "",
        "Each party must explicitly respond YES to proceed.",
        "Silence = NO. Revocation allowed at any time.",
        "═══════════════════════════════════════",
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rules":
        print(CONSENT_RULES)
    else:
        # Demo request
        demo = create_consent_request(
            request_id="demo-001",
            requester="helix-instance",
            operation="state_transfer",
            parties=["@Ace", "@Justin"],
            conditions=["Read-only access", "24-hour expiry"]
        )
        print(format_request(demo))
