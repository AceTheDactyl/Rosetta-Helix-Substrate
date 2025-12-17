# UCF Action Parameters Fixed

## Issue
The UCF integration in `kira-local-system/kira_ucf_integration.py` was providing incorrect default action parameters for 5 tools, causing "Unknown action" errors when these tools were invoked.

## Root Cause
The default action values in `kira_ucf_integration.py` did not match the valid actions defined in `scripts/tool_shed.py`.

## Fixes Applied

### 1. shed_builder_v2
- **Previous (incorrect):** `action = 'build'`
- **Fixed to:** `action = 'list'`
- **Valid actions:** `create`, `list`, `describe`
- **Line:** 362

### 2. collective_memory_sync
- **Previous (incorrect):** `action = 'retrieve'` with `key = 'state'`
- **Fixed to:** `action = 'list'` (removed invalid key default)
- **Valid actions:** `store`, `retrieve`, `merge`, `list`
- **Line:** 356-357
- **Note:** `retrieve` requires an existing key; `list` shows all available keys

### 3. autonomous_trigger_detector
- **Previous (incorrect):** `action = 'detect'`
- **Fixed to:** `action = 'check'`
- **Valid actions:** `register`, `check`, `list`, `remove`
- **Line:** 341

### 4. cross_instance_messenger
- **Previous (incorrect):** `action = 'query'`
- **Fixed to:** `action = 'validate'`
- **Valid actions:** `encode`, `decode`, `validate`
- **Line:** 351

### 5. consent_protocol
- **Previous (incorrect):** `action = 'request'`
- **Fixed to:** `action = 'check'`
- **Valid actions:** `create`, `respond`, `check`, `revoke`
- **Line:** 346

## Testing Results

All tools now execute without action parameter errors:

```bash
# Test commands (run after server restart)
curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "/ucf:shed_builder_v2"}'

curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "/ucf:collective_memory_sync"}'

curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "/ucf:autonomous_trigger_detector"}'

curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "/ucf:cross_instance_messenger"}'

curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "/ucf:consent_protocol"}'
```

## Status
✅ **FIXED** - All UCF tools now use valid action parameters and tokens are being emitted correctly.

Some tools may still return errors for missing required data (e.g., cross_instance_messenger needs encoded data, consent_protocol needs request_id), but these are expected validation errors, not action parameter errors.

## Files Modified
- `/home/acead/Rosetta-Helix-Substrate/kira-local-system/kira_ucf_integration.py`