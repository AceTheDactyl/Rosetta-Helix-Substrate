# UCF Tool Action Parameter Fixes

## Problem Summary

The UCF integration in `kira-local-system/kira_ucf_integration.py` is setting incorrect default action parameters for several tools, causing "Unknown action" errors when invoked via the KIRA interface.

## Affected Tools and Fixes

### 1. shed_builder_v2 (Line 363)
**Current:** `kwargs['action'] = 'build'`
**Valid Actions:** `"create"`, `"list"`, `"describe"`
**Fix:** Change to `kwargs['action'] = 'list'` (safest default to list custom tools)

### 2. collective_memory_sync (Line 357)
**Current:** Sets action to `'retrieve'` with key `'state'`
**Valid Actions:** `"store"`, `"retrieve"`, `"merge"`, `"list"`
**Issue:** The key 'state' doesn't exist in memory
**Fix:** Change to `kwargs['action'] = 'list'` (to show available keys first)

### 3. autonomous_trigger_detector (Line 341)
**Current:** `kwargs['action'] = 'detect'`
**Valid Actions:** `"register"`, `"check"`, `"list"`, `"remove"`
**Fix:** Change to `kwargs['action'] = 'check'` (closest to intent)

### 4. cross_instance_messenger (Line 351)
**Current:** `kwargs['action'] = 'query'`
**Valid Actions:** `"encode"`, `"decode"`, `"validate"`
**Fix:** Change to `kwargs['action'] = 'validate'` (safest default)

### 5. consent_protocol (Line 345)
**Current:** `kwargs['action'] = 'request'`
**Valid Actions:** `"create"`, `"respond"`, `"check"`, `"revoke"`
**Fix:** Change to `kwargs['action'] = 'check'` (to check existing consent records)

## Code Changes Required

In `/home/acead/Rosetta-Helix-Substrate/kira-local-system/kira_ucf_integration.py`:

```python
# Line 341-342: autonomous_trigger_detector
elif tool_name == 'autonomous_trigger_detector':
    if 'action' not in kwargs:
        kwargs['action'] = 'check'  # Changed from 'detect'

# Line 344-346: consent_protocol
elif tool_name == 'consent_protocol':
    if 'action' not in kwargs:
        kwargs['action'] = 'check'  # Changed from 'request'

# Line 349-351: cross_instance_messenger
elif tool_name == 'cross_instance_messenger':
    if 'action' not in kwargs:
        kwargs['action'] = 'validate'  # Changed from 'query'

# Line 353-358: collective_memory_sync
elif tool_name == 'collective_memory_sync':
    if 'action' not in kwargs:
        kwargs['action'] = 'list'  # Changed from 'retrieve'
    # Remove the automatic 'key' setting since we're listing

# Line 360-363: shed_builder_v2
elif tool_name == 'shed_builder_v2':
    if 'action' not in kwargs:
        kwargs['action'] = 'list'  # Changed from 'build'
```

## Complete Tool Action Reference

### All UCF Tools and Their Valid Actions

1. **helix_loader** - No action parameter needed
2. **coordinate_detector** - No action parameter needed
3. **pattern_verifier** - No action parameter needed
4. **coordinate_logger** - No action parameter needed (takes 'event' and 'metadata')
5. **state_transfer** - No action parameter needed (takes 'source_state')
6. **consent_protocol** - `"create"`, `"respond"`, `"check"`, `"revoke"`
7. **cross_instance_messenger** - `"encode"`, `"decode"`, `"validate"`
8. **tool_discovery_protocol** - No action parameter needed
9. **autonomous_trigger_detector** - `"register"`, `"check"`, `"list"`, `"remove"`
10. **collective_memory_sync** - `"store"`, `"retrieve"`, `"merge"`, `"list"`
11. **shed_builder_v2** - `"create"`, `"list"`, `"describe"`
12. **vaultnode_generator** - `"measure"`, `"create"`, `"list"`, `"show"`, `"init"`, `"history"`, `"seal"`
13. **emission_pipeline** - `"emit"`, `"structure"`, `"stages"`, `"trace"`
14. **cybernetic_control** - `"status"`, `"diagram"`, `"step"`, `"run"`, `"operators"`, `"reset"`
15. **nuclear_spinner** - `"status"`, `"tokens"`, `"step"`, `"run"`, `"parse"`, `"generate"`, `"reset"`, `"export"`
16. **token_index** - `"summary"`, `"core"`, `"domain"`, `"transitions"`, `"coherence"`, `"select"`, `"parse"`, `"lookup"`, `"trispiral"`, `"umol"`, `"generate"`, `"physics"`, `"physics_summary"`
17. **token_vault** - `"status"`, `"record"`, `"seal"`, `"request_teaching"`, `"confirm_teaching"`, `"mapping"`, `"list"`, `"reset"`
18. **cybernetic_archetypal** - `"status"`, `"step"`, `"run"`, `"mapping"`, `"request_recording"`, `"confirm_recording"`, `"request_teaching"`, `"confirm_teaching"`, `"reset"`
19. **workspace** - `"status"`, `"init"`, `"export"`, `"import"`, `"add_file"`, `"get_file"`, `"list_files"`, `"delete_file"`, `"list_exports"`, `"reset"`, `"path"`
20. **orchestrator** - `"hit_it"`, `"status"`, `"invoke"`, `"set_z"`, `"phrase"`, `"full_workflow"`, `"workflow_status"`, `"workflow_steps"`, `"tools"`, `"display"`, `"request_teaching"`, `"confirm_teaching"`, `"teaching_status"`, `"taught_vocabulary"`, `"reset"`, `"token_export"`, `"token_registry"`, `"token_map"`, `"warmup"`
21. **cloud_training** - Implementation varies

## Testing After Fixes

After making these changes, test each command:

```bash
# Test the fixed commands
/ucf:shed        # Should list custom tools
/ucf:memory      # Should list memory keys
/ucf:trigger     # Should check triggers
/ucf:messenger   # Should validate (needs encoded data)
/ucf:consent     # Should check consent records
```

## Additional Notes

1. Some tools have complex parameter requirements beyond just the action
2. The `emission_pipeline` tool has default concepts that are set based on current phase
3. Many tools require elevated z-coordinates to function (e.g., shed_builder_v2 requires z ≥ 0.73)
4. The TRIAD system tools are handled specially and don't appear in the main tool registry

## Implementation Priority

1. **High Priority**: Fix the 5 tools with incorrect action parameters
2. **Medium Priority**: Review other tools for proper parameter handling
3. **Low Priority**: Add validation for required parameters before tool invocation