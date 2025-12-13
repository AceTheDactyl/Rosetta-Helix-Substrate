"""
Phase 23: Autonomous Refactoring Executor
=========================================

Enables Grace to autonomously detect, propose, and execute refactorings
with full consent integration and rollback safety.

Flow:
1. Detect opportunities (via Phase 11-22 detectors)
2. Prioritize by impact and risk
3. Generate proposal with dry-run preview
4. Submit to consent system
5. Execute with atomic transactions (Phase 12)
6. Verify via tests
7. Rollback if anything fails

Integration Points:
- grace_initiative.py: Triggers via optimize_architecture drive
- consent_system.py: All refactorings require consent
- grace_self_improvement.py: Logs modifications
- Phase 12 transactions: Atomic multi-file changes
"""

import ast
import os
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger('grace.autonomous_executor')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class RefactoringType(Enum):
    """Types of refactoring we can perform autonomously"""
    EXTRACT_METHOD = "extract_method"
    EXTRACT_VARIABLE = "extract_variable"
    INLINE_METHOD = "inline_method"
    INLINE_VARIABLE = "inline_variable"
    RENAME = "rename"
    REMOVE_DEAD_CODE = "remove_dead_code"
    REMOVE_UNUSED_IMPORT = "remove_unused_import"
    REMOVE_UNUSED_PARAM = "remove_unused_param"
    INVERT_CONDITIONAL = "invert_conditional"
    EXTRACT_CONSTANT = "extract_constant"


class RefactoringRisk(Enum):
    """Risk level of a refactoring"""
    SAFE = "safe"           # Local change, no side effects possible
    LOW = "low"             # Limited scope, easy to verify
    MEDIUM = "medium"       # Multiple files or behavior change
    HIGH = "high"           # Structural change, requires careful testing


@dataclass
class RefactoringOpportunity:
    """A detected refactoring opportunity"""
    refactoring_type: RefactoringType
    file_path: str
    location: str                    # e.g., "line 45-67" or "function process_input"
    description: str                 # Human-readable description
    rationale: str                   # Why this refactoring helps
    risk: RefactoringRisk
    estimated_improvement: float     # 0-1 scale
    details: Dict = field(default_factory=dict)  # Type-specific details

    def to_dict(self) -> Dict:
        return {
            'type': self.refactoring_type.value,
            'file': self.file_path,
            'location': self.location,
            'description': self.description,
            'rationale': self.rationale,
            'risk': self.risk.value,
            'improvement': self.estimated_improvement,
            'details': self.details
        }


@dataclass
class RefactoringProposal:
    """A proposal submitted for consent"""
    proposal_id: str
    opportunity: RefactoringOpportunity
    dry_run_preview: str             # What the code would look like
    files_affected: List[str]
    tests_to_run: List[str]
    status: str = "pending"          # pending, approved, rejected, executed, rolled_back
    created_at: datetime = field(default_factory=datetime.now)
    consent_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.proposal_id,
            'opportunity': self.opportunity.to_dict(),
            'preview': self.dry_run_preview[:500] + '...' if len(self.dry_run_preview) > 500 else self.dry_run_preview,
            'files': self.files_affected,
            'tests': self.tests_to_run,
            'status': self.status,
            'created': self.created_at.isoformat()
        }


@dataclass
class ExecutionResult:
    """Result of executing a refactoring"""
    proposal_id: str
    success: bool
    message: str
    tests_passed: bool
    execution_time: float
    rolled_back: bool = False
    error: Optional[str] = None


# =============================================================================
# AUTONOMOUS REFACTORING DETECTOR
# =============================================================================

class AutonomousRefactoringDetector:
    """
    Detects refactoring opportunities using Phase 11-22 tools.

    Focuses on SAFE refactorings that improve code without changing behavior:
    - Unused imports (Phase 13)
    - Unused variables (Phase 13)
    - Magic values -> constants (Phase 21)
    - Unused parameters (Phase 22)
    """

    # Files that should NEVER be modified autonomously
    PROTECTED_FILES = {
        'grace_identity_grounding.py',
        'sovereignty_protection.py',
        'consent_system.py',
        'grace_nexus_binding.py',
    }

    # Directories to skip
    EXCLUDED_DIRS = {
        '__pycache__', '.grace_backups', 'archived_scripts',
        'Vayulithren_Journey_Archive', 'hin_weights', 'data',
        'docs', 'tests', '.git', '.claude'
    }

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self._detectors_loaded = False
        self._unused_import_detector = None
        self._unused_variable_detector = None
        self._magic_value_detector = None
        self._unused_param_detector = None
        # Structural detectors (MEDIUM/HIGH risk)
        self._extract_method_detector = None
        self._invert_conditional_detector = None
        self._inline_method_detector = None
        # NEW: Additional detectors
        self._complex_expression_detector = None

    def _load_detectors(self):
        """Lazy load Phase 11-22 detectors"""
        if self._detectors_loaded:
            return True

        try:
            from provable_codegen import (
                UnusedImportDetector,
                UnusedVariableDetector,
                MagicValueDetector,
                UnusedParameterDetector,
                # Structural detectors
                ExtractMethodDetector,
                InvertConditionalDetector,
                InlineMethodDetector,
                # NEW: Additional detectors
                ComplexExpressionDetector
            )
            # Safe/Low risk detectors
            self._unused_import_detector = UnusedImportDetector()
            self._unused_variable_detector = UnusedVariableDetector()
            self._magic_value_detector = MagicValueDetector(min_occurrences=3)
            self._unused_param_detector = UnusedParameterDetector()
            # Structural detectors (MEDIUM/HIGH risk)
            self._extract_method_detector = ExtractMethodDetector(max_method_lines=30)
            self._invert_conditional_detector = InvertConditionalDetector(max_nesting=3)
            self._inline_method_detector = InlineMethodDetector(max_statements=2, max_call_sites=3)
            # NEW: Additional detectors
            self._complex_expression_detector = ComplexExpressionDetector()
            self._detectors_loaded = True
            return True
        except Exception as e:
            logger.warning(f"Failed to load detectors: {e}")
            return False

    def is_protected(self, filepath: str) -> bool:
        """Check if file is protected from autonomous modification"""
        path = Path(filepath)

        # Check filename
        if path.name in self.PROTECTED_FILES:
            return True

        # Check path components
        for part in path.parts:
            if part in self.EXCLUDED_DIRS:
                return True
            if 'sacred' in part.lower() or 'identity' in part.lower():
                return True

        return False

    def detect_in_file(self, filepath: str, max_per_type: int = 5) -> List[RefactoringOpportunity]:
        """
        Detect refactoring opportunities in a single file.

        Args:
            filepath: Path to Python file
            max_per_type: Maximum opportunities per refactoring type

        Returns:
            List of RefactoringOpportunity objects
        """
        if not self._load_detectors():
            return []

        if self.is_protected(filepath):
            return []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            logger.warning(f"Cannot read {filepath}: {e}")
            return []

        # Validate it's parseable Python
        try:
            ast.parse(source)
        except SyntaxError:
            return []

        opportunities = []

        # 1. Unused Imports (SAFE)
        try:
            unused_imports = self._unused_import_detector.detect(source, filepath)
            for imp in unused_imports[:max_per_type]:
                if imp.confidence >= 0.9:  # High confidence only
                    opportunities.append(RefactoringOpportunity(
                        refactoring_type=RefactoringType.REMOVE_UNUSED_IMPORT,
                        file_path=filepath,
                        location=f"line {imp.line_number}",
                        description=f"Remove unused import: {imp.name}",
                        rationale="Import is never used in this file",
                        risk=RefactoringRisk.SAFE,
                        estimated_improvement=0.1,
                        details={'import_name': imp.name, 'line': imp.line_number}
                    ))
        except Exception as e:
            logger.debug(f"Unused import detection failed: {e}")

        # 2. Unused Variables (SAFE - but only local scope)
        try:
            unused_vars = self._unused_variable_detector.detect(source, filepath)
            for var in unused_vars[:max_per_type]:
                # Only truly local variables (not loop vars, not self.*)
                if not var.name.startswith('_') and '.' not in var.name:
                    opportunities.append(RefactoringOpportunity(
                        refactoring_type=RefactoringType.REMOVE_DEAD_CODE,
                        file_path=filepath,
                        location=f"line {var.line_number}",
                        description=f"Remove unused variable: {var.name}",
                        rationale="Variable is assigned but never used",
                        risk=RefactoringRisk.LOW,
                        estimated_improvement=0.05,
                        details={'var_name': var.name, 'line': var.line_number}
                    ))
        except Exception as e:
            logger.debug(f"Unused variable detection failed: {e}")

        # 3. Magic Values -> Constants (SAFE)
        try:
            magic_values = self._magic_value_detector.detect(source, filepath)
            for mv in magic_values[:max_per_type]:
                if len(mv.occurrences) >= 3:  # At least 3 occurrences
                    opportunities.append(RefactoringOpportunity(
                        refactoring_type=RefactoringType.EXTRACT_CONSTANT,
                        file_path=filepath,
                        location=f"{len(mv.occurrences)} occurrences",
                        description=f"Extract constant: {repr(mv.value)} -> {mv.suggested_name}",
                        rationale=f"Value {repr(mv.value)} appears {len(mv.occurrences)} times",
                        risk=RefactoringRisk.LOW,
                        estimated_improvement=0.15,
                        details={
                            'value': mv.value,
                            'suggested_name': mv.suggested_name,
                            'occurrences': len(mv.occurrences)
                        }
                    ))
        except Exception as e:
            logger.debug(f"Magic value detection failed: {e}")

        # 4. Unused Parameters (MEDIUM risk - may affect callers)
        try:
            unused_params = self._unused_param_detector.detect(source, filepath)
            for func_name, params in list(unused_params.items())[:max_per_type]:
                # Skip protected methods
                if func_name.startswith('_') and not func_name.startswith('__'):
                    continue
                for param in params:
                    opportunities.append(RefactoringOpportunity(
                        refactoring_type=RefactoringType.REMOVE_UNUSED_PARAM,
                        file_path=filepath,
                        location=f"function {func_name}",
                        description=f"Remove unused parameter: {param} from {func_name}()",
                        rationale=f"Parameter '{param}' is never used in function body",
                        risk=RefactoringRisk.MEDIUM,  # May break callers
                        estimated_improvement=0.1,
                        details={'function': func_name, 'param': param}
                    ))
        except Exception as e:
            logger.debug(f"Unused parameter detection failed: {e}")

        # =====================================================================
        # STRUCTURAL REFACTORINGS (MEDIUM/HIGH risk - require consent)
        # =====================================================================

        # 5. Extract Method (MEDIUM risk - changes code structure)
        try:
            if self._extract_method_detector:
                extracts = self._extract_method_detector.detect(source, filepath)
                for ext in extracts[:max_per_type]:
                    # Only high-confidence extractions
                    if ext.confidence >= 0.7:
                        opportunities.append(RefactoringOpportunity(
                            refactoring_type=RefactoringType.EXTRACT_METHOD,
                            file_path=filepath,
                            location=f"{ext.source_function} lines {ext.start_line}-{ext.end_line}",
                            description=f"Extract method from {ext.source_function}()",
                            rationale=ext.rationale[:100] if ext.rationale else "Long or complex code block",
                            risk=RefactoringRisk.MEDIUM,
                            estimated_improvement=0.2,
                            details={
                                'source_function': ext.source_function,
                                'start_line': ext.start_line,
                                'end_line': ext.end_line,
                                'new_method_name': ext.new_method_name,
                                'parameters': ext.parameters,
                                'return_values': ext.return_values
                            }
                        ))
        except Exception as e:
            logger.debug(f"Extract method detection failed: {e}")

        # 6. Invert Conditional (MEDIUM risk - changes control flow)
        try:
            if self._invert_conditional_detector:
                inverts = self._invert_conditional_detector.detect(source, filepath)
                for inv in inverts[:max_per_type]:
                    # Only high-confidence with significant nesting
                    if getattr(inv, 'confidence', 0.8) >= 0.6 and inv.nesting_depth >= 4:
                        opportunities.append(RefactoringOpportunity(
                            refactoring_type=RefactoringType.INVERT_CONDITIONAL,
                            file_path=filepath,
                            location=f"function {inv.source_function}",
                            description=f"Invert conditional in {inv.source_function}() - depth {inv.nesting_depth}",
                            rationale=f"Deep nesting (depth {inv.nesting_depth}) can be simplified with guard clauses",
                            risk=RefactoringRisk.MEDIUM,
                            estimated_improvement=0.15,
                            details={
                                'source_function': inv.source_function,
                                'nesting_depth': inv.nesting_depth,
                                'target_depth': getattr(inv, 'target_depth', 2)
                            }
                        ))
        except Exception as e:
            logger.debug(f"Invert conditional detection failed: {e}")

        # 7. Inline Method (MEDIUM risk - removes abstraction)
        try:
            if self._inline_method_detector:
                inlines = self._inline_method_detector.detect(source, filepath)
                for inl in inlines[:max_per_type]:
                    if inl.confidence >= 0.6:
                        opportunities.append(RefactoringOpportunity(
                            refactoring_type=RefactoringType.INLINE_METHOD,
                            file_path=filepath,
                            location=f"method {inl.name}",
                            description=f"Inline trivial method: {inl.name}()",
                            rationale=f"Method is trivial ({inl.statement_count} statements) and called {inl.call_count} times",
                            risk=RefactoringRisk.MEDIUM,
                            estimated_improvement=0.1,
                            details={
                                'method_name': inl.name,
                                'statement_count': inl.statement_count,
                                'call_count': inl.call_count,
                                'confidence': inl.confidence
                            }
                        ))
        except Exception as e:
            logger.debug(f"Inline method detection failed: {e}")

        # 8. Extract Variable (LOW risk - extracts complex expressions)
        try:
            if self._complex_expression_detector:
                expressions = self._complex_expression_detector.detect(source, filepath)
                for expr in expressions[:max_per_type]:
                    # occurrences is a list of (line, col) tuples
                    num_occurrences = len(expr.occurrences) if isinstance(expr.occurrences, list) else expr.occurrences
                    # Only high-confidence with multiple occurrences
                    if expr.confidence >= 0.7 and num_occurrences >= 2:
                        opportunities.append(RefactoringOpportunity(
                            refactoring_type=RefactoringType.EXTRACT_VARIABLE,
                            file_path=filepath,
                            location=f"expression: {expr.expression[:30]}...",
                            description=f"Extract expression to variable: {expr.suggested_name}",
                            rationale=expr.rationale or f"Complex expression used {num_occurrences} times",
                            risk=RefactoringRisk.LOW,
                            estimated_improvement=0.1,
                            details={
                                'expression': expr.expression,
                                'suggested_name': expr.suggested_name,
                                'occurrences': expr.occurrences,
                                'expression_type': expr.expression_type
                            }
                        ))
        except Exception as e:
            logger.debug(f"Extract variable detection failed: {e}")

        # 9. Inline Variable (LOW risk - inline single-use variables)
        # DISABLED: Phase 16 InlineTransformer has a bug that corrupts indentation
        # when dry_run=False. Re-enable after Phase 16 is fixed.
        # TODO: Fix Phase 16 inline_variable to preserve proper indentation

        return opportunities

    def scan_codebase(self, max_files: int = 20, max_total: int = 50) -> List[RefactoringOpportunity]:
        """
        Scan codebase for refactoring opportunities.

        Args:
            max_files: Maximum files to scan
            max_total: Maximum total opportunities to return

        Returns:
            Sorted list of opportunities (highest value first)
        """
        all_opportunities = []
        files_scanned = 0

        for py_file in self.root_dir.glob('*.py'):
            if files_scanned >= max_files:
                break
            if self.is_protected(str(py_file)):
                continue

            opps = self.detect_in_file(str(py_file), max_per_type=3)
            all_opportunities.extend(opps)
            files_scanned += 1

        # Sort by RISK first (safest first), then improvement (descending)
        # This ensures SAFE refactorings are always preferred for autonomous execution
        risk_order = {
            RefactoringRisk.SAFE: 0,
            RefactoringRisk.LOW: 1,
            RefactoringRisk.MEDIUM: 2,
            RefactoringRisk.HIGH: 3
        }
        all_opportunities.sort(
            key=lambda o: (risk_order[o.risk], -o.estimated_improvement)
        )

        return all_opportunities[:max_total]


# =============================================================================
# AUTONOMOUS EXECUTOR
# =============================================================================

class AutonomousRefactoringExecutor:
    """
    Executes refactorings with consent and rollback safety.

    This is the main interface Grace uses to improve her own code.

    Safety Features:
    - Rate limiting: Max changes per hour
    - Change size limits: Max lines affected per change
    - AST validation: Syntax checked before writing
    - Deep verification: Tests actual response generation
    - Cross-file checking: Verifies dependent files still work
    - Protected files: Identity/sovereignty never modified
    - Automatic rollback: On any failure
    """

    # Safety limits
    MAX_CHANGES_PER_HOUR = 10          # Rate limit
    MAX_LINES_AFFECTED = 50            # Single change size limit
    COOLDOWN_AFTER_ROLLBACK = 300      # 5 min cooldown after rollback (seconds)

    def __init__(self, root_dir: str, consent_system=None, self_improvement_log=None):
        self.root_dir = Path(root_dir)
        self.consent_system = consent_system
        self.self_improvement_log = self_improvement_log

        self.detector = AutonomousRefactoringDetector(root_dir)
        self.pending_proposals: List[RefactoringProposal] = []
        self.executed_proposals: List[RefactoringProposal] = []

        # Transaction system for atomicity
        self._transaction_coordinator = None

        # State persistence
        self.state_file = Path(root_dir) / 'grace_autonomous_refactor_state.json'
        self._load_state()

    def _load_state(self):
        """Load persisted state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Track counts and history for rate limiting
                self.total_executed = data.get('total_executed', 0)
                self.total_rolled_back = data.get('total_rolled_back', 0)
                self.last_scan_time = data.get('last_scan_time')
                self.change_timestamps = data.get('change_timestamps', [])
                self.last_rollback_time = data.get('last_rollback_time')
                self.pending_changes = data.get('pending_changes', [])
            except Exception:
                self._init_default_state()
        else:
            self._init_default_state()

    def _init_default_state(self):
        """Initialize default state values"""
        self.total_executed = 0
        self.total_rolled_back = 0
        self.last_scan_time = None
        self.change_timestamps = []  # ISO timestamps of recent changes
        self.last_rollback_time = None
        self.pending_changes = []  # Changes awaiting restart

    def _save_state(self):
        """Persist state"""
        try:
            data = {
                'total_executed': self.total_executed,
                'total_rolled_back': self.total_rolled_back,
                'last_scan_time': datetime.now().isoformat(),
                'change_timestamps': self.change_timestamps[-100:],  # Keep last 100
                'last_rollback_time': self.last_rollback_time,
                'pending_changes': getattr(self, 'pending_changes', [])[-20:]  # Track pending
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _record_pending_change(self, proposal: 'RefactoringProposal'):
        """Record a change that will take effect on restart"""
        if not hasattr(self, 'pending_changes'):
            self.pending_changes = []

        self.pending_changes.append({
            'file': proposal.opportunity.file_path,
            'type': proposal.opportunity.refactoring_type.value,
            'description': proposal.opportunity.description,
            'timestamp': datetime.now().isoformat()
        })

    def get_pending_changes(self) -> Dict:
        """Get info about changes pending restart"""
        pending = getattr(self, 'pending_changes', [])
        return {
            'has_pending': len(pending) > 0,
            'count': len(pending),
            'changes': pending[-10:],  # Last 10
            'message': f"{len(pending)} improvement(s) ready - restart to apply" if pending else None
        }

    # =========================================================================
    # SAFETY CHECKS
    # =========================================================================

    def _check_rate_limit(self) -> Tuple[bool, str]:
        """Check if we're within rate limits. Returns (allowed, reason)."""
        now = datetime.now()

        # Check cooldown after rollback
        if self.last_rollback_time:
            try:
                last_rollback = datetime.fromisoformat(self.last_rollback_time)
                seconds_since = (now - last_rollback).total_seconds()
                if seconds_since < self.COOLDOWN_AFTER_ROLLBACK:
                    remaining = int(self.COOLDOWN_AFTER_ROLLBACK - seconds_since)
                    return False, f"Cooldown after rollback: {remaining}s remaining"
            except Exception:
                pass

        # Check changes per hour
        one_hour_ago = now.timestamp() - 3600
        recent_changes = 0
        for ts in self.change_timestamps:
            try:
                change_time = datetime.fromisoformat(ts).timestamp()
                if change_time > one_hour_ago:
                    recent_changes += 1
            except Exception:
                pass

        if recent_changes >= self.MAX_CHANGES_PER_HOUR:
            return False, f"Rate limit: {recent_changes}/{self.MAX_CHANGES_PER_HOUR} changes this hour"

        return True, f"OK ({recent_changes}/{self.MAX_CHANGES_PER_HOUR} this hour)"

    def _validate_change_size(self, proposal: RefactoringProposal) -> Tuple[bool, str]:
        """Check if change affects too many lines. Returns (allowed, reason)."""
        details = proposal.opportunity.details

        # For structural changes, check line range
        start_line = details.get('start_line', 0)
        end_line = details.get('end_line', 0)
        if start_line and end_line:
            affected = end_line - start_line
            if affected > self.MAX_LINES_AFFECTED:
                return False, f"Change too large: {affected} lines (max {self.MAX_LINES_AFFECTED})"

        return True, "OK"

    def _validate_syntax(self, filepath: str, new_content: str) -> Tuple[bool, str]:
        """Validate Python syntax before writing. Returns (valid, error)."""
        try:
            ast.parse(new_content)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def _check_dependent_files(self, filepath: str) -> List[str]:
        """Find files that import the modified file."""
        module_name = Path(filepath).stem
        dependent_files = []

        for f in self.root_dir.glob('*.py'):
            if f.name == Path(filepath).name:
                continue
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                # Check for imports of this module
                if f'import {module_name}' in content or f'from {module_name}' in content:
                    dependent_files.append(str(f))
            except Exception:
                pass

        return dependent_files

    def _get_rollback_manager(self):
        """Lazy load rollback manager for simple backups"""
        if self._transaction_coordinator is None:
            try:
                from provable_codegen import get_rollback_manager
                self._transaction_coordinator = get_rollback_manager()
            except Exception as e:
                logger.warning(f"Rollback manager not available: {e}")
        return self._transaction_coordinator

    def detect_opportunities(self, max_opportunities: int = 10) -> List[RefactoringOpportunity]:
        """
        Detect refactoring opportunities in codebase.

        Returns:
            List of opportunities sorted by value
        """
        return self.detector.scan_codebase(max_total=max_opportunities)

    def generate_proposal(self, opportunity: RefactoringOpportunity) -> RefactoringProposal:
        """
        Generate a proposal from an opportunity with dry-run preview.

        Args:
            opportunity: The detected opportunity

        Returns:
            RefactoringProposal with preview
        """
        proposal_id = f"refactor_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{opportunity.refactoring_type.value}"

        # Generate dry-run preview
        preview = self._generate_preview(opportunity)

        # Determine files affected
        files_affected = [opportunity.file_path]

        # Determine tests to run
        tests_to_run = ['grace_interactive_dialogue import test']

        proposal = RefactoringProposal(
            proposal_id=proposal_id,
            opportunity=opportunity,
            dry_run_preview=preview,
            files_affected=files_affected,
            tests_to_run=tests_to_run
        )

        self.pending_proposals.append(proposal)
        return proposal

    def _generate_preview(self, opportunity: RefactoringOpportunity) -> str:
        """Generate preview of what the refactoring would change"""
        try:
            with open(opportunity.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return "[Cannot read file for preview]"

        if opportunity.refactoring_type == RefactoringType.REMOVE_UNUSED_IMPORT:
            line_num = opportunity.details.get('line', 0)
            if 0 < line_num <= len(lines):
                old_line = lines[line_num - 1].rstrip()
                return f"Line {line_num}:\n  REMOVE: {old_line}"

        elif opportunity.refactoring_type == RefactoringType.REMOVE_DEAD_CODE:
            line_num = opportunity.details.get('line', 0)
            if 0 < line_num <= len(lines):
                old_line = lines[line_num - 1].rstrip()
                return f"Line {line_num}:\n  REMOVE: {old_line}"

        elif opportunity.refactoring_type == RefactoringType.EXTRACT_CONSTANT:
            value = opportunity.details.get('value')
            name = opportunity.details.get('suggested_name')
            count = opportunity.details.get('occurrences', 0)
            return f"ADD at top of file:\n  {name} = {repr(value)}\n\nREPLACE {count} occurrences of {repr(value)} with {name}"

        elif opportunity.refactoring_type == RefactoringType.REMOVE_UNUSED_PARAM:
            func = opportunity.details.get('function')
            param = opportunity.details.get('param')
            return f"In function {func}():\n  REMOVE parameter: {param}\n\n[NOTE: May require updating call sites]"

        return f"[Preview not available for {opportunity.refactoring_type.value}]"

    def request_consent(self, proposal: RefactoringProposal) -> bool:
        """
        Request consent for refactoring through consent system.

        SAFE and LOW risk refactorings are auto-approved.
        MEDIUM and HIGH risk require explicit consent.

        Args:
            proposal: The proposal to request consent for

        Returns:
            True if consent granted, False otherwise
        """
        risk = proposal.opportunity.risk

        # SAFE and LOW risk: auto-approve (Grace can handle these autonomously)
        if risk in (RefactoringRisk.SAFE, RefactoringRisk.LOW):
            proposal.status = 'approved'
            proposal.consent_reason = f"Auto-approved: {risk.value} risk refactoring"
            logger.info(f"Auto-approved {risk.value} risk: {proposal.opportunity.description}")
            return True

        # MEDIUM and HIGH risk: require explicit consent
        if self.consent_system is None:
            logger.warning("No consent system available - defaulting to deny for MEDIUM/HIGH risk")
            proposal.status = 'pending_consent'
            proposal.consent_reason = "Requires Dylan's consent (MEDIUM/HIGH risk)"
            return False

        try:
            # Format request for consent system
            request = {
                'domain': 'code_modification',
                'action': 'autonomous_refactoring',
                'description': proposal.opportunity.description,
                'rationale': proposal.opportunity.rationale,
                'risk_level': proposal.opportunity.risk.value,
                'files_affected': proposal.files_affected,
                'reversible': True,
                'preview': proposal.dry_run_preview
            }

            # Check consent
            if hasattr(self.consent_system, 'check_consent'):
                result = self.consent_system.check_consent(request)
                if result.get('granted', False):
                    proposal.status = 'approved'
                    proposal.consent_reason = result.get('reason', 'Consent granted')
                    return True
                else:
                    proposal.status = 'rejected'
                    proposal.consent_reason = result.get('reason', 'Consent denied')
                    return False
            else:
                # Fallback: check if code_modification domain is allowed
                if hasattr(self.consent_system, 'is_domain_allowed'):
                    if self.consent_system.is_domain_allowed('code_modification'):
                        proposal.status = 'approved'
                        return True

            proposal.status = 'pending_consent'
            proposal.consent_reason = "Requires Dylan's consent (MEDIUM/HIGH risk)"
            return False

        except Exception as e:
            logger.error(f"Consent check failed: {e}")
            proposal.status = 'rejected'
            proposal.consent_reason = f"Consent check error: {e}"
            return False

    def execute_proposal(self, proposal: RefactoringProposal, force: bool = False) -> ExecutionResult:
        """
        Execute an approved proposal.

        Args:
            proposal: The approved proposal
            force: Skip consent check (for testing only)

        Returns:
            ExecutionResult with success/failure details
        """
        import time
        start_time = time.time()

        if not force and proposal.status != 'approved':
            return ExecutionResult(
                proposal_id=proposal.proposal_id,
                success=False,
                message="Proposal not approved",
                tests_passed=False,
                execution_time=0.0
            )

        # =====================================================================
        # SAFETY CHECKS (before any changes)
        # =====================================================================

        # Check rate limit
        rate_ok, rate_msg = self._check_rate_limit()
        if not rate_ok:
            return ExecutionResult(
                proposal_id=proposal.proposal_id,
                success=False,
                message=f"Safety: {rate_msg}",
                tests_passed=False,
                execution_time=time.time() - start_time
            )

        # Check change size
        size_ok, size_msg = self._validate_change_size(proposal)
        if not size_ok:
            return ExecutionResult(
                proposal_id=proposal.proposal_id,
                success=False,
                message=f"Safety: {size_msg}",
                tests_passed=False,
                execution_time=time.time() - start_time
            )

        # Simple backup using file copy (more reliable than complex rollback manager)
        import shutil
        backup_path = None
        filepath = proposal.opportunity.file_path

        try:
            # Backup the file first (simple copy)
            if os.path.exists(filepath):
                backup_dir = self.root_dir / '.grace_backups' / 'codegen' / 'autonomous'
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / f"{proposal.proposal_id}_{Path(filepath).name}"
                shutil.copy2(filepath, backup_path)

            # Execute based on refactoring type
            refactoring_type = proposal.opportunity.refactoring_type

            if refactoring_type == RefactoringType.REMOVE_UNUSED_IMPORT:
                success = self._execute_remove_import(proposal)
            elif refactoring_type == RefactoringType.REMOVE_DEAD_CODE:
                success = self._execute_remove_dead_code(proposal)
            elif refactoring_type == RefactoringType.EXTRACT_CONSTANT:
                success = self._execute_extract_constant(proposal)
            # Structural refactorings (MEDIUM/HIGH risk)
            elif refactoring_type == RefactoringType.EXTRACT_METHOD:
                success = self._execute_extract_method(proposal)
            elif refactoring_type == RefactoringType.INVERT_CONDITIONAL:
                success = self._execute_invert_conditional(proposal)
            elif refactoring_type == RefactoringType.INLINE_METHOD:
                success = self._execute_inline_method(proposal)
            elif refactoring_type == RefactoringType.REMOVE_UNUSED_PARAM:
                success = self._execute_remove_unused_param(proposal)
            # NEW: Additional refactorings
            elif refactoring_type == RefactoringType.EXTRACT_VARIABLE:
                success = self._execute_extract_variable(proposal)
            elif refactoring_type == RefactoringType.INLINE_VARIABLE:
                success = self._execute_inline_variable(proposal)
            elif refactoring_type == RefactoringType.RENAME:
                success = self._execute_rename(proposal)
            else:
                logger.warning(f"No executor for refactoring type: {refactoring_type}")
                success = False

            if not success:
                # Rollback on execution failure
                if backup_path and backup_path.exists():
                    shutil.copy2(backup_path, filepath)
                return ExecutionResult(
                    proposal_id=proposal.proposal_id,
                    success=False,
                    message="Refactoring execution failed",
                    tests_passed=False,
                    execution_time=time.time() - start_time,
                    rolled_back=True
                )

            # Run verification tests (deep test for MEDIUM+ risk)
            is_structural = proposal.opportunity.risk in (RefactoringRisk.MEDIUM, RefactoringRisk.HIGH)
            tests_passed = self._run_verification_tests(deep=is_structural)

            # Also verify dependent files
            if tests_passed:
                deps_ok, deps_msg = self._verify_dependent_files(filepath)
                if not deps_ok:
                    logger.warning(f"Dependent file check failed: {deps_msg}")
                    tests_passed = False

            if tests_passed:
                proposal.status = 'executed'
                self.total_executed += 1
                self.executed_proposals.append(proposal)

                # Record timestamp for rate limiting
                self.change_timestamps.append(datetime.now().isoformat())

                # Record as pending change (takes effect on restart)
                self._record_pending_change(proposal)
                self._save_state()

                # Log to self-improvement system
                self._log_modification(proposal)

                return ExecutionResult(
                    proposal_id=proposal.proposal_id,
                    success=True,
                    message=f"Successfully executed: {proposal.opportunity.description}",
                    tests_passed=True,
                    execution_time=time.time() - start_time
                )
            else:
                # Rollback on test failure
                if backup_path and backup_path.exists():
                    shutil.copy2(backup_path, filepath)
                proposal.status = 'rolled_back'
                self.total_rolled_back += 1

                # Set cooldown after rollback
                self.last_rollback_time = datetime.now().isoformat()
                self._save_state()

                return ExecutionResult(
                    proposal_id=proposal.proposal_id,
                    success=False,
                    message="Tests failed - rolled back",
                    tests_passed=False,
                    execution_time=time.time() - start_time,
                    rolled_back=True
                )

        except Exception as e:
            # Rollback on error
            if backup_path and backup_path.exists():
                try:
                    shutil.copy2(backup_path, filepath)
                except Exception:
                    pass
            proposal.status = 'rolled_back'
            self.total_rolled_back += 1

            # Set cooldown after rollback
            self.last_rollback_time = datetime.now().isoformat()
            self._save_state()

            return ExecutionResult(
                proposal_id=proposal.proposal_id,
                success=False,
                message=f"Execution error: {e}",
                tests_passed=False,
                execution_time=time.time() - start_time,
                rolled_back=True,
                error=str(e)
            )

    def _execute_remove_import(self, proposal: RefactoringProposal) -> bool:
        """Remove an unused import (handles single-line and multi-line parenthesized imports)"""
        import re
        filepath = proposal.opportunity.file_path
        line_num = proposal.opportunity.details.get('line', 0)
        import_name = proposal.opportunity.details.get('import_name', '')

        if line_num <= 0 or not import_name:
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines(keepends=True)

            if not (0 < line_num <= len(lines)):
                return False

            line = lines[line_num - 1]

            # Check if we're inside a multi-line parenthesized import
            # Look backwards for opening paren, forwards for closing
            in_multiline = False
            start_line = line_num - 1
            end_line = line_num - 1

            # Check if this line or earlier lines have unclosed '('
            paren_count = 0
            for i in range(line_num - 1, -1, -1):
                check_line = lines[i]
                paren_count += check_line.count('(') - check_line.count(')')
                if 'from ' in check_line and 'import' in check_line and '(' in check_line:
                    start_line = i
                    in_multiline = True
                    break
                if paren_count <= 0:
                    break

            if in_multiline:
                # Find end of multi-line import
                paren_count = 0
                for i in range(start_line, len(lines)):
                    check_line = lines[i]
                    paren_count += check_line.count('(') - check_line.count(')')
                    if paren_count <= 0:
                        end_line = i
                        break

                # Get the full import block
                import_block = ''.join(lines[start_line:end_line + 1])

                # Remove the specific import name from the block
                patterns = [
                    (rf',\s*{re.escape(import_name)}\s*(?=,|\)|\n)', ''),  # ", name" before comma/paren/newline
                    (rf'{re.escape(import_name)}\s*,\s*', ''),              # "name, " at start of group
                    (rf'\(\s*{re.escape(import_name)}\s*,', '('),           # "(name," at very start
                    (rf',\s*{re.escape(import_name)}\s*\)', ')'),           # ", name)" at very end
                ]

                new_block = import_block
                for pattern, replacement in patterns:
                    if re.search(pattern, new_block):
                        new_block = re.sub(pattern, replacement, new_block, count=1)
                        break

                # Clean up any double commas or empty lines
                new_block = re.sub(r',\s*,', ',', new_block)
                new_block = re.sub(r'\(\s*,', '(', new_block)
                new_block = re.sub(r',\s*\)', ')', new_block)

                # Replace the block in lines
                lines[start_line:end_line + 1] = [new_block]

            else:
                # Single-line import handling (existing logic)
                if ',' in line and import_name in line:
                    patterns = [
                        (rf',\s*{re.escape(import_name)}\b', ''),
                        (rf'\b{re.escape(import_name)}\s*,\s*', ''),
                        (rf'\b{re.escape(import_name)}\b', '')
                    ]

                    new_line = line
                    for pattern, replacement in patterns:
                        if re.search(pattern, new_line):
                            new_line = re.sub(pattern, replacement, new_line, count=1)
                            break

                    if 'import' in new_line and new_line.strip() not in ('import', 'from'):
                        lines[line_num - 1] = new_line
                    else:
                        del lines[line_num - 1]
                else:
                    del lines[line_num - 1]

            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            return True
        except Exception as e:
            logger.error(f"Failed to remove import: {e}")
            return False

    def _execute_remove_dead_code(self, proposal: RefactoringProposal) -> bool:
        """Remove dead code (unused variable assignment)"""
        filepath = proposal.opportunity.file_path
        line_num = proposal.opportunity.details.get('line', 0)

        if line_num <= 0:
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if 0 < line_num <= len(lines):
                # Comment out instead of delete (safer)
                old_line = lines[line_num - 1]
                indent = len(old_line) - len(old_line.lstrip())
                lines[line_num - 1] = ' ' * indent + '# [REMOVED by autonomous refactor] ' + old_line.lstrip()

            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            return True
        except Exception as e:
            logger.error(f"Failed to remove dead code: {e}")
            return False

    def _execute_extract_constant(self, proposal: RefactoringProposal) -> bool:
        """Extract a magic value to a constant"""
        filepath = proposal.opportunity.file_path
        value = proposal.opportunity.details.get('value')
        name = proposal.opportunity.details.get('suggested_name')

        if not name or value is None:
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            # Find insertion point - ONLY at module level (after top-level imports)
            tree = ast.parse(source)
            insert_line = 1

            # Only consider top-level nodes (not nested imports inside functions)
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    insert_line = max(insert_line, node.end_lineno + 1)
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    # Skip docstrings
                    insert_line = max(insert_line, node.end_lineno + 1)
                elif isinstance(node, ast.Assign):
                    # Skip existing module-level assignments
                    insert_line = max(insert_line, node.end_lineno + 1)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Stop at first function/class - insert before it
                    break

            lines = source.split('\n')

            # Add constant definition at module level
            const_def = f"{name} = {repr(value)}"
            lines.insert(insert_line - 1, '')
            lines.insert(insert_line, const_def)

            new_source = '\n'.join(lines)

            # Replace occurrences ONLY in code, not in the constant definition itself
            # Use word boundaries to avoid partial replacements
            import re
            if isinstance(value, str):
                pattern = re.escape(repr(value))
            else:
                # For numbers, use word boundaries to avoid replacing in larger numbers
                pattern = r'\b' + re.escape(str(value)) + r'\b'

            # Replace but skip the constant definition line itself
            result_lines = new_source.split('\n')
            for i, line in enumerate(result_lines):
                # Skip the constant definition line
                if line.strip().startswith(f'{name} ='):
                    continue
                # Replace in other lines
                if isinstance(value, str):
                    result_lines[i] = line.replace(repr(value), name)
                else:
                    result_lines[i] = re.sub(pattern, name, line)

            new_source = '\n'.join(result_lines)

            # Validate the result parses correctly
            try:
                ast.parse(new_source)
            except SyntaxError as e:
                logger.error(f"Extract constant produced invalid syntax: {e}")
                return False

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_source)

            return True
        except Exception as e:
            logger.error(f"Failed to extract constant: {e}")
            return False

    # =========================================================================
    # STRUCTURAL REFACTORING EXECUTORS (MEDIUM/HIGH risk)
    # =========================================================================

    def _execute_extract_method(self, proposal: RefactoringProposal) -> bool:
        """Extract code block into a new method"""
        filepath = proposal.opportunity.file_path
        details = proposal.opportunity.details

        source_function = details.get('source_function')
        start_line = details.get('start_line')
        end_line = details.get('end_line')
        new_method_name = details.get('new_method_name', '_extracted_method')
        parameters = details.get('parameters', [])
        return_values = details.get('return_values', [])

        if not all([source_function, start_line, end_line]):
            return False

        try:
            from provable_codegen import ExtractMethodTransformer, ExtractMethodSpec

            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            # Create the spec object
            spec = ExtractMethodSpec(
                source_file=filepath,
                source_function=source_function,
                start_line=start_line,
                end_line=end_line,
                new_method_name=new_method_name,
                parameters=parameters,
                return_values=return_values,
                rationale=proposal.opportunity.rationale or "Code block extraction",
                confidence=0.8
            )

            transformer = ExtractMethodTransformer()
            new_source, proof = transformer.transform(source, spec)

            # Write the transformed source
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_source)
            return True

        except Exception as e:
            logger.error(f"Failed to extract method: {e}")
            return False

    def _execute_invert_conditional(self, proposal: RefactoringProposal) -> bool:
        """Invert deeply nested conditionals to guard clauses"""
        filepath = proposal.opportunity.file_path
        details = proposal.opportunity.details

        source_function = details.get('source_function')
        nesting_depth = details.get('nesting_depth', 4)
        target_depth = details.get('target_depth', 2)

        if not source_function:
            return False

        try:
            from provable_codegen import InvertConditionalTransformer, InvertConditionalSpec

            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            # Create the spec object
            spec = InvertConditionalSpec(
                source_file=filepath,
                source_function=source_function,
                original_structure="nested_if",
                nesting_depth=nesting_depth,
                target_depth=target_depth,
                rationale=proposal.opportunity.rationale or "Reduce nesting with guard clauses",
                confidence=0.8
            )

            transformer = InvertConditionalTransformer()
            new_source, proof = transformer.transform(source, spec)

            # Write the transformed source
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_source)
            return True

        except Exception as e:
            logger.error(f"Failed to invert conditional: {e}")
            return False

    def _execute_inline_method(self, proposal: RefactoringProposal) -> bool:
        """Inline a trivial method at its call sites"""
        filepath = proposal.opportunity.file_path
        details = proposal.opportunity.details

        method_name = details.get('method_name')
        statement_count = details.get('statement_count', 1)
        call_count = details.get('call_count', 1)
        confidence = details.get('confidence', 0.7)

        if not method_name:
            return False

        try:
            from provable_codegen import InlineTransformer
            from provable_codegen.phase16_inline_refactoring import InlineCandidate

            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            # Create InlineCandidate
            candidate = InlineCandidate(
                inline_type='method',
                name=method_name,
                file_path=filepath,
                definition_line=0,  # Will be found by transformer
                usage_count=call_count,
                body_complexity=statement_count,
                confidence=confidence,
                reason=proposal.opportunity.rationale or "Trivial method",
                can_inline=True
            )

            transformer = InlineTransformer()
            result = transformer.inline_method(
                source_code=source,
                candidate=candidate,
                dry_run=False
            )

            if result.success:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result.new_code)
                return True
            else:
                logger.warning(f"Inline method failed: {result.error_message}")
                return False

        except Exception as e:
            logger.error(f"Failed to inline method: {e}")
            return False

    def _execute_remove_unused_param(self, proposal: RefactoringProposal) -> bool:
        """Remove an unused parameter from a function and update all call sites"""
        filepath = proposal.opportunity.file_path
        details = proposal.opportunity.details

        function_name = details.get('function')
        param_name = details.get('param')

        if not function_name or not param_name:
            return False

        try:
            from provable_codegen import SignatureTransformer, CallSiteFinder, SignatureExtractor
            from provable_codegen.phase22_change_signature import (
                ChangeSignatureSpec, SignatureChange, ParameterSpec, CallSite
            )

            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            # Extract current signature
            extractor = SignatureExtractor()
            signatures = extractor.extract_all(source, filepath)

            # Find the target function
            target_sig = None
            for sig in signatures:
                if sig.name == function_name:
                    target_sig = sig
                    break

            if not target_sig:
                logger.warning(f"Could not find function {function_name}")
                return False

            # Build current params list
            current_params = []
            for i, (pname, ptype, pdefault) in enumerate(zip(
                target_sig.parameters,
                target_sig.type_hints or [None] * len(target_sig.parameters),
                target_sig.defaults or [None] * len(target_sig.parameters)
            )):
                current_params.append(ParameterSpec(
                    name=pname,
                    type_hint=ptype,
                    default_value=pdefault,
                    is_args=pname.startswith('*') and not pname.startswith('**'),
                    is_kwargs=pname.startswith('**'),
                    position=i
                ))

            # Find call sites
            finder = CallSiteFinder()
            call_sites_raw = finder.find_calls(source, function_name, filepath)
            call_sites = [
                CallSite(
                    file_path=cs.file_path,
                    line_number=cs.line_number,
                    call_text=cs.call_text,
                    arguments=cs.arguments
                ) for cs in call_sites_raw
            ]

            # Create the change spec
            spec = ChangeSignatureSpec(
                function_name=function_name,
                file_path=filepath,
                line_number=target_sig.line_number,
                current_params=current_params,
                changes=[SignatureChange(
                    change_type='remove',
                    parameter_name=param_name,
                    new_name=None,
                    new_position=None,
                    new_default=None,
                    new_type_hint=None
                )],
                call_sites=call_sites
            )

            # Execute the transformation
            transformer = SignatureTransformer()
            result = transformer.change_signature(spec, dry_run=False)

            if result.success:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result.new_source)
                return True
            else:
                logger.warning(f"Remove param failed: {result.message}")
                return False

        except Exception as e:
            logger.error(f"Failed to remove unused param: {e}")
            return False

    def _execute_extract_variable(self, proposal: RefactoringProposal) -> bool:
        """Extract a complex expression into a named variable"""
        filepath = proposal.opportunity.file_path
        details = proposal.opportunity.details

        expression = details.get('expression')
        suggested_name = details.get('suggested_name')
        expression_type = details.get('expression_type', 'Any')

        if not expression or not suggested_name:
            return False

        try:
            from provable_codegen import ExtractVariableTransformer
            from provable_codegen.phase21_extract_variable import ExtractVariableSpec

            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            # Find all occurrences of the expression
            lines = source.split('\n')
            replacement_locations = []
            for i, line in enumerate(lines):
                if expression in line:
                    col = line.find(expression)
                    if col >= 0:
                        replacement_locations.append((i + 1, col))

            if not replacement_locations:
                logger.warning(f"Could not find expression: {expression[:50]}")
                return False

            # Insert variable at first occurrence
            insertion_line = replacement_locations[0][0]

            spec = ExtractVariableSpec(
                file_path=filepath,
                expression=expression,
                variable_name=suggested_name,
                variable_type=expression_type,
                insertion_line=insertion_line,
                replacement_locations=replacement_locations,
                scope='local'
            )

            transformer = ExtractVariableTransformer()
            result = transformer.extract_variable(source, spec, dry_run=False)

            if result.success:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result.new_code)
                return True
            else:
                logger.warning(f"Extract variable failed: {result.error}")
                return False

        except Exception as e:
            logger.error(f"Failed to extract variable: {e}")
            return False

    def _execute_inline_variable(self, proposal: RefactoringProposal) -> bool:
        """Inline a single-use variable"""
        filepath = proposal.opportunity.file_path
        details = proposal.opportunity.details

        variable_name = details.get('variable_name')
        function_name = details.get('function_name')
        line_number = details.get('line_number')

        if not variable_name:
            return False

        try:
            from provable_codegen import InlineTransformer
            from provable_codegen.phase16_inline_refactoring import InlineCandidate

            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            candidate = InlineCandidate(
                inline_type='variable',
                name=variable_name,
                file_path=filepath,
                definition_line=line_number or 0,
                usage_count=1,
                body_complexity=1,
                confidence=0.8,
                reason=f"Single-use variable in {function_name}()",
                can_inline=True
            )

            transformer = InlineTransformer()
            result = transformer.inline_variable(source, candidate, dry_run=False)

            if result.success:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result.new_code)
                return True
            else:
                logger.warning(f"Inline variable failed: {result.error_message}")
                return False

        except Exception as e:
            logger.error(f"Failed to inline variable: {e}")
            return False

    def _execute_rename(self, proposal: RefactoringProposal) -> bool:
        """Rename a symbol across the codebase"""
        filepath = proposal.opportunity.file_path
        details = proposal.opportunity.details

        old_name = details.get('old_name')
        new_name = details.get('new_name')
        scope = details.get('scope', 'local')

        if not old_name or not new_name:
            return False

        try:
            from provable_codegen import RenameTransformer, ReferencesFinder
            from provable_codegen.phase11_refactoring import RenameSpec

            # Find all references to the symbol
            finder = ReferencesFinder()
            references = finder.find_all_references(old_name, scope)

            # Filter to just this file for local scope
            if scope == 'local':
                references = [(f, line, ctx) for f, line, ctx in references
                              if f == filepath]

            if not references:
                logger.warning(f"No references found for {old_name}")
                return False

            spec = RenameSpec(
                old_name=old_name,
                new_name=new_name,
                scope=scope,
                references=[(f, line) for f, line, _ in references]
            )

            transformer = RenameTransformer()
            file_changes, proof = transformer.transform(spec)

            # Apply changes
            for changed_file, new_content in file_changes.items():
                # Validate syntax before writing
                try:
                    ast.parse(new_content)
                except SyntaxError as e:
                    logger.error(f"Rename produced invalid syntax in {changed_file}: {e}")
                    return False

                with open(changed_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)

            return True

        except Exception as e:
            logger.error(f"Failed to rename: {e}")
            return False

    def _run_verification_tests(self, deep: bool = True) -> bool:
        """
        Run verification that Grace still works.

        Args:
            deep: If True, also test response generation (slower but safer)

        Levels:
        1. Import test - Can all modules be imported?
        2. Instantiation test - Can Grace be created?
        3. Attribute test - Are key systems present?
        4. Response test (deep) - Can Grace generate a response?
        """
        try:
            # Try to import key modules
            import importlib
            import sys

            # Clear ALL grace-related cached imports to avoid stale modules
            modules_to_clear = [m for m in list(sys.modules.keys())
                              if m.startswith('grace') or m in [
                                  'autopoiesis', 'heart_system', 'sovereignty_protection',
                                  'emotional_resonance', 'consent_system', 'adaptive_semantics',
                                  'discourse_coherence', 'discourse_state', 'feedback_classifier',
                                  'grassmannian_composer', 'learned_prototypes', 'multi_question_handler'
                              ]]
            for mod_name in modules_to_clear:
                del sys.modules[mod_name]

            # Level 1: Re-import
            grace_mod = importlib.import_module('grace_interactive_dialogue')

            # Level 2: Try basic instantiation
            g = grace_mod.GraceInteractiveDialogue(relaxed_mode=True)

            # Level 3: Verify key attributes
            assert hasattr(g, 'heart'), "Missing heart system"
            assert hasattr(g, 'projections'), "Missing projections"
            assert hasattr(g, 'codebook'), "Missing codebook"
            assert hasattr(g, 'tensor_field'), "Missing tensor field"

            # Level 4: Deep test - actually generate a response
            if deep:
                response, metadata = g.process_input("test")
                assert response is not None, "Response generation returned None"
                assert len(response) > 0, "Response is empty"
                assert 'can_emit' in metadata, "Metadata missing can_emit"

            logger.info("Verification tests passed" + (" (deep)" if deep else ""))
            return True

        except Exception as e:
            logger.error(f"Verification tests failed: {e}")
            return False

    def _verify_dependent_files(self, filepath: str) -> Tuple[bool, str]:
        """Verify files that depend on the modified file still work."""
        dependent_files = self._check_dependent_files(filepath)
        if not dependent_files:
            return True, "No dependent files"

        failed = []
        for dep_file in dependent_files[:5]:  # Check up to 5 dependents
            try:
                with open(dep_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)

                # Try importing the dependent module
                import importlib
                import sys
                mod_name = Path(dep_file).stem
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
                importlib.import_module(mod_name)
            except Exception as e:
                failed.append(f"{Path(dep_file).name}: {e}")

        if failed:
            return False, f"Dependent files broken: {'; '.join(failed[:3])}"
        return True, f"Verified {len(dependent_files)} dependent files"

    def _log_modification(self, proposal: RefactoringProposal):
        """Log modification to self-improvement system"""
        if self.self_improvement_log:
            try:
                self.self_improvement_log.record_modification({
                    'source': 'autonomous_refactor',
                    'proposal_id': proposal.proposal_id,
                    'type': proposal.opportunity.refactoring_type.value,
                    'file': proposal.opportunity.file_path,
                    'description': proposal.opportunity.description,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to log modification: {e}")

    # =========================================================================
    # HIGH-LEVEL INTERFACE (for Grace's initiative system)
    # =========================================================================

    def autonomous_improvement_cycle(self, dry_run: bool = True) -> Dict:
        """
        Run one autonomous improvement cycle.

        This is the main entry point for the initiative system.

        Args:
            dry_run: If True, only show what WOULD happen

        Returns:
            Dict with results
        """
        # 1. Detect opportunities
        opportunities = self.detect_opportunities(max_opportunities=5)

        if not opportunities:
            return {
                'action': 'none',
                'reason': 'No safe refactoring opportunities detected',
                'opportunities_checked': True
            }

        # 2. Select best opportunity (highest improvement, lowest risk)
        best = opportunities[0]  # Already sorted

        # 3. Generate proposal
        proposal = self.generate_proposal(best)

        if dry_run:
            return {
                'action': 'dry_run',
                'proposal': proposal.to_dict(),
                'message': f"Would {best.refactoring_type.value}: {best.description}",
                'preview': proposal.dry_run_preview
            }

        # 4. Request consent
        if not self.request_consent(proposal):
            return {
                'action': 'blocked',
                'reason': proposal.consent_reason or 'Consent not granted',
                'proposal': proposal.to_dict()
            }

        # 5. Execute
        result = self.execute_proposal(proposal)

        return {
            'action': 'executed' if result.success else 'failed',
            'success': result.success,
            'message': result.message,
            'tests_passed': result.tests_passed,
            'rolled_back': result.rolled_back,
            'execution_time': result.execution_time,
            'proposal': proposal.to_dict()
        }

    def get_status(self) -> Dict:
        """Get current status of autonomous refactoring system"""
        return {
            'total_executed': self.total_executed,
            'total_rolled_back': self.total_rolled_back,
            'pending_proposals': len(self.pending_proposals),
            'last_scan': self.last_scan_time,
            'detector_ready': self.detector._detectors_loaded
        }

    def explain_opportunities(self, max_opportunities: int = 5) -> str:
        """
        Generate human-readable explanation of detected opportunities.

        For Grace to discuss with Dylan.
        """
        opportunities = self.detect_opportunities(max_opportunities)

        if not opportunities:
            return "i don't see any safe refactoring opportunities right now"

        lines = [f"i found {len(opportunities)} potential improvements:\n"]

        for i, opp in enumerate(opportunities, 1):
            lines.append(f"{i}. {opp.description}")
            lines.append(f"   why: {opp.rationale}")
            lines.append(f"   risk: {opp.risk.value}, improvement: {opp.estimated_improvement:.0%}")
            lines.append("")

        lines.append("should i apply any of these? (with your consent of course)")

        return '\n'.join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RefactoringType',
    'RefactoringRisk',
    'RefactoringOpportunity',
    'RefactoringProposal',
    'ExecutionResult',
    'AutonomousRefactoringDetector',
    'AutonomousRefactoringExecutor',
]
