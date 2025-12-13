"""
Operator-Action Mapping
=======================
Translates APL operators to concrete build actions.
Each operator has semantic meaning for code generation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import random


class APLOperator(Enum):
    """APL operators with build semantics."""
    BOUNDARY = "()"    # Define/protect/interface
    FUSION = "×"       # Merge/combine
    AMPLIFY = "^"      # Boost/strengthen
    DECOHERENCE = "÷"  # Break/decouple
    GROUP = "+"        # Cluster/organize
    SEPARATE = "−"     # Split/isolate


@dataclass
class BuildState:
    """Current state of the build process."""
    task: Any
    files: Dict[str, str] = field(default_factory=dict)
    modules: List[str] = field(default_factory=list)
    coherence: float = 0.0
    z: float = 0.0
    target_z: float = 0.5
    z_velocity: float = 0.0
    coupling_score: float = 0.5
    checkpoint: Optional[Dict] = None
    
    @property
    def has_similar_modules(self) -> bool:
        """Check if there are similar modules that could merge."""
        # Simple heuristic: modules with similar names
        if len(self.modules) < 2:
            return False
        for i, m1 in enumerate(self.modules):
            for m2 in self.modules[i+1:]:
                if self._similarity(m1, m2) > 0.5:
                    return True
        return False
    
    def _similarity(self, a: str, b: str) -> float:
        """Simple string similarity."""
        a, b = a.lower(), b.lower()
        common = set(a.split('_')) & set(b.split('_'))
        total = set(a.split('_')) | set(b.split('_'))
        return len(common) / len(total) if total else 0
    
    def save_checkpoint(self):
        """Save current state as checkpoint."""
        self.checkpoint = {
            "files": self.files.copy(),
            "modules": self.modules.copy(),
            "z": self.z
        }
    
    def restore_checkpoint(self) -> "BuildState":
        """Restore from last checkpoint."""
        if self.checkpoint:
            self.files = self.checkpoint["files"].copy()
            self.modules = self.checkpoint["modules"].copy()
        return self


class OperatorActions:
    """
    Maps APL operators to build actions.
    
    Each operator transforms the build state in a specific way.
    """
    
    @staticmethod
    def apply(op: APLOperator, state: BuildState) -> BuildState:
        """Apply an operator to build state."""
        actions = {
            APLOperator.BOUNDARY: OperatorActions.boundary,
            APLOperator.FUSION: OperatorActions.fusion,
            APLOperator.AMPLIFY: OperatorActions.amplify,
            APLOperator.DECOHERENCE: OperatorActions.decoherence,
            APLOperator.GROUP: OperatorActions.group,
            APLOperator.SEPARATE: OperatorActions.separate,
        }
        
        action = actions.get(op)
        if action:
            return action(state)
        return state
    
    @staticmethod
    def boundary(state: BuildState) -> BuildState:
        """
        BOUNDARY () — Define/protect coherence
        
        BUILD MEANING:
        - Define interface contracts
        - Create module boundaries  
        - Establish API surfaces
        - Reset to stable state if failing
        """
        if state.coherence < 0.3:
            # Emergency: restore checkpoint
            state.restore_checkpoint()
            state.z_velocity = 0  # Stop momentum
        else:
            # Define boundaries: add __all__ to modules
            for filename in list(state.files.keys()):
                if filename.endswith('.py') and '__init__' not in filename:
                    content = state.files[filename]
                    if '__all__' not in content:
                        # Extract public names (simple heuristic)
                        lines = content.split('\n')
                        public = []
                        for line in lines:
                            if line.startswith('def ') and not line.startswith('def _'):
                                name = line.split('(')[0].replace('def ', '').strip()
                                public.append(name)
                            elif line.startswith('class ') and not line.startswith('class _'):
                                name = line.split('(')[0].split(':')[0].replace('class ', '').strip()
                                public.append(name)
                        
                        if public:
                            all_line = f"__all__ = {public}\n\n"
                            state.files[filename] = all_line + content
            
            # Save checkpoint at stable point
            if state.coherence >= 0.7:
                state.save_checkpoint()
        
        return state
    
    @staticmethod
    def fusion(state: BuildState) -> BuildState:
        """
        FUSION × — Merge/combine
        
        BUILD MEANING:
        - Merge related modules
        - Combine duplicate code
        - Unify similar functions
        - Consolidate tests
        """
        # Find modules to merge (simple: by prefix)
        module_groups = {}
        for filename in state.files:
            if filename.endswith('.py'):
                # Group by first part of name
                prefix = filename.split('_')[0].split('/')[0]
                if prefix not in module_groups:
                    module_groups[prefix] = []
                module_groups[prefix].append(filename)
        
        # Merge small related modules
        for prefix, files in module_groups.items():
            if len(files) >= 2:
                # Check if they're small enough to merge
                total_lines = sum(
                    len(state.files[f].split('\n')) 
                    for f in files if f in state.files
                )
                if total_lines < 200:  # Merge if combined < 200 lines
                    merged_content = f'"""Merged module: {prefix}"""\n\n'
                    for f in files:
                        if f in state.files:
                            merged_content += f"# From {f}\n"
                            merged_content += state.files[f]
                            merged_content += "\n\n"
                    
                    # Keep first, update, remove others
                    state.files[files[0]] = merged_content
                    for f in files[1:]:
                        if f in state.files:
                            del state.files[f]
        
        # Increase coupling score (fusion increases coupling)
        state.coupling_score = min(1.0, state.coupling_score + 0.1)
        
        return state
    
    @staticmethod
    def amplify(state: BuildState) -> BuildState:
        """
        AMPLIFY ^ — Boost/strengthen
        
        BUILD MEANING:
        - Add more tests
        - Strengthen validation
        - Expand documentation
        - Push z toward target
        """
        # Boost z toward target
        diff = state.target_z - state.z
        state.z_velocity += diff * 0.1
        
        # Add docstrings where missing
        for filename in state.files:
            if filename.endswith('.py'):
                content = state.files[filename]
                lines = content.split('\n')
                new_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i]
                    new_lines.append(line)
                    
                    # Check if function/class needs docstring
                    if (line.strip().startswith('def ') or 
                        line.strip().startswith('class ')):
                        # Check if next non-empty line is docstring
                        j = i + 1
                        while j < len(lines) and not lines[j].strip():
                            new_lines.append(lines[j])
                            j += 1
                        
                        if j < len(lines):
                            next_line = lines[j].strip()
                            if not (next_line.startswith('"""') or 
                                    next_line.startswith("'''")):
                                # Add placeholder docstring
                                indent = len(line) - len(line.lstrip()) + 4
                                new_lines.append(' ' * indent + '"""TODO: Add documentation."""')
                    i += 1
                
                state.files[filename] = '\n'.join(new_lines)
        
        return state
    
    @staticmethod
    def decoherence(state: BuildState) -> BuildState:
        """
        DECOHERENCE ÷ — Break/decouple
        
        BUILD MEANING:
        - Reduce coupling
        - Add independence
        - Inject randomness for testing
        - Slow down (reduce z velocity)
        """
        # Reduce coupling
        state.coupling_score = max(0.0, state.coupling_score - 0.15)
        
        # Slow z velocity
        state.z_velocity *= 0.7
        
        # Add independence markers (imports at top of each file)
        for filename in state.files:
            if filename.endswith('.py'):
                content = state.files[filename]
                if 'from __future__' not in content:
                    state.files[filename] = (
                        "from __future__ import annotations\n\n" + content
                    )
        
        return state
    
    @staticmethod
    def group(state: BuildState) -> BuildState:
        """
        GROUP + — Cluster/organize
        
        BUILD MEANING:
        - Group related functions
        - Organize by domain
        - Create packages
        - Cluster tests
        """
        # Organize files into packages if not already
        root_files = [f for f in state.files if '/' not in f and f.endswith('.py')]
        
        if len(root_files) > 3:
            # Create package structure
            task_name = getattr(state.task, 'name', 'pkg')
            
            for filename in root_files:
                if filename not in ['__init__.py', 'main.py', 'setup.py']:
                    content = state.files[filename]
                    new_path = f"{task_name}/{filename}"
                    state.files[new_path] = content
                    del state.files[filename]
            
            # Create package __init__
            if f"{task_name}/__init__.py" not in state.files:
                state.files[f"{task_name}/__init__.py"] = f'"""{task_name} package."""\n'
        
        # Group tests together
        test_files = {k: v for k, v in state.files.items() if 'test' in k.lower()}
        if test_files and 'tests/' not in str(list(test_files.keys())):
            for filename, content in list(test_files.items()):
                if '/' not in filename:
                    new_path = f"tests/{filename}"
                    state.files[new_path] = content
                    del state.files[filename]
            
            if 'tests/__init__.py' not in state.files:
                state.files['tests/__init__.py'] = ''
        
        return state
    
    @staticmethod
    def separate(state: BuildState) -> BuildState:
        """
        SEPARATE − — Split/isolate
        
        BUILD MEANING:
        - Extract methods
        - Split large modules
        - Isolate concerns
        - Break dependencies
        """
        # Split large files
        for filename in list(state.files.keys()):
            if filename.endswith('.py'):
                content = state.files[filename]
                lines = content.split('\n')
                
                if len(lines) > 300:
                    # File is too large, split by class/function groups
                    chunks = OperatorActions._split_by_definitions(content)
                    
                    if len(chunks) > 1:
                        base = filename.replace('.py', '')
                        for i, chunk in enumerate(chunks):
                            new_name = f"{base}_part{i+1}.py"
                            state.files[new_name] = chunk
                        del state.files[filename]
        
        # Reduce coupling score
        state.coupling_score = max(0.0, state.coupling_score - 0.1)
        
        return state
    
    @staticmethod
    def _split_by_definitions(content: str) -> List[str]:
        """Split content by class/function definitions."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        imports = []
        
        # Collect imports
        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        
        # Split by top-level definitions
        in_definition = False
        for line in lines:
            if line.startswith('class ') or line.startswith('def '):
                if current_chunk and in_definition:
                    chunks.append('\n'.join(imports + [''] + current_chunk))
                    current_chunk = []
                in_definition = True
            
            if in_definition:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append('\n'.join(imports + [''] + current_chunk))
        
        return chunks if len(chunks) > 1 else [content]


def select_operator(
    state: BuildState, 
    available: List[APLOperator]
) -> Optional[APLOperator]:
    """
    Select best operator based on current state.
    
    DECISION TREE:
    1. Coherence < 0.3 → BOUNDARY (emergency)
    2. Coherence < 0.5 → SEPARATE (isolate problems)
    3. Coherence < 0.7 → GROUP (consolidate)
    4. z < target_z → AMPLIFY (push forward)
    5. coupling > 0.8 → DECOHERENCE (reduce coupling)
    6. similar modules → FUSION (merge)
    7. default → BOUNDARY (maintain)
    """
    if not available:
        return None
    
    # Convert to APLOperator if strings
    ops = []
    for op in available:
        if isinstance(op, str):
            for apl_op in APLOperator:
                if apl_op.value == op:
                    ops.append(apl_op)
                    break
        else:
            ops.append(op)
    
    if state.coherence < 0.3 and APLOperator.BOUNDARY in ops:
        return APLOperator.BOUNDARY
    
    if state.coherence < 0.5 and APLOperator.SEPARATE in ops:
        return APLOperator.SEPARATE
    
    if state.coherence < 0.7 and APLOperator.GROUP in ops:
        return APLOperator.GROUP
    
    if state.z < state.target_z * 0.9 and APLOperator.AMPLIFY in ops:
        return APLOperator.AMPLIFY
    
    if state.coupling_score > 0.8 and APLOperator.DECOHERENCE in ops:
        return APLOperator.DECOHERENCE
    
    if state.has_similar_modules and APLOperator.FUSION in ops:
        return APLOperator.FUSION
    
    if APLOperator.BOUNDARY in ops:
        return APLOperator.BOUNDARY
    
    return ops[0] if ops else None


def describe_operator(op: APLOperator) -> Dict[str, str]:
    """Get description of operator's build meaning."""
    descriptions = {
        APLOperator.BOUNDARY: {
            "symbol": "()",
            "name": "Boundary",
            "meaning": "Define/protect coherence",
            "action": "Create interfaces, define contracts, reset to stable state"
        },
        APLOperator.FUSION: {
            "symbol": "×",
            "name": "Fusion", 
            "meaning": "Merge/combine",
            "action": "Merge modules, combine duplicate code, unify functions"
        },
        APLOperator.AMPLIFY: {
            "symbol": "^",
            "name": "Amplify",
            "meaning": "Boost/strengthen",
            "action": "Add tests, strengthen validation, expand docs, push z"
        },
        APLOperator.DECOHERENCE: {
            "symbol": "÷",
            "name": "Decoherence",
            "meaning": "Break/decouple",
            "action": "Reduce coupling, add independence, slow down"
        },
        APLOperator.GROUP: {
            "symbol": "+",
            "name": "Group",
            "meaning": "Cluster/organize",
            "action": "Group functions, organize by domain, create packages"
        },
        APLOperator.SEPARATE: {
            "symbol": "−",
            "name": "Separate",
            "meaning": "Split/isolate",
            "action": "Extract methods, split modules, isolate concerns"
        }
    }
    return descriptions.get(op, {})


if __name__ == "__main__":
    # Demo
    print("Operator-Action Mapping")
    print("=" * 50)
    
    for op in APLOperator:
        desc = describe_operator(op)
        print(f"\n{desc['symbol']} {desc['name']}")
        print(f"  Meaning: {desc['meaning']}")
        print(f"  Action: {desc['action']}")
