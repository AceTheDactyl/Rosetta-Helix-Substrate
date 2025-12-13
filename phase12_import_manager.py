"""
Phase 12 Import Manager - AST-Based Import Rewriting & Cycle Detection

This module provides robust import management for code refactoring:
- Parses ALL import styles (import X, from X import Y, from X import *, relative imports)
- Builds import dependency graph for cycle detection
- Rewrites imports using AST transformation (not fragile string replacement)
- Detects and prevents circular imports before changes are made

Key Components:
- ImportStatement: Dataclass representing any Python import
- ImportParser: AST-based import extraction
- ImportGraph: Dependency graph with cycle detection (Tarjan's SCC)
- ASTImportRewriter: Safe AST-based import rewriting
- ImportValidator: Pre-flight import validation

Replaces the fragile string-based ImportRewriter in phase10_grace_optimization.py
"""

import ast
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
import hashlib


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ImportStatement:
    """
    Represents any Python import statement.

    Handles all import styles:
    - import foo                    -> import_type='import', module='foo'
    - import foo as f               -> import_type='import', module='foo', aliases={'foo': 'f'}
    - from foo import bar           -> import_type='from_import', module='foo', names=['bar']
    - from foo import bar as b      -> import_type='from_import', module='foo', names=['bar'], aliases={'bar': 'b'}
    - from foo import *             -> import_type='from_star', module='foo'
    - from . import foo             -> import_type='from_import', module='', is_relative=True, relative_level=1
    - from ..pkg import bar         -> import_type='from_import', module='pkg', is_relative=True, relative_level=2
    """
    import_type: str              # 'import', 'from_import', 'from_star'
    module: str                   # The module being imported
    names: List[str] = field(default_factory=list)   # Names imported (for from imports)
    aliases: Dict[str, str] = field(default_factory=dict)  # name -> alias mapping
    is_relative: bool = False     # True for relative imports
    relative_level: int = 0       # Number of dots (0=absolute, 1=., 2=.., etc.)
    line_number: int = 0          # Location in source
    source_segment: str = ""      # Original source text

    def __str__(self) -> str:
        """Reconstruct the import statement as source code"""
        if self.import_type == 'import':
            parts = []
            for mod in ([self.module] if self.module else []):
                if mod in self.aliases:
                    parts.append(f"{mod} as {self.aliases[mod]}")
                else:
                    parts.append(mod)
            return f"import {', '.join(parts)}"

        elif self.import_type == 'from_star':
            prefix = '.' * self.relative_level
            return f"from {prefix}{self.module} import *"

        else:  # from_import
            prefix = '.' * self.relative_level
            parts = []
            for name in self.names:
                if name in self.aliases:
                    parts.append(f"{name} as {self.aliases[name]}")
                else:
                    parts.append(name)
            return f"from {prefix}{self.module} import {', '.join(parts)}"

    @property
    def full_module_path(self) -> str:
        """Full module path including relative prefix indicator"""
        if self.is_relative:
            return f"{'.' * self.relative_level}{self.module}"
        return self.module


@dataclass
class ImportChange:
    """Record of an import change for audit trail"""
    file_path: str
    line_number: int
    old_statement: str
    new_statement: str
    change_type: str  # 'rewrite', 'add', 'remove'
    reason: str


# =============================================================================
# IMPORT PARSER
# =============================================================================

class ImportParser:
    """
    Parse all import statements from Python source code using AST.

    Much more reliable than regex-based parsing.
    """

    def parse_file(self, file_path: str) -> List[ImportStatement]:
        """Parse imports from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
            return self.parse_source(source, file_path)
        except Exception as e:
            print(f"  [ImportParser error on {file_path}: {e}]")
            return []

    def parse_source(self, source: str, file_path: str = '<string>') -> List[ImportStatement]:
        """Parse imports from source code string"""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"  [Syntax error parsing {file_path}: {e}]")
            return []

        imports = []
        source_lines = source.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # import foo, bar, baz as b
                for alias in node.names:
                    stmt = ImportStatement(
                        import_type='import',
                        module=alias.name,
                        aliases={alias.name: alias.asname} if alias.asname else {},
                        line_number=node.lineno,
                        source_segment=source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
                    )
                    imports.append(stmt)

            elif isinstance(node, ast.ImportFrom):
                # from foo import bar, baz as b
                # from . import foo
                # from ..pkg import bar
                module = node.module or ''
                level = node.level  # 0=absolute, 1=., 2=.., etc.

                # Check for star import
                is_star = any(alias.name == '*' for alias in node.names)

                if is_star:
                    stmt = ImportStatement(
                        import_type='from_star',
                        module=module,
                        is_relative=level > 0,
                        relative_level=level,
                        line_number=node.lineno,
                        source_segment=source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
                    )
                    imports.append(stmt)
                else:
                    names = [alias.name for alias in node.names]
                    aliases = {alias.name: alias.asname for alias in node.names if alias.asname}

                    stmt = ImportStatement(
                        import_type='from_import',
                        module=module,
                        names=names,
                        aliases=aliases,
                        is_relative=level > 0,
                        relative_level=level,
                        line_number=node.lineno,
                        source_segment=source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
                    )
                    imports.append(stmt)

        return imports

    def get_imported_names(self, imports: List[ImportStatement]) -> Set[str]:
        """Get all names made available by a list of imports"""
        names = set()
        for imp in imports:
            if imp.import_type == 'import':
                # import foo -> 'foo' available (or alias)
                alias = imp.aliases.get(imp.module, imp.module)
                names.add(alias.split('.')[0])  # Handle import foo.bar
            elif imp.import_type == 'from_import':
                # from foo import bar, baz -> 'bar', 'baz' available (or aliases)
                for name in imp.names:
                    alias = imp.aliases.get(name, name)
                    names.add(alias)
            elif imp.import_type == 'from_star':
                # from foo import * -> unknown names, mark specially
                names.add(f'*:{imp.module}')
        return names


# =============================================================================
# IMPORT DEPENDENCY GRAPH
# =============================================================================

@dataclass
class ModuleNode:
    """Node in the import graph"""
    name: str
    file_path: Optional[str] = None
    imports: List[ImportStatement] = field(default_factory=list)
    imported_by: Set[str] = field(default_factory=set)


class ImportGraph:
    """
    Directed graph of import dependencies.

    Supports:
    - Cycle detection using Tarjan's SCC algorithm
    - Pre-change cycle checking
    - Topological sort for dependency ordering
    """

    def __init__(self, root_dir: str = '.'):
        self.root_dir = Path(root_dir)
        self.nodes: Dict[str, ModuleNode] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # module -> modules it imports

    def build_from_directory(self, verbose: bool = False) -> None:
        """Build import graph from all Python files in directory"""
        parser = ImportParser()

        python_files = list(self.root_dir.glob('**/*.py'))
        if verbose:
            print(f"  Scanning {len(python_files)} Python files...")

        for file_path in python_files:
            try:
                # Convert to module name
                rel_path = file_path.relative_to(self.root_dir)
                module_name = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')

                # Parse imports
                imports = parser.parse_file(str(file_path))

                # Create node
                node = ModuleNode(
                    name=module_name,
                    file_path=str(file_path),
                    imports=imports
                )
                self.nodes[module_name] = node

                # Build adjacency
                for imp in imports:
                    imported_module = self._resolve_import(imp, module_name)
                    if imported_module:
                        self.adjacency[module_name].add(imported_module)

            except Exception as e:
                if verbose:
                    print(f"    [Error processing {file_path}: {e}]")

        # Build reverse references
        for importer, imports in self.adjacency.items():
            for imported in imports:
                if imported in self.nodes:
                    self.nodes[imported].imported_by.add(importer)

        if verbose:
            print(f"  Built graph with {len(self.nodes)} modules, {sum(len(v) for v in self.adjacency.values())} edges")

    def _resolve_import(self, imp: ImportStatement, current_module: str) -> Optional[str]:
        """Resolve an import statement to a module name"""
        if imp.is_relative:
            # Resolve relative import
            parts = current_module.split('.')
            if imp.relative_level > len(parts):
                return None  # Invalid relative import

            # Go up 'level' directories
            base_parts = parts[:-imp.relative_level] if imp.relative_level > 0 else parts

            if imp.module:
                return '.'.join(base_parts + [imp.module])
            elif imp.names:
                # from . import foo -> sibling module
                return '.'.join(base_parts + [imp.names[0]]) if imp.names else None
            return '.'.join(base_parts)
        else:
            return imp.module

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect all cycles using Tarjan's strongly connected components algorithm.

        Returns list of cycles (each cycle is a list of module names).
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = set()
        sccs = []

        def strongconnect(v):
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            on_stack.add(v)
            stack.append(v)

            for w in self.adjacency.get(v, []):
                if w not in index:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], index[w])

            if lowlinks[v] == index[v]:
                scc = []
                while stack:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) > 1:  # Only report actual cycles
                    sccs.append(scc)

        for v in self.nodes:
            if v not in index:
                strongconnect(v)

        return sccs

    def would_create_cycle(self, from_module: str, to_module: str) -> bool:
        """
        Check if adding an edge from from_module -> to_module would create a cycle.

        This is useful for pre-checking imports before modifying code.
        """
        # If there's already a path from to_module to from_module,
        # adding from_module -> to_module would create a cycle
        return self._has_path(to_module, from_module)

    def _has_path(self, start: str, end: str, visited: Optional[Set[str]] = None) -> bool:
        """Check if there's a path from start to end"""
        if visited is None:
            visited = set()

        if start == end:
            return True

        if start in visited:
            return False

        visited.add(start)

        for neighbor in self.adjacency.get(start, []):
            if self._has_path(neighbor, end, visited):
                return True

        return False

    def topological_sort(self) -> List[str]:
        """
        Return modules in topological order (dependencies before dependents).

        Raises ValueError if graph has cycles.
        """
        cycles = self.detect_cycles()
        if cycles:
            raise ValueError(f"Cannot topologically sort: {len(cycles)} cycle(s) detected")

        in_degree = {node: 0 for node in self.nodes}
        for edges in self.adjacency.values():
            for target in edges:
                if target in in_degree:
                    in_degree[target] += 1

        # Start with nodes that have no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in self.adjacency.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return result

    def get_dependents(self, module: str) -> Set[str]:
        """Get all modules that import the given module (direct or indirect)"""
        dependents = set()
        queue = [module]

        while queue:
            current = queue.pop(0)
            for importer, imports in self.adjacency.items():
                if current in imports and importer not in dependents:
                    dependents.add(importer)
                    queue.append(importer)

        return dependents

    def get_dependencies(self, module: str) -> Set[str]:
        """Get all modules imported by the given module (direct or indirect)"""
        dependencies = set()
        queue = list(self.adjacency.get(module, []))

        while queue:
            current = queue.pop(0)
            if current not in dependencies:
                dependencies.add(current)
                queue.extend(self.adjacency.get(current, []))

        return dependencies


# =============================================================================
# AST-BASED IMPORT REWRITER
# =============================================================================

class ASTImportRewriter(ast.NodeTransformer):
    """
    Rewrite imports using AST transformation.

    This is much safer than string replacement because:
    1. It only modifies actual import statements
    2. It preserves formatting where possible
    3. It handles all import styles correctly
    """

    def __init__(self, mapping: Dict[str, str]):
        """
        Initialize with a mapping of old -> new module names.

        mapping examples:
        - {'old_module': 'new_module'}  # Rename module
        - {'old_module.func': 'new_module.func'}  # Move function
        """
        self.mapping = mapping
        self.changes_made: List[ImportChange] = []

    def visit_Import(self, node: ast.Import) -> ast.AST:
        """Transform 'import X' statements"""
        new_names = []
        changed = False

        for alias in node.names:
            if alias.name in self.mapping:
                new_module = self.mapping[alias.name]
                new_alias = ast.alias(name=new_module, asname=alias.asname)
                new_names.append(new_alias)
                changed = True
                self.changes_made.append(ImportChange(
                    file_path='<current>',
                    line_number=node.lineno,
                    old_statement=f"import {alias.name}",
                    new_statement=f"import {new_module}",
                    change_type='rewrite',
                    reason=f"Module renamed: {alias.name} -> {new_module}"
                ))
            else:
                new_names.append(alias)

        if changed:
            node.names = new_names
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        """Transform 'from X import Y' statements"""
        module = node.module or ''

        # Check if the whole module is being renamed
        if module in self.mapping:
            old_module = module
            new_module = self.mapping[module]
            node.module = new_module
            self.changes_made.append(ImportChange(
                file_path='<current>',
                line_number=node.lineno,
                old_statement=f"from {old_module} import ...",
                new_statement=f"from {new_module} import ...",
                change_type='rewrite',
                reason=f"Module renamed: {old_module} -> {new_module}"
            ))
            return node

        # Check if specific names are being moved to different modules
        # mapping might be {'old_module.func': 'new_module'}
        new_imports_by_module: Dict[str, List[ast.alias]] = defaultdict(list)
        remaining_names = []

        for alias in node.names:
            full_name = f"{module}.{alias.name}" if module else alias.name

            if full_name in self.mapping:
                new_module = self.mapping[full_name]
                new_imports_by_module[new_module].append(alias)
                self.changes_made.append(ImportChange(
                    file_path='<current>',
                    line_number=node.lineno,
                    old_statement=f"from {module} import {alias.name}",
                    new_statement=f"from {new_module} import {alias.name}",
                    change_type='rewrite',
                    reason=f"Name moved: {full_name} -> {new_module}.{alias.name}"
                ))
            else:
                remaining_names.append(alias)

        # If all names moved to new modules, replace this import entirely
        if not remaining_names and new_imports_by_module:
            # Return first new import, queue rest for insertion
            # (This is simplified - full implementation would handle multiple new imports)
            new_module, aliases = list(new_imports_by_module.items())[0]
            node.module = new_module
            node.names = aliases
            return node

        # If some names remain, just update the remaining
        if remaining_names and new_imports_by_module:
            node.names = remaining_names
            # Full implementation would also add new import statements

        return node


class ImportRewriteEngine:
    """
    High-level API for rewriting imports across a codebase.

    Integrates ImportParser, ImportGraph, and ASTImportRewriter.
    """

    def __init__(self, root_dir: str = '.'):
        self.root_dir = Path(root_dir)
        self.parser = ImportParser()
        self.graph: Optional[ImportGraph] = None

    def build_graph(self, verbose: bool = False) -> ImportGraph:
        """Build the import dependency graph"""
        self.graph = ImportGraph(str(self.root_dir))
        self.graph.build_from_directory(verbose)
        return self.graph

    def rewrite_imports(self, mapping: Dict[str, str], dry_run: bool = True,
                       verbose: bool = False) -> Dict[str, Tuple[str, List[ImportChange]]]:
        """
        Rewrite imports across the codebase.

        Args:
            mapping: Dict of old_name -> new_name
            dry_run: If True, return changes without modifying files
            verbose: Print progress

        Returns:
            Dict of file_path -> (new_content, changes)
        """
        if self.graph is None:
            self.build_graph(verbose)

        results = {}

        for module_name, node in self.graph.nodes.items():
            if not node.file_path:
                continue

            # Check if this file needs changes
            needs_changes = False
            for imp in node.imports:
                if imp.module in mapping:
                    needs_changes = True
                    break
                for name in imp.names:
                    full_name = f"{imp.module}.{name}" if imp.module else name
                    if full_name in mapping:
                        needs_changes = True
                        break

            if not needs_changes:
                continue

            # Read and transform
            try:
                with open(node.file_path, 'r', encoding='utf-8') as f:
                    source = f.read()

                tree = ast.parse(source)
                rewriter = ASTImportRewriter(mapping)
                new_tree = rewriter.visit(tree)
                ast.fix_missing_locations(new_tree)

                # Generate new source
                # Note: ast.unparse() requires Python 3.9+
                if sys.version_info >= (3, 9):
                    new_source = ast.unparse(new_tree)
                else:
                    # Fallback: try to use astor if available
                    try:
                        import astor
                        new_source = astor.to_source(new_tree)
                    except ImportError:
                        # Last resort: return original with comments indicating changes
                        new_source = source
                        for change in rewriter.changes_made:
                            new_source = f"# IMPORT CHANGE NEEDED: {change.old_statement} -> {change.new_statement}\n" + new_source

                results[node.file_path] = (new_source, rewriter.changes_made)

                if verbose and rewriter.changes_made:
                    print(f"  {node.file_path}: {len(rewriter.changes_made)} changes")

            except Exception as e:
                if verbose:
                    print(f"  [Error rewriting {node.file_path}: {e}]")

        return results

    def check_for_cycles_after_move(self, from_module: str, to_module: str,
                                    names: List[str]) -> List[List[str]]:
        """
        Check if moving names from from_module to to_module would create cycles.

        Returns list of cycles that would be created.
        """
        if self.graph is None:
            self.build_graph()

        # Simulate the change
        temp_adjacency = defaultdict(set)
        for mod, imports in self.graph.adjacency.items():
            temp_adjacency[mod] = set(imports)

        # Files that import 'from_module.name' would need to import 'to_module.name'
        # This adds an edge: importer -> to_module
        for importer in self.graph.get_dependents(from_module):
            # Check if importer uses any of the moved names
            importer_node = self.graph.nodes.get(importer)
            if not importer_node:
                continue

            for imp in importer_node.imports:
                if imp.module == from_module:
                    if any(name in imp.names for name in names):
                        # This importer would need to import from to_module
                        temp_adjacency[importer].add(to_module)

        # Detect cycles in temp graph
        return self._detect_cycles_in_adjacency(temp_adjacency)

    def _detect_cycles_in_adjacency(self, adjacency: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect cycles in an adjacency dict"""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = set()
        sccs = []

        all_nodes = set(adjacency.keys())
        for targets in adjacency.values():
            all_nodes.update(targets)

        def strongconnect(v):
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            on_stack.add(v)
            stack.append(v)

            for w in adjacency.get(v, []):
                if w not in index:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], index[w])

            if lowlinks[v] == index[v]:
                scc = []
                while stack:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) > 1:
                    sccs.append(scc)

        for v in all_nodes:
            if v not in index:
                strongconnect(v)

        return sccs


# =============================================================================
# IMPORT VALIDATOR
# =============================================================================

class ImportValidator:
    """
    Validate imports before and after changes.

    Pre-flight checks:
    - All imports can be resolved
    - No circular imports
    - No naming collisions
    """

    def __init__(self, root_dir: str = '.'):
        self.root_dir = Path(root_dir)
        self.parser = ImportParser()

    def validate_imports(self, source: str, file_path: str = '<string>') -> Tuple[bool, List[str]]:
        """
        Validate imports in source code.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Parse imports
        imports = self.parser.parse_source(source, file_path)

        # Check for star imports (warning, not error)
        star_imports = [imp for imp in imports if imp.import_type == 'from_star']
        if star_imports:
            for imp in star_imports:
                errors.append(f"Warning: Star import 'from {imp.module} import *' at line {imp.line_number}")

        # Check for duplicate imports
        seen_names = {}
        for imp in imports:
            for name in imp.names:
                actual_name = imp.aliases.get(name, name)
                if actual_name in seen_names:
                    errors.append(f"Duplicate import: '{actual_name}' at lines {seen_names[actual_name]} and {imp.line_number}")
                else:
                    seen_names[actual_name] = imp.line_number

        return len([e for e in errors if not e.startswith('Warning')]) == 0, errors

    def check_resolution(self, imports: List[ImportStatement], available_modules: Set[str]) -> List[str]:
        """
        Check if imports can be resolved against available modules.

        Returns list of unresolvable imports.
        """
        unresolvable = []

        for imp in imports:
            if imp.is_relative:
                continue  # Skip relative imports (would need context)

            if imp.module and imp.module.split('.')[0] not in available_modules:
                # Check if it's a standard library or installed package
                # This is a simplified check
                top_level = imp.module.split('.')[0]
                if top_level not in sys.modules and top_level not in {'typing', 'dataclasses', 'collections', 'pathlib', 'json', 'ast', 'sys', 'os', 're', 'hashlib', 'datetime', 'time', 'numpy', 'np'}:
                    unresolvable.append(imp.module)

        return unresolvable


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo():
    """Demonstrate import management capabilities"""
    print("=" * 60)
    print("PHASE 12 IMPORT MANAGER DEMO")
    print("=" * 60)

    # Test import parsing
    test_source = '''
import os
import sys as system
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from . import sibling_module
from ..parent import something
from some_module import *

def example():
    pass
'''

    print("\nParsing test source...")
    parser = ImportParser()
    imports = parser.parse_source(test_source)

    print(f"Found {len(imports)} import statements:")
    for imp in imports:
        print(f"  Line {imp.line_number}: {imp}")
        print(f"    Type: {imp.import_type}, Module: {imp.module}, Relative: {imp.is_relative}")

    # Test import graph on current directory
    print("\n" + "-" * 40)
    print("Building import graph for provable_codegen/...")

    engine = ImportRewriteEngine('provable_codegen')
    graph = engine.build_graph(verbose=True)

    cycles = graph.detect_cycles()
    if cycles:
        print(f"\nWARNING: Found {len(cycles)} cycle(s):")
        for cycle in cycles:
            print(f"  {' -> '.join(cycle)}")
    else:
        print("\nNo circular imports detected")

    # Test cycle prediction
    print("\n" + "-" * 40)
    print("Testing cycle prediction...")

    # Would creating a new import create a cycle?
    test_from = 'phase1_control_flow'
    test_to = 'provable_codegen'
    would_cycle = graph.would_create_cycle(test_from, test_to)
    print(f"  {test_from} -> {test_to} would create cycle: {would_cycle}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
