"""
Phase 15: Move Refactoring
===========================

Detect and execute move refactorings:
- Move Method: Method uses another class more than its own (Feature Envy)
- Move Class: Class in wrong module based on dependencies
- Move Function: Standalone function better suited elsewhere

Detection based on:
- Feature Envy: Method accesses another class's data more than its own
- Coupling analysis: Method tightly coupled to foreign class
- Cohesion analysis: Method doesn't fit with class's responsibility
"""

import ast
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MoveSpec:
    """Specification for a move refactoring"""
    item_type: str          # 'method', 'function', 'class'
    item_name: str
    source_location: str    # module.Class or module
    target_location: str    # suggested destination
    reason: str
    confidence: float
    coupling_score: float   # How coupled to target (higher = stronger case)
    cohesion_loss: float    # How much cohesion lost if removed from source


@dataclass
class FeatureEnvyInfo:
    """Information about feature envy in a method"""
    method_name: str
    class_name: str
    file_path: str
    own_accesses: int       # Accesses to self
    foreign_accesses: Dict[str, int]  # class_name -> access_count
    envied_class: str       # Class most accessed
    envy_ratio: float       # foreign / (own + foreign)


@dataclass
class MoveResult:
    """Result of a move operation"""
    spec: MoveSpec
    success: bool
    files_modified: List[str]
    imports_updated: List[str]
    error_message: Optional[str] = None


@dataclass
class ClassCoupling:
    """Coupling information for a class"""
    class_name: str
    file_path: str
    depends_on: Dict[str, int]      # class -> number of references
    depended_by: Dict[str, int]     # class -> number of references
    total_coupling: int


# =============================================================================
# FEATURE ENVY DETECTOR
# =============================================================================

class AccessVisitor(ast.NodeVisitor):
    """Track attribute accesses in a method"""

    def __init__(self):
        self.self_accesses: Set[str] = set()
        self.foreign_accesses: Dict[str, Set[str]] = defaultdict(set)
        self.current_target: Optional[str] = None

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            target = node.value.id
            if target == 'self':
                self.self_accesses.add(node.attr)
            else:
                self.foreign_accesses[target].add(node.attr)
        elif isinstance(node.value, ast.Attribute):
            # Handle chained access like self.other.attr
            self.generic_visit(node.value)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Track method calls
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                target = node.func.value.id
                if target == 'self':
                    self.self_accesses.add(node.func.attr)
                else:
                    self.foreign_accesses[target].add(node.func.attr)
        self.generic_visit(node)


class FeatureEnvyDetector:
    """Detect methods that use another class more than their own"""

    def __init__(self, envy_threshold: float = 0.5):
        """
        Args:
            envy_threshold: Ratio of foreign/total accesses to flag as envy
        """
        self.envy_threshold = envy_threshold

    def detect(self, source_code: str, file_path: str = '<string>') -> List[FeatureEnvyInfo]:
        """Detect feature envy in all methods"""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

        results = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith('__'):
                            continue  # Skip dunder methods

                        envy = self._analyze_method(item, class_name, file_path)
                        if envy and envy.envy_ratio >= self.envy_threshold:
                            results.append(envy)

        return results

    def _analyze_method(self, method: ast.FunctionDef, class_name: str,
                       file_path: str) -> Optional[FeatureEnvyInfo]:
        """Analyze a single method for feature envy"""
        visitor = AccessVisitor()
        visitor.visit(method)

        own_count = len(visitor.self_accesses)
        foreign_counts = {k: len(v) for k, v in visitor.foreign_accesses.items()}

        if not foreign_counts:
            return None

        # Find most envied class
        envied_class = max(foreign_counts.keys(), key=lambda k: foreign_counts[k])
        envied_count = foreign_counts[envied_class]

        total = own_count + envied_count
        if total == 0:
            return None

        ratio = envied_count / total

        return FeatureEnvyInfo(
            method_name=method.name,
            class_name=class_name,
            file_path=file_path,
            own_accesses=own_count,
            foreign_accesses=foreign_counts,
            envied_class=envied_class,
            envy_ratio=ratio
        )

    def to_move_specs(self, envy_list: List[FeatureEnvyInfo]) -> List[MoveSpec]:
        """Convert feature envy detections to move specifications"""
        specs = []
        for envy in envy_list:
            specs.append(MoveSpec(
                item_type='method',
                item_name=envy.method_name,
                source_location=f"{envy.file_path}:{envy.class_name}",
                target_location=envy.envied_class,
                reason=f"Method accesses {envy.envied_class} {envy.foreign_accesses[envy.envied_class]} times "
                       f"vs own class {envy.own_accesses} times",
                confidence=min(envy.envy_ratio, 0.95),
                coupling_score=envy.foreign_accesses[envy.envied_class],
                cohesion_loss=envy.own_accesses
            ))
        return specs


# =============================================================================
# MOVE METHOD DETECTOR
# =============================================================================

class MoveMethodDetector:
    """Detect methods that should be moved based on various heuristics"""

    def __init__(self):
        self.feature_envy_detector = FeatureEnvyDetector()

    def detect(self, source_code: str, file_path: str = '<string>') -> List[MoveSpec]:
        """Detect methods that should be moved"""
        specs = []

        # Feature envy detection
        envy_results = self.feature_envy_detector.detect(source_code, file_path)
        specs.extend(self.feature_envy_detector.to_move_specs(envy_results))

        # Coupling-based detection
        coupling_specs = self._detect_by_coupling(source_code, file_path)
        specs.extend(coupling_specs)

        # Deduplicate
        seen = set()
        unique = []
        for spec in specs:
            key = (spec.item_name, spec.source_location)
            if key not in seen:
                seen.add(key)
                unique.append(spec)

        return unique

    def _detect_by_coupling(self, source_code: str, file_path: str) -> List[MoveSpec]:
        """Detect methods based on coupling analysis"""
        # This analyzes parameter types and return types
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

        specs = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        spec = self._analyze_method_coupling(item, class_name, file_path)
                        if spec:
                            specs.append(spec)

        return specs

    def _analyze_method_coupling(self, method: ast.FunctionDef, class_name: str,
                                  file_path: str) -> Optional[MoveSpec]:
        """Analyze method's coupling to determine if move is beneficial"""
        # Check parameters for foreign type hints
        foreign_types = []

        for arg in method.args.args:
            if arg.annotation:
                type_name = self._extract_type_name(arg.annotation)
                if type_name and type_name != class_name and type_name not in ('self', 'cls'):
                    foreign_types.append(type_name)

        # If method takes mostly foreign types, might belong elsewhere
        if len(foreign_types) >= 2:
            # Find most common foreign type
            from collections import Counter
            common = Counter(foreign_types).most_common(1)
            if common:
                target = common[0][0]
                return MoveSpec(
                    item_type='method',
                    item_name=method.name,
                    source_location=f"{file_path}:{class_name}",
                    target_location=target,
                    reason=f"Method parameters suggest affinity with {target}",
                    confidence=0.5,
                    coupling_score=len(foreign_types),
                    cohesion_loss=1.0
                )

        return None

    def _extract_type_name(self, annotation: ast.AST) -> Optional[str]:
        """Extract type name from annotation"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id
        elif isinstance(annotation, ast.Attribute):
            return annotation.attr
        return None


# =============================================================================
# MOVE CLASS DETECTOR
# =============================================================================

class ImportAnalyzer(ast.NodeVisitor):
    """Analyze imports in a file"""

    def __init__(self):
        self.imports: Dict[str, str] = {}  # name -> module
        self.from_imports: Dict[str, str] = {}  # name -> module

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ''
        for alias in node.names:
            name = alias.asname or alias.name
            self.from_imports[name] = module
        self.generic_visit(node)


class MoveClassDetector:
    """Detect classes that should be moved to different modules"""

    def __init__(self):
        pass

    def detect(self, source_code: str, file_path: str = '<string>') -> List[MoveSpec]:
        """Detect classes that might be better placed elsewhere"""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

        # Analyze imports
        import_analyzer = ImportAnalyzer()
        import_analyzer.visit(tree)

        specs = []
        module_name = Path(file_path).stem

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                spec = self._analyze_class(node, module_name, import_analyzer, file_path)
                if spec:
                    specs.append(spec)

        return specs

    def _analyze_class(self, class_node: ast.ClassDef, module_name: str,
                       imports: ImportAnalyzer, file_path: str) -> Optional[MoveSpec]:
        """Analyze a class for potential relocation"""
        # Check bases - if inheriting from another module extensively
        foreign_bases = []
        for base in class_node.bases:
            base_name = self._get_base_name(base)
            if base_name and base_name in imports.from_imports:
                foreign_bases.append(imports.from_imports[base_name])

        # If class inherits primarily from one foreign module, might belong there
        if foreign_bases:
            from collections import Counter
            common = Counter(foreign_bases).most_common(1)
            if common and common[0][1] >= 1:
                target_module = common[0][0]
                return MoveSpec(
                    item_type='class',
                    item_name=class_node.name,
                    source_location=file_path,
                    target_location=target_module,
                    reason=f"Class inherits from {target_module}, consider colocating",
                    confidence=0.4,
                    coupling_score=common[0][1],
                    cohesion_loss=0.5
                )

        # Check naming - if class name suggests different module
        class_name = class_node.name.lower()
        if module_name not in class_name and len(class_name) > 5:
            # Try to infer better module from name
            suggested = self._suggest_module_from_name(class_node.name)
            if suggested and suggested != module_name:
                return MoveSpec(
                    item_type='class',
                    item_name=class_node.name,
                    source_location=file_path,
                    target_location=suggested,
                    reason=f"Class name suggests affinity with {suggested} module",
                    confidence=0.3,
                    coupling_score=0.5,
                    cohesion_loss=0.3
                )

        return None

    def _get_base_name(self, base: ast.AST) -> Optional[str]:
        """Get name of base class"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
        return None

    def _suggest_module_from_name(self, class_name: str) -> Optional[str]:
        """Suggest module based on class name patterns"""
        name_lower = class_name.lower()

        # Common patterns
        patterns = {
            'error': 'exceptions',
            'exception': 'exceptions',
            'handler': 'handlers',
            'manager': 'managers',
            'factory': 'factories',
            'builder': 'builders',
            'adapter': 'adapters',
            'service': 'services',
            'repository': 'repositories',
            'controller': 'controllers',
            'view': 'views',
            'model': 'models',
            'test': 'tests',
            'mock': 'mocks',
            'util': 'utils',
            'helper': 'helpers',
        }

        for pattern, module in patterns.items():
            if pattern in name_lower:
                return module

        return None


# =============================================================================
# MOVE TRANSFORMER
# =============================================================================

class MoveMethodTransformer:
    """Execute move method refactoring"""

    def __init__(self):
        pass

    def move(self, spec: MoveSpec, source_code: str, target_code: str,
             dry_run: bool = True) -> MoveResult:
        """
        Move a method from source to target.

        Note: This is a simplified implementation. Full implementation would:
        1. Parse both source and target
        2. Extract method from source class
        3. Add method to target class
        4. Update all call sites
        5. Add delegation if needed
        """
        if spec.item_type != 'method':
            return MoveResult(
                spec=spec,
                success=False,
                files_modified=[],
                imports_updated=[],
                error_message="Only method moves are supported"
            )

        try:
            source_tree = ast.parse(source_code)
            target_tree = ast.parse(target_code)
        except SyntaxError as e:
            return MoveResult(
                spec=spec,
                success=False,
                files_modified=[],
                imports_updated=[],
                error_message=f"Parse error: {e}"
            )

        # Find the method in source
        method_node = self._find_method(source_tree, spec.item_name, spec.source_location)
        if not method_node:
            return MoveResult(
                spec=spec,
                success=False,
                files_modified=[],
                imports_updated=[],
                error_message=f"Method {spec.item_name} not found in source"
            )

        if dry_run:
            return MoveResult(
                spec=spec,
                success=True,
                files_modified=[spec.source_location.split(':')[0], spec.target_location],
                imports_updated=[],
                error_message=None
            )

        # TODO: Implement actual move transformation
        return MoveResult(
            spec=spec,
            success=False,
            files_modified=[],
            imports_updated=[],
            error_message="Full move implementation pending"
        )

    def _find_method(self, tree: ast.AST, method_name: str,
                     source_location: str) -> Optional[ast.FunctionDef]:
        """Find a method in the AST"""
        # Parse source_location to get class name
        if ':' in source_location:
            class_name = source_location.split(':')[1]
        else:
            class_name = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if class_name and node.name != class_name:
                    continue
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            return item
        return None

    def generate_delegation(self, spec: MoveSpec) -> str:
        """Generate delegation code for moved method"""
        return f'''
    def {spec.item_name}(self, *args, **kwargs):
        """Delegated to {spec.target_location}"""
        return self.{spec.target_location.lower()}.{spec.item_name}(*args, **kwargs)
'''


# =============================================================================
# COUPLING ANALYZER
# =============================================================================

class CouplingAnalyzer:
    """Analyze coupling between classes across codebase"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.class_files: Dict[str, str] = {}  # class_name -> file_path
        self.class_couplings: Dict[str, ClassCoupling] = {}

    def analyze(self, exclude_dirs: List[str] = None) -> Dict[str, ClassCoupling]:
        """Analyze coupling across all classes"""
        exclude = set(exclude_dirs or ['__pycache__', '.git', '.grace_backups'])

        # First pass: collect all classes
        for py_file in self.root_dir.rglob('*.py'):
            if any(ex in py_file.parts for ex in exclude):
                continue
            self._collect_classes(py_file)

        # Second pass: analyze dependencies
        for py_file in self.root_dir.rglob('*.py'):
            if any(ex in py_file.parts for ex in exclude):
                continue
            self._analyze_file_coupling(py_file)

        return self.class_couplings

    def _collect_classes(self, file_path: Path) -> None:
        """Collect class definitions from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
            tree = ast.parse(source)
        except:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.class_files[node.name] = str(file_path)
                self.class_couplings[node.name] = ClassCoupling(
                    class_name=node.name,
                    file_path=str(file_path),
                    depends_on={},
                    depended_by={},
                    total_coupling=0
                )

    def _analyze_file_coupling(self, file_path: Path) -> None:
        """Analyze coupling in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
            tree = ast.parse(source)
        except:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                if class_name not in self.class_couplings:
                    continue

                # Analyze references to other classes
                references = self._count_class_references(node)

                for ref_class, count in references.items():
                    if ref_class in self.class_couplings:
                        # Update depends_on
                        self.class_couplings[class_name].depends_on[ref_class] = count
                        # Update depended_by
                        self.class_couplings[ref_class].depended_by[class_name] = count

        # Calculate total coupling
        for coupling in self.class_couplings.values():
            coupling.total_coupling = (
                sum(coupling.depends_on.values()) +
                sum(coupling.depended_by.values())
            )

    def _count_class_references(self, class_node: ast.ClassDef) -> Dict[str, int]:
        """Count references to other classes within a class"""
        references: Dict[str, int] = defaultdict(int)

        for node in ast.walk(class_node):
            if isinstance(node, ast.Name):
                if node.id in self.class_files and node.id != class_node.name:
                    references[node.id] += 1
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id in self.class_files:
                        references[node.value.id] += 1

        return dict(references)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_feature_envy(source_code: str, file_path: str = '<string>',
                        threshold: float = 0.5) -> List[FeatureEnvyInfo]:
    """Detect feature envy in source code"""
    detector = FeatureEnvyDetector(threshold)
    return detector.detect(source_code, file_path)


def detect_move_opportunities(source_code: str, file_path: str = '<string>') -> List[MoveSpec]:
    """Detect all move opportunities in source code"""
    results = []

    method_detector = MoveMethodDetector()
    results.extend(method_detector.detect(source_code, file_path))

    class_detector = MoveClassDetector()
    results.extend(class_detector.detect(source_code, file_path))

    return results


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate move refactoring detection"""
    print("=" * 70)
    print("PHASE 15: MOVE REFACTORING - DEMONSTRATION")
    print("=" * 70)
    print()

    sample_code = '''
class Order:
    def __init__(self, customer):
        self.customer = customer
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def get_customer_name(self):
        # Feature envy - uses customer more than self
        return self.customer.first_name + " " + self.customer.last_name

    def get_customer_address(self):
        # Feature envy - heavily uses customer
        return (self.customer.street + ", " +
                self.customer.city + ", " +
                self.customer.state + " " +
                self.customer.zip_code)

    def calculate_customer_discount(self):
        # Feature envy - all about customer
        if self.customer.loyalty_years > 5:
            return self.customer.base_discount * 1.5
        elif self.customer.loyalty_years > 2:
            return self.customer.base_discount * 1.2
        return self.customer.base_discount


class Customer:
    def __init__(self):
        self.first_name = ""
        self.last_name = ""
        self.street = ""
        self.city = ""
        self.state = ""
        self.zip_code = ""
        self.loyalty_years = 0
        self.base_discount = 0.1
'''

    print("Analyzing sample code for feature envy...")
    print()

    # Detect feature envy
    envy_detector = FeatureEnvyDetector(threshold=0.5)
    envy_results = envy_detector.detect(sample_code, 'order.py')

    print("Feature Envy Detected:")
    print("-" * 50)
    for envy in envy_results:
        print(f"  Method: {envy.class_name}.{envy.method_name}")
        print(f"    Own accesses: {envy.own_accesses}")
        print(f"    Foreign accesses to {envy.envied_class}: {envy.foreign_accesses[envy.envied_class]}")
        print(f"    Envy ratio: {envy.envy_ratio:.2%}")
        print()

    # Convert to move specs
    specs = envy_detector.to_move_specs(envy_results)

    print("Move Suggestions:")
    print("-" * 50)
    for spec in specs:
        print(f"  Move {spec.item_name}")
        print(f"    From: {spec.source_location}")
        print(f"    To: {spec.target_location}")
        print(f"    Confidence: {spec.confidence:.2%}")
        print(f"    Reason: {spec.reason}")
        print()

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    demo()
