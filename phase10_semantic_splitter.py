"""
Phase 10 Semantic Splitter - Semantic-Aware Module Splitting

This module replaces prefix-based clustering with behavioral analysis.
It analyzes what methods actually DO (what they read, write, call) rather
than just their names to determine semantic domains.

Key Components:
- MethodBehaviorExtractor: AST visitor to extract reads/writes/calls
- ClassBehaviorAnalyzer: Analyze all methods in a class
- LCOMCalculator: Compute LCOM1/LCOM4 cohesion metrics
- MethodCallGraph: Build method→method call relationships
- SemanticDomainDetector: Detect purpose from behavior patterns
- SemanticModuleSplitter: Generate split plan based on semantic clusters

Mathematical Foundation:
- LCOM (Lack of Cohesion of Methods) metrics for split decisions
- Connected component analysis for natural domain boundaries
- Fingerprint-based domain classification
"""

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MethodBehavior:
    """Behavioral analysis of a single method"""
    name: str
    calls_methods: Set[str] = field(default_factory=set)      # self.method() calls
    reads_attributes: Set[str] = field(default_factory=set)   # self.attr reads
    writes_attributes: Set[str] = field(default_factory=set)  # self.attr = ... writes
    external_calls: Set[str] = field(default_factory=set)     # module.func() calls
    local_variables: Set[str] = field(default_factory=set)    # local vars defined
    complexity: int = 1                                        # cyclomatic complexity
    line_start: int = 0
    line_end: int = 0
    is_private: bool = False
    is_dunder: bool = False
    docstring: Optional[str] = None

    @property
    def all_attributes(self) -> Set[str]:
        """All instance attributes accessed (read or write)"""
        return self.reads_attributes | self.writes_attributes


@dataclass
class SemanticCluster:
    """A group of methods that belong together semantically"""
    name: str                                                   # Cluster name (e.g., "memory")
    methods: List[str] = field(default_factory=list)           # Method names in this cluster
    shared_attributes: Set[str] = field(default_factory=set)   # Instance vars shared by these methods
    cohesion_score: float = 0.0                                # Internal cohesion (0-1)
    coupling_to_others: float = 0.0                            # Coupling to other clusters (0-1, lower better)
    rationale: str = ""                                        # Why these methods belong together
    confidence: float = 0.0                                    # Confidence in this clustering


@dataclass
class CohesionMetrics:
    """LCOM and related cohesion metrics for a class"""
    lcom1: float = 0.0          # Classic LCOM - count of method pairs with no shared vars
    lcom2: float = 0.0          # LCOM2 - (pairs_no_shared - pairs_shared) / pairs_total
    lcom3: float = 0.0          # LCOM3 - (methods - avg_methods_per_var) / (methods - 1)
    lcom4: int = 1              # LCOM4 - connected components in method-attribute graph
    tight_coupling: float = 0.0 # Fraction of methods with >50% shared attributes
    loose_coupling: float = 0.0 # Fraction of methods with <10% shared attributes
    recommendation: str = "keep" # "split" / "keep" / "refactor"

    @property
    def should_split(self) -> bool:
        """Heuristic: should this class be split?"""
        return self.lcom4 > 1 or self.lcom1 > 0.5 or self.loose_coupling > 0.7


@dataclass
class SplitPlan:
    """Plan for splitting a module into semantic domains"""
    source_file: str
    clusters: Dict[str, List[str]] = field(default_factory=dict)  # cluster_name -> method names
    target_files: Dict[str, str] = field(default_factory=dict)    # cluster_name -> target filename
    shared_imports: List[str] = field(default_factory=list)       # Imports needed by all
    cluster_imports: Dict[str, List[str]] = field(default_factory=dict)  # Per-cluster imports
    rationale: str = ""
    expected_cohesion_improvement: float = 0.0


# =============================================================================
# AST ANALYSIS - METHOD BEHAVIOR EXTRACTION
# =============================================================================

class MethodBehaviorExtractor(ast.NodeVisitor):
    """
    Extract behavioral information from a method's AST.

    Tracks:
    - self.method() calls -> calls_methods
    - self.attr reads -> reads_attributes
    - self.attr = ... writes -> writes_attributes
    - module.func() calls -> external_calls
    - local variable definitions -> local_variables
    - cyclomatic complexity
    """

    def __init__(self, method_name: str):
        self.method_name = method_name
        self.calls_methods: Set[str] = set()
        self.reads_attributes: Set[str] = set()
        self.writes_attributes: Set[str] = set()
        self.external_calls: Set[str] = set()
        self.local_variables: Set[str] = set()
        self.complexity = 1  # Base complexity

        # Track assignment targets to distinguish reads from writes
        self._in_assignment_target = False
        self._in_augmented_assign = False

    def visit_Assign(self, node: ast.Assign):
        """Track assignment targets"""
        # Visit targets as write context
        self._in_assignment_target = True
        for target in node.targets:
            self.visit(target)
        self._in_assignment_target = False

        # Visit value as read context
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        """Track augmented assignments (+=, -=, etc.)"""
        # Target is both read AND write
        self._in_augmented_assign = True
        self.visit(node.target)
        self._in_augmented_assign = False

        # Value is read
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Track annotated assignments"""
        if node.target:
            self._in_assignment_target = True
            self.visit(node.target)
            self._in_assignment_target = False
        if node.value:
            self.visit(node.value)

    def visit_Attribute(self, node: ast.Attribute):
        """Track self.attribute accesses"""
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            if self._in_assignment_target:
                self.writes_attributes.add(node.attr)
            elif self._in_augmented_assign:
                # Augmented assign reads AND writes
                self.reads_attributes.add(node.attr)
                self.writes_attributes.add(node.attr)
            else:
                self.reads_attributes.add(node.attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Track method and function calls"""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'self':
                    # self.method() call
                    self.calls_methods.add(node.func.attr)
                else:
                    # module.function() call
                    self.external_calls.add(f"{node.func.value.id}.{node.func.attr}")
            elif isinstance(node.func.value, ast.Attribute):
                # Could be self.obj.method() - track the chain
                parts = self._extract_attribute_chain(node.func)
                if parts and parts[0] == 'self':
                    # self.something.method()
                    self.reads_attributes.add(parts[1] if len(parts) > 1 else parts[0])
        elif isinstance(node.func, ast.Name):
            # Direct function call
            self.external_calls.add(node.func.id)

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """Track local variable definitions"""
        if self._in_assignment_target and node.id != 'self':
            self.local_variables.add(node.id)
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        """Track complexity from if statements"""
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """Track complexity from for loops"""
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        """Track complexity from while loops"""
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        """Track complexity from exception handlers"""
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        """Track complexity from boolean operators"""
        # Each 'and'/'or' adds a path
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension):
        """Track complexity from comprehensions"""
        self.complexity += 1
        self.generic_visit(node)

    def _extract_attribute_chain(self, node: ast.Attribute) -> List[str]:
        """Extract attribute chain like self.foo.bar -> ['self', 'foo', 'bar']"""
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return parts

    def get_behavior(self) -> MethodBehavior:
        """Return the extracted behavior"""
        return MethodBehavior(
            name=self.method_name,
            calls_methods=self.calls_methods,
            reads_attributes=self.reads_attributes,
            writes_attributes=self.writes_attributes,
            external_calls=self.external_calls,
            local_variables=self.local_variables,
            complexity=self.complexity
        )


class ClassBehaviorAnalyzer:
    """Analyze all methods in a class for behavioral clustering"""

    def analyze_class(self, class_node: ast.ClassDef, source_code: str) -> Dict[str, MethodBehavior]:
        """
        Extract behavior for all methods in a class.

        Args:
            class_node: AST node for the class
            source_code: Full source code (for line info)

        Returns:
            Dict mapping method name to MethodBehavior
        """
        behaviors = {}

        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                extractor = MethodBehaviorExtractor(node.name)
                extractor.visit(node)

                behavior = extractor.get_behavior()
                behavior.line_start = node.lineno
                behavior.line_end = node.end_lineno or node.lineno
                behavior.is_private = node.name.startswith('_') and not node.name.startswith('__')
                behavior.is_dunder = node.name.startswith('__') and node.name.endswith('__')

                # Extract docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                    behavior.docstring = node.body[0].value.value

                behaviors[node.name] = behavior

        return behaviors

    def analyze_module(self, source_code: str, class_name: Optional[str] = None) -> Dict[str, MethodBehavior]:
        """
        Analyze a module's class(es) for method behaviors.

        Args:
            source_code: Source code to analyze
            class_name: Optional specific class to analyze (analyzes first/largest if not specified)

        Returns:
            Dict mapping method name to MethodBehavior
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"  [Syntax error in source: {e}]")
            return {}

        # Find class(es)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        if not classes:
            return {}

        if class_name:
            target_class = next((c for c in classes if c.name == class_name), None)
            if not target_class:
                print(f"  [Class {class_name} not found]")
                return {}
        else:
            # Pick the largest class
            target_class = max(classes, key=lambda c: len([n for n in c.body
                                                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]))

        return self.analyze_class(target_class, source_code)


# =============================================================================
# COHESION METRICS - LCOM CALCULATION
# =============================================================================

class LCOMCalculator:
    """
    Calculate LCOM (Lack of Cohesion of Methods) metrics.

    LCOM metrics measure how well methods in a class relate to each other
    through shared instance variables. Higher LCOM = less cohesive = consider splitting.
    """

    def compute_all(self, behaviors: Dict[str, MethodBehavior]) -> CohesionMetrics:
        """Compute all LCOM metrics"""
        metrics = CohesionMetrics()

        if len(behaviors) < 2:
            metrics.recommendation = "keep"
            return metrics

        metrics.lcom1 = self.compute_lcom1(behaviors)
        metrics.lcom2 = self.compute_lcom2(behaviors)
        metrics.lcom3 = self.compute_lcom3(behaviors)
        metrics.lcom4 = self.compute_lcom4(behaviors)
        metrics.tight_coupling, metrics.loose_coupling = self.compute_coupling_distribution(behaviors)

        # Recommendation based on metrics
        if metrics.lcom4 > 1:
            metrics.recommendation = "split"
        elif metrics.lcom1 > 0.6:
            metrics.recommendation = "split"
        elif metrics.lcom1 > 0.4:
            metrics.recommendation = "refactor"
        else:
            metrics.recommendation = "keep"

        return metrics

    def compute_lcom1(self, behaviors: Dict[str, MethodBehavior]) -> float:
        """
        LCOM1: Fraction of method pairs that share NO instance variables.

        Lower is better (more cohesive).
        Range: 0.0 (all pairs share vars) to 1.0 (no pairs share vars)
        """
        methods = list(behaviors.keys())
        n = len(methods)
        if n < 2:
            return 0.0

        no_shared_pairs = 0
        total_pairs = n * (n - 1) // 2

        for i in range(n):
            for j in range(i + 1, n):
                m1 = behaviors[methods[i]]
                m2 = behaviors[methods[j]]

                attrs1 = m1.all_attributes
                attrs2 = m2.all_attributes

                if not attrs1.intersection(attrs2):
                    no_shared_pairs += 1

        return no_shared_pairs / total_pairs if total_pairs > 0 else 0.0

    def compute_lcom2(self, behaviors: Dict[str, MethodBehavior]) -> float:
        """
        LCOM2: (pairs_no_shared - pairs_shared) / total_pairs

        Negative = cohesive, Positive = not cohesive
        Normalized to 0-1 range where 0.5 is neutral
        """
        methods = list(behaviors.keys())
        n = len(methods)
        if n < 2:
            return 0.0

        no_shared_pairs = 0
        shared_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                m1 = behaviors[methods[i]]
                m2 = behaviors[methods[j]]

                attrs1 = m1.all_attributes
                attrs2 = m2.all_attributes

                if attrs1.intersection(attrs2):
                    shared_pairs += 1
                else:
                    no_shared_pairs += 1

        total_pairs = no_shared_pairs + shared_pairs
        if total_pairs == 0:
            return 0.5

        # Normalize to 0-1 range
        raw = (no_shared_pairs - shared_pairs) / total_pairs
        return (raw + 1) / 2  # Map [-1, 1] to [0, 1]

    def compute_lcom3(self, behaviors: Dict[str, MethodBehavior]) -> float:
        """
        LCOM3: (methods - avg_methods_per_attr) / (methods - 1)

        Measures how many methods use each attribute on average.
        Range: 0.0 (cohesive) to 1.0 (not cohesive)
        """
        methods = list(behaviors.keys())
        n = len(methods)
        if n < 2:
            return 0.0

        # Count how many methods access each attribute
        attr_method_count: Dict[str, int] = defaultdict(int)
        for name, behavior in behaviors.items():
            for attr in behavior.all_attributes:
                attr_method_count[attr] += 1

        if not attr_method_count:
            return 1.0  # No shared attributes = not cohesive

        # Average methods per attribute
        avg_methods_per_attr = sum(attr_method_count.values()) / len(attr_method_count)

        lcom3 = (n - avg_methods_per_attr) / (n - 1) if n > 1 else 0
        return max(0.0, min(1.0, lcom3))

    def compute_lcom4(self, behaviors: Dict[str, MethodBehavior]) -> int:
        """
        LCOM4: Number of connected components in method-attribute graph.

        Methods are connected if they share attributes OR one calls the other.
        LCOM4 = 1 means perfectly cohesive.
        LCOM4 > 1 suggests the class could be split into that many classes.
        """
        methods = list(behaviors.keys())
        n = len(methods)
        if n < 2:
            return 1

        # Build adjacency using Union-Find
        parent = {m: m for m in methods}
        rank = {m: 0 for m in methods}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Connect methods that share attributes
        for i in range(n):
            for j in range(i + 1, n):
                m1 = behaviors[methods[i]]
                m2 = behaviors[methods[j]]

                # Connected if share attributes
                if m1.all_attributes.intersection(m2.all_attributes):
                    union(methods[i], methods[j])
                    continue

                # Connected if one calls the other
                if methods[j] in m1.calls_methods or methods[i] in m2.calls_methods:
                    union(methods[i], methods[j])

        # Count connected components
        components = len(set(find(m) for m in methods))
        return components

    def compute_coupling_distribution(self, behaviors: Dict[str, MethodBehavior]) -> Tuple[float, float]:
        """
        Compute distribution of coupling strengths.

        Returns:
            (tight_coupling, loose_coupling) - fractions of method pairs
            tight_coupling: pairs sharing >50% of their combined attributes
            loose_coupling: pairs sharing <10% of their combined attributes
        """
        methods = list(behaviors.keys())
        n = len(methods)
        if n < 2:
            return 0.0, 0.0

        tight_count = 0
        loose_count = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                m1 = behaviors[methods[i]]
                m2 = behaviors[methods[j]]

                attrs1 = m1.all_attributes
                attrs2 = m2.all_attributes
                union_attrs = attrs1 | attrs2
                intersect_attrs = attrs1 & attrs2

                if not union_attrs:
                    loose_count += 1
                else:
                    ratio = len(intersect_attrs) / len(union_attrs)
                    if ratio > 0.5:
                        tight_count += 1
                    elif ratio < 0.1:
                        loose_count += 1

                total_pairs += 1

        return (tight_count / total_pairs if total_pairs > 0 else 0.0,
                loose_count / total_pairs if total_pairs > 0 else 0.0)


# =============================================================================
# METHOD CALL GRAPH
# =============================================================================

class MethodCallGraph:
    """Build and analyze call relationships between methods"""

    def __init__(self, behaviors: Dict[str, MethodBehavior]):
        self.behaviors = behaviors
        self.graph = self._build_graph()
        self.reverse_graph = self._build_reverse_graph()

    def _build_graph(self) -> Dict[str, Set[str]]:
        """Build adjacency list: method -> methods it calls"""
        graph = {name: set() for name in self.behaviors}
        for name, behavior in self.behaviors.items():
            for called in behavior.calls_methods:
                if called in graph:
                    graph[name].add(called)
        return graph

    def _build_reverse_graph(self) -> Dict[str, Set[str]]:
        """Build reverse adjacency: method -> methods that call it"""
        reverse = {name: set() for name in self.behaviors}
        for caller, callees in self.graph.items():
            for callee in callees:
                if callee in reverse:
                    reverse[callee].add(caller)
        return reverse

    def find_clusters(self) -> List[Set[str]]:
        """
        Find strongly connected components using Tarjan's algorithm.
        Returns list of SCCs (each SCC is a set of method names).
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []

        def strongconnect(v):
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            on_stack[v] = True
            stack.append(v)

            for w in self.graph.get(v, []):
                if w not in index:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack.get(w, False):
                    lowlinks[v] = min(lowlinks[v], index[w])

            if lowlinks[v] == index[v]:
                scc = set()
                while stack:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.add(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in self.behaviors:
            if v not in index:
                strongconnect(v)

        return sccs

    def find_helper_methods(self) -> Set[str]:
        """
        Find private methods only called by one or zero public methods.
        These are good candidates to be moved with their caller.
        """
        helpers = set()

        for name, behavior in self.behaviors.items():
            if not behavior.is_private:
                continue

            # Count public callers
            public_callers = [c for c in self.reverse_graph[name]
                            if not self.behaviors[c].is_private]

            if len(public_callers) <= 1:
                helpers.add(name)

        return helpers

    def get_call_distance(self, method1: str, method2: str) -> int:
        """
        Get shortest call distance between two methods.
        Returns -1 if not connected.
        """
        if method1 == method2:
            return 0

        # BFS from method1
        from collections import deque
        visited = {method1}
        queue = deque([(method1, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor == method2:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # Try reverse direction
        visited = {method2}
        queue = deque([(method2, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor == method1:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return -1  # Not connected


# =============================================================================
# SEMANTIC DOMAIN DETECTION
# =============================================================================

class SemanticDomainDetector:
    """
    Detect semantic domains from method behavior, not just names.

    Domains are inferred from:
    1. What attributes are accessed (memory methods access memory_*, etc.)
    2. What external modules are called
    3. What other methods are called (clustering by call patterns)
    4. Docstring keywords
    """

    # Domain fingerprints - attributes and patterns that indicate domain
    DOMAIN_FINGERPRINTS = {
        'memory': {
            'attributes': ['memory', 'conversation', 'history', 'context', 'recall',
                          'episodic', 'associative', 'remember', 'store', 'retrieve'],
            'external_calls': ['json.load', 'json.dump', 'json.loads', 'json.dumps'],
            'docstring_keywords': ['memory', 'remember', 'recall', 'store', 'retrieve', 'history'],
            'patterns': ['save', 'load', 'store', 'retrieve', 'recall', 'remember', 'memory']
        },
        'emotion': {
            'attributes': ['heart', 'emotion', 'valence', 'arousal', 'feeling', 'mood',
                          'affect', 'sentiment', 'empathy', 'drive'],
            'external_calls': [],
            'docstring_keywords': ['emotion', 'feeling', 'mood', 'affect', 'sentiment'],
            'patterns': ['feel', 'emotion', 'affect', 'sentiment', 'mood', 'heart']
        },
        'response': {
            'attributes': ['response', 'output', 'tensor_field', 'codebook', 'composer',
                          'generation', 'emit', 'produce', 'grammar'],
            'external_calls': ['np.', 'numpy.'],
            'docstring_keywords': ['response', 'generate', 'output', 'compose', 'emit'],
            'patterns': ['generate', 'compose', 'emit', 'encode', 'decode', 'response', 'output']
        },
        'identity': {
            'attributes': ['identity', 'binding', 'projections', 'nexus', 'grounding',
                          'self_model', 'sovereignty', 'consent', 'anchor'],
            'external_calls': [],
            'docstring_keywords': ['identity', 'self', 'ground', 'anchor', 'verify'],
            'patterns': ['verify', 'ground', 'anchor', 'protect', 'identity', 'bind']
        },
        'learning': {
            'attributes': ['learned', 'adaptation', 'update', 'improvement', 'meta',
                          'training', 'evolve', 'adapt', 'feedback'],
            'external_calls': [],
            'docstring_keywords': ['learn', 'adapt', 'improve', 'update', 'train'],
            'patterns': ['learn', 'adapt', 'improve', 'update', 'train', 'evolve']
        },
        'lifecycle': {
            'attributes': ['sleep', 'wake', 'dream', 'consolidate', 'rest', 'active',
                          'state', 'transition', 'awake'],
            'external_calls': [],
            'docstring_keywords': ['sleep', 'wake', 'dream', 'rest', 'state'],
            'patterns': ['sleep', 'wake', 'dream', 'consolidate', 'rest', 'goodnight']
        },
        'dialogue': {
            'attributes': ['turn', 'input', 'user', 'message', 'conversation', 'discourse',
                          'speaker', 'utterance'],
            'external_calls': [],
            'docstring_keywords': ['dialogue', 'conversation', 'turn', 'input', 'user'],
            'patterns': ['process', 'input', 'dialogue', 'conversation', 'turn', 'speak']
        },
        'introspection': {
            'attributes': ['introspect', 'reflect', 'meta', 'self_aware', 'thinking',
                          'cognitive', 'reason', 'understand'],
            'external_calls': [],
            'docstring_keywords': ['introspect', 'reflect', 'think', 'reason', 'understand'],
            'patterns': ['introspect', 'reflect', 'think', 'reason', 'understand', 'know']
        },
        'utility': {
            'attributes': [],
            'external_calls': [],
            'docstring_keywords': ['helper', 'utility', 'internal', 'private'],
            'patterns': ['_has_', '_safe_', '_log', '_check', '_validate', '_ensure', 'helper', 'util']
        }
    }

    def detect_domain(self, behavior: MethodBehavior) -> Tuple[str, float]:
        """
        Detect the semantic domain of a method.

        Args:
            behavior: MethodBehavior for the method

        Returns:
            (domain_name, confidence_score)
        """
        scores: Dict[str, float] = {}
        all_attrs = behavior.all_attributes
        name_lower = behavior.name.lower()
        docstring_lower = (behavior.docstring or '').lower()

        for domain, fingerprint in self.DOMAIN_FINGERPRINTS.items():
            score = 0.0
            evidence = []

            # Check attribute matches (weight: 0.4)
            attr_matches = sum(1 for attr in all_attrs
                             if any(fp in attr.lower() for fp in fingerprint['attributes']))
            if attr_matches > 0:
                score += min(attr_matches * 0.2, 0.4)
                evidence.append(f"attrs:{attr_matches}")

            # Check external call matches (weight: 0.2)
            call_matches = sum(1 for call in behavior.external_calls
                             if any(fp in call for fp in fingerprint['external_calls']))
            if call_matches > 0:
                score += min(call_matches * 0.1, 0.2)
                evidence.append(f"calls:{call_matches}")

            # Check name pattern matches (weight: 0.3)
            pattern_matches = sum(1 for p in fingerprint['patterns'] if p in name_lower)
            if pattern_matches > 0:
                score += min(pattern_matches * 0.15, 0.3)
                evidence.append(f"name:{pattern_matches}")

            # Check docstring keywords (weight: 0.1)
            if fingerprint['docstring_keywords']:
                doc_matches = sum(1 for k in fingerprint['docstring_keywords'] if k in docstring_lower)
                if doc_matches > 0:
                    score += min(doc_matches * 0.05, 0.1)
                    evidence.append(f"doc:{doc_matches}")

            scores[domain] = score

        if not scores or max(scores.values()) == 0:
            return ('core', 0.0)

        best_domain = max(scores, key=scores.get)
        confidence = scores[best_domain]

        # Require minimum confidence
        if confidence < 0.15:
            return ('core', confidence)

        return (best_domain, confidence)

    def classify_all(self, behaviors: Dict[str, MethodBehavior]) -> Dict[str, Tuple[str, float]]:
        """
        Classify all methods into domains.

        Returns:
            Dict mapping method name to (domain, confidence)
        """
        return {name: self.detect_domain(behavior) for name, behavior in behaviors.items()}


# =============================================================================
# SEMANTIC MODULE SPLITTER
# =============================================================================

class SemanticModuleSplitter:
    """
    Enhanced splitter that uses semantic analysis instead of just name prefixes.

    Process:
    1. Extract behaviors for all methods
    2. Compute cohesion metrics (LCOM)
    3. Detect semantic domains for each method
    4. Build clusters respecting call relationships
    5. Generate split plan with proper imports
    """

    def __init__(self, min_cohesion: float = 0.3, min_methods_per_cluster: int = 3):
        self.min_cohesion = min_cohesion
        self.min_methods_per_cluster = min_methods_per_cluster
        self.lcom_calc = LCOMCalculator()
        self.domain_detector = SemanticDomainDetector()

    def analyze(self, source_code: str, class_name: Optional[str] = None,
                verbose: bool = False) -> Dict[str, Any]:
        """
        Perform full semantic analysis of a class.

        Args:
            source_code: Source code to analyze
            class_name: Specific class to analyze (or largest)
            verbose: Print progress

        Returns:
            Analysis dict with behaviors, cohesion, domains, recommendation
        """
        if verbose:
            print("  Analyzing method behaviors...")

        analyzer = ClassBehaviorAnalyzer()
        behaviors = analyzer.analyze_module(source_code, class_name)

        if not behaviors:
            return {'error': 'No methods found to analyze'}

        if verbose:
            print(f"  Found {len(behaviors)} methods")

        # Compute cohesion
        if verbose:
            print("  Computing cohesion metrics...")
        cohesion = self.lcom_calc.compute_all(behaviors)

        # Build call graph
        if verbose:
            print("  Building call graph...")
        call_graph = MethodCallGraph(behaviors)
        helper_methods = call_graph.find_helper_methods()

        # Detect domains
        if verbose:
            print("  Detecting semantic domains...")
        domain_classifications = self.domain_detector.classify_all(behaviors)

        # Group by domain
        domains: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for name, (domain, confidence) in domain_classifications.items():
            domains[domain].append((name, confidence))

        # Sort methods within each domain by confidence
        for domain in domains:
            domains[domain].sort(key=lambda x: -x[1])

        if verbose:
            print(f"  Detected {len(domains)} domains: {list(domains.keys())}")

        return {
            'behaviors': behaviors,
            'cohesion': cohesion,
            'call_graph': call_graph,
            'helper_methods': helper_methods,
            'domain_classifications': domain_classifications,
            'domains': dict(domains),
            'should_split': cohesion.should_split,
            'method_count': len(behaviors),
            'recommendation': cohesion.recommendation
        }

    def _merge_related_clusters(self, clusters: Dict[str, List[str]],
                               behaviors: Dict[str, MethodBehavior],
                               verbose: bool = False) -> Dict[str, List[str]]:
        """
        Merge clusters that are closely related (share many attributes).

        This prevents over-fragmentation where tightly coupled domains
        would create import cycles or awkward dependencies.
        """
        if len(clusters) < 3:
            return clusters

        # Compute shared attributes between clusters
        cluster_attrs: Dict[str, Set[str]] = {}
        for cluster_name, methods in clusters.items():
            all_attrs = set()
            for m in methods:
                if m in behaviors:
                    all_attrs.update(behaviors[m].all_attributes)
            cluster_attrs[cluster_name] = all_attrs

        # Find clusters with >50% attribute overlap
        merged = dict(clusters)
        cluster_names = list(clusters.keys())

        for i, c1 in enumerate(cluster_names):
            if c1 not in merged:
                continue
            for c2 in cluster_names[i+1:]:
                if c2 not in merged or c1 == 'core' or c2 == 'core':
                    continue

                attrs1 = cluster_attrs.get(c1, set())
                attrs2 = cluster_attrs.get(c2, set())

                if not attrs1 or not attrs2:
                    continue

                union_size = len(attrs1 | attrs2)
                intersect_size = len(attrs1 & attrs2)

                if union_size > 0 and intersect_size / union_size > 0.5:
                    # Merge c2 into c1
                    if verbose:
                        print(f"    Merging '{c2}' into '{c1}' (attribute overlap: {intersect_size}/{union_size})")
                    merged[c1].extend(merged[c2])
                    del merged[c2]
                    # Update attrs for merged cluster
                    cluster_attrs[c1] = attrs1 | attrs2

        return merged

    def generate_split_plan(self, analysis: Dict[str, Any], source_file: str,
                           verbose: bool = False) -> Optional[SplitPlan]:
        """
        Generate a split plan based on semantic analysis.

        Args:
            analysis: Result from analyze()
            source_file: Path to source file
            verbose: Print progress

        Returns:
            SplitPlan if splitting recommended, None otherwise
        """
        if not analysis.get('should_split', False):
            if verbose:
                print("  Split not recommended by cohesion metrics")
            return None

        behaviors = analysis['behaviors']
        domains = analysis['domains']
        helper_methods = analysis['helper_methods']
        call_graph = analysis['call_graph']

        # Build initial clusters from domains
        # Use a smarter strategy: keep domains with high average confidence
        clusters: Dict[str, List[str]] = {}

        for domain, method_list in domains.items():
            method_names = [m[0] for m in method_list]
            avg_confidence = sum(c for _, c in method_list) / len(method_list) if method_list else 0

            # Keep domain if:
            # 1. Enough methods (>= threshold)
            # 2. OR high confidence (>= 0.4 avg) even if small
            # 3. OR it's 'core' (catch-all)
            keep_domain = (len(method_list) >= self.min_methods_per_cluster or
                          avg_confidence >= 0.4 or
                          domain == 'core')

            if keep_domain:
                clusters[domain] = method_names
            elif verbose:
                print(f"    Merging small low-confidence domain '{domain}' ({len(method_list)} methods, {avg_confidence:.2f} conf) into core")

        # Domains not kept go to 'core'
        if 'core' not in clusters:
            clusters['core'] = []

        for domain, method_list in domains.items():
            if domain not in clusters:
                clusters['core'].extend([m[0] for m in method_list])

        # Merge related domains if they're closely coupled
        # E.g., if 'emotion' and 'lifecycle' share many attributes, merge them
        merged = self._merge_related_clusters(clusters, behaviors, verbose)
        clusters = merged

        # Move helpers to same cluster as their caller
        for helper in helper_methods:
            if helper not in behaviors:
                continue

            # Find which cluster calls this helper
            callers = call_graph.reverse_graph.get(helper, set())
            if callers:
                # Find caller's cluster
                caller = next(iter(callers))
                caller_cluster = None
                for cluster_name, methods in clusters.items():
                    if caller in methods:
                        caller_cluster = cluster_name
                        break

                # Move helper to caller's cluster
                if caller_cluster:
                    for cluster_name, methods in list(clusters.items()):
                        if helper in methods and cluster_name != caller_cluster:
                            methods.remove(helper)
                            clusters[caller_cluster].append(helper)
                            break

        # Remove empty clusters
        clusters = {k: v for k, v in clusters.items() if v}

        # Final check: don't split if only 1 cluster remains or all in core
        non_core_clusters = [c for c in clusters if c != 'core']
        if len(non_core_clusters) < 1:
            if verbose:
                print("  Only core cluster after merging - no split needed")
            return None

        # Compute cohesion improvement
        original_lcom1 = analysis['cohesion'].lcom1

        # Estimate new cohesion for each cluster
        cluster_cohesions = []
        for cluster_name, methods in clusters.items():
            cluster_behaviors = {m: behaviors[m] for m in methods if m in behaviors}
            if len(cluster_behaviors) > 1:
                cluster_cohesion = self.lcom_calc.compute_lcom1(cluster_behaviors)
                cluster_cohesions.append(cluster_cohesion)

        expected_improvement = original_lcom1 - (sum(cluster_cohesions) / len(cluster_cohesions)
                                                 if cluster_cohesions else 0)

        # Generate target filenames
        source_path = Path(source_file)
        base_name = source_path.stem

        target_files = {}
        for cluster_name in clusters:
            if cluster_name == 'core':
                target_files[cluster_name] = source_file  # Core stays in original
            else:
                target_files[cluster_name] = str(source_path.parent / f"{base_name}_{cluster_name}.py")

        return SplitPlan(
            source_file=source_file,
            clusters=clusters,
            target_files=target_files,
            rationale=f"Split based on {len(clusters)} semantic domains with expected cohesion improvement of {expected_improvement:.2f}",
            expected_cohesion_improvement=expected_improvement
        )

    def format_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results as human-readable report"""
        if 'error' in analysis:
            return f"Analysis error: {analysis['error']}"

        lines = []
        lines.append("=" * 60)
        lines.append("SEMANTIC ANALYSIS REPORT")
        lines.append("=" * 60)

        # Cohesion metrics
        cohesion = analysis['cohesion']
        lines.append(f"\nCohesion Metrics:")
        lines.append(f"  LCOM1 (fraction of unrelated pairs): {cohesion.lcom1:.3f}")
        lines.append(f"  LCOM4 (connected components): {cohesion.lcom4}")
        lines.append(f"  Tight coupling: {cohesion.tight_coupling:.1%}")
        lines.append(f"  Loose coupling: {cohesion.loose_coupling:.1%}")
        lines.append(f"  Recommendation: {cohesion.recommendation.upper()}")

        # Domain breakdown
        lines.append(f"\nSemantic Domains ({len(analysis['domains'])} detected):")
        for domain, methods in sorted(analysis['domains'].items(), key=lambda x: -len(x[1])):
            avg_confidence = sum(c for _, c in methods) / len(methods) if methods else 0
            lines.append(f"  {domain}: {len(methods)} methods (avg confidence: {avg_confidence:.2f})")
            for method_name, confidence in methods[:5]:  # Show top 5
                lines.append(f"    - {method_name} ({confidence:.2f})")
            if len(methods) > 5:
                lines.append(f"    ... and {len(methods) - 5} more")

        # Helper methods
        helpers = analysis.get('helper_methods', set())
        if helpers:
            lines.append(f"\nHelper Methods (private, single-caller): {len(helpers)}")
            for h in list(helpers)[:10]:
                lines.append(f"  - {h}")
            if len(helpers) > 10:
                lines.append(f"  ... and {len(helpers) - 10} more")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def generate_class_specs(self, split_plan: SplitPlan,
                            analysis: Dict[str, Any]) -> Dict[str, 'ClassSpec']:
        """
        Generate ClassSpecs for each cluster using Phase 2.5.

        This allows split modules to have proper class structure with:
        - Shared attributes as properties
        - Methods with proper signatures
        - Invariants inferred from attribute usage

        Args:
            split_plan: The split plan from generate_split_plan()
            analysis: The analysis dict from analyze()

        Returns:
            Dict mapping cluster_name to ClassSpec
        """
        from .phase2_5_class_composition import ClassSpec, MethodSpec, PropertySpec

        behaviors = analysis['behaviors']
        class_specs = {}

        for cluster_name, methods in split_plan.clusters.items():
            if cluster_name == 'core':
                continue  # Core stays in original class

            # Get behaviors for methods in this cluster
            cluster_behaviors = {m: behaviors[m] for m in methods if m in behaviors}

            # Find shared attributes across all methods in cluster
            all_read_attrs = set()
            all_write_attrs = set()
            for method_name, behavior in cluster_behaviors.items():
                all_read_attrs.update(behavior.reads_attributes)
                all_write_attrs.update(behavior.writes_attributes)

            shared_attrs = all_read_attrs | all_write_attrs

            # Build PropertySpecs from shared attributes
            properties = []
            for attr in sorted(shared_attrs):
                # Try to infer type from usage (simplified)
                prop_type = 'Any'  # Default
                default_val = None

                # Common patterns
                if 'memory' in attr or 'history' in attr or 'cache' in attr:
                    prop_type = 'Dict'
                    default_val = '{}'
                elif 'list' in attr or 'buffer' in attr or 'queue' in attr:
                    prop_type = 'List'
                    default_val = '[]'
                elif 'state' in attr or 'status' in attr or 'flag' in attr:
                    prop_type = 'bool'
                    default_val = 'False'
                elif 'count' in attr or 'index' in attr or 'num' in attr:
                    prop_type = 'int'
                    default_val = '0'
                elif 'score' in attr or 'value' in attr or 'rate' in attr:
                    prop_type = 'float'
                    default_val = '0.0'

                properties.append(PropertySpec(
                    name=attr,
                    type=prop_type,
                    description=f"Attribute used by {cluster_name} methods",
                    default=default_val
                ))

            # Build MethodSpecs from behaviors
            method_specs = []
            for method_name, behavior in cluster_behaviors.items():
                # Skip dunders except __init__
                if behavior.is_dunder and method_name != '__init__':
                    continue

                method_specs.append(MethodSpec(
                    name=method_name,
                    description=behavior.docstring or f"Method from {cluster_name} domain",
                    parameters=[],  # Would need deeper analysis
                    return_type='Any',
                    accesses_self=list(behavior.reads_attributes),
                    modifies_self=list(behavior.writes_attributes)
                ))

            # Infer invariants from shared attributes
            invariants = []
            for prop in properties:
                if prop.type == 'Dict':
                    invariants.append(f"isinstance(self.{prop.name}, dict)")
                elif prop.type == 'List':
                    invariants.append(f"isinstance(self.{prop.name}, list)")

            # Generate class name from cluster
            class_name = ''.join(word.title() for word in cluster_name.split('_'))
            class_name = f"Grace{class_name}Handler"

            class_specs[cluster_name] = ClassSpec(
                name=class_name,
                description=f"Handles {cluster_name} functionality extracted from GraceInteractiveDialogue",
                properties=properties,
                methods=method_specs,
                invariants=invariants[:5],  # Limit invariants
                imports_needed=['from typing import Any, Dict, List, Optional']
            )

        return class_specs

    def generate_split_files(self, split_plan: SplitPlan,
                            analysis: Dict[str, Any],
                            source_code: str,
                            verbose: bool = False) -> Dict[str, str]:
        """
        Generate complete file contents for each split target.

        This creates actual Python files that can be written.

        Args:
            split_plan: The split plan
            analysis: The analysis dict
            source_code: Original source code
            verbose: Print progress

        Returns:
            Dict mapping target_file_path to file_content
        """
        from .phase2_5_class_composition import ClassComposer

        behaviors = analysis['behaviors']
        files = {}

        # Generate ClassSpecs
        class_specs = self.generate_class_specs(split_plan, analysis)

        for cluster_name, target_file in split_plan.target_files.items():
            if cluster_name == 'core':
                continue  # Core file handled separately

            if verbose:
                print(f"  Generating {target_file}...")

            spec = class_specs.get(cluster_name)
            if not spec:
                continue

            # Generate class code using Phase 2.5
            try:
                composer = ClassComposer()
                generated = composer.compose_class(spec, verbose=False)
                class_code = generated.code
            except Exception as e:
                if verbose:
                    print(f"    [Warning: Could not compose class: {e}]")
                # Fallback: generate a simple stub
                class_code = self._generate_stub_class(cluster_name, split_plan.clusters[cluster_name], behaviors)

            # Build file content
            content_lines = [
                f'"""',
                f'{spec.description}',
                f'',
                f'Auto-generated by Phase 10 Semantic Splitter',
                f'"""',
                f'',
            ]

            # Add imports
            content_lines.extend(spec.imports_needed)
            content_lines.append('')
            content_lines.append('')

            # Add class code
            content_lines.append(class_code)

            files[target_file] = '\n'.join(content_lines)

        return files

    def _generate_stub_class(self, cluster_name: str, methods: List[str],
                            behaviors: Dict[str, 'MethodBehavior']) -> str:
        """Generate a simple stub class when Phase 2.5 fails"""
        class_name = ''.join(word.title() for word in cluster_name.split('_'))
        class_name = f"Grace{class_name}Handler"

        lines = [
            f"class {class_name}:",
            f'    """Handles {cluster_name} functionality"""',
            f'',
            f'    def __init__(self, grace_instance):',
            f'        self.grace = grace_instance',
            f''
        ]

        for method_name in methods:
            behavior = behaviors.get(method_name)
            if behavior and behavior.is_dunder:
                continue  # Skip dunders

            docstring = behavior.docstring if behavior else "Method stub"
            lines.extend([
                f'    def {method_name}(self, *args, **kwargs):',
                f'        """{docstring}"""',
                f'        return self.grace.{method_name}(*args, **kwargs)',
                f''
            ])

        return '\n'.join(lines)


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo():
    """Demonstrate semantic analysis capabilities"""
    print("=" * 60)
    print("PHASE 10 SEMANTIC SPLITTER DEMO")
    print("=" * 60)

    # Test with a sample class
    test_code = '''
class ExampleSystem:
    """A system with multiple semantic domains"""

    def __init__(self):
        self.memory = {}
        self.emotion_state = 0.5
        self.response_buffer = []
        self.identity_verified = False

    # Memory domain
    def save_memory(self, key, value):
        """Save to memory"""
        self.memory[key] = value

    def load_memory(self, key):
        """Load from memory"""
        return self.memory.get(key)

    def clear_memory(self):
        """Clear all memory"""
        self.memory.clear()

    # Emotion domain
    def update_emotion(self, delta):
        """Update emotional state"""
        self.emotion_state += delta
        self.emotion_state = max(0, min(1, self.emotion_state))

    def get_feeling(self):
        """Get current feeling"""
        return "happy" if self.emotion_state > 0.5 else "sad"

    # Response domain
    def generate_response(self, input_text):
        """Generate a response"""
        feeling = self.get_feeling()
        memory = self.load_memory("context")
        return f"[{feeling}] Response to: {input_text}"

    def emit_output(self):
        """Emit buffered output"""
        output = self.response_buffer
        self.response_buffer = []
        return output

    # Identity domain
    def verify_identity(self):
        """Verify identity"""
        self.identity_verified = True

    def _check_permissions(self):
        """Internal permission check"""
        return self.identity_verified

    # Utility
    def _log(self, message):
        """Internal logging"""
        print(f"[LOG] {message}")
'''

    splitter = SemanticModuleSplitter()

    print("\nAnalyzing sample class...")
    analysis = splitter.analyze(test_code, verbose=True)

    print("\n" + splitter.format_analysis_report(analysis))

    print("\nGenerating split plan...")
    plan = splitter.generate_split_plan(analysis, "example_system.py", verbose=True)

    if plan:
        print(f"\nSplit Plan:")
        print(f"  Source: {plan.source_file}")
        print(f"  Clusters: {len(plan.clusters)}")
        for cluster_name, methods in plan.clusters.items():
            print(f"    {cluster_name}: {len(methods)} methods")
            for m in methods:
                print(f"      - {m}")
        print(f"  Expected cohesion improvement: {plan.expected_cohesion_improvement:.2f}")
    else:
        print("\n  No split recommended")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
