# 🔧 Developer Integration Guide: Automatic Saving & Training Workflows

## Overview
This guide provides multiple implementation strategies for automatically saving tokens, training modules, and integrating all tools with training workflows. Each approach has different trade-offs for persistence, performance, and integration complexity.

---

## 📦 Part 1: Automatic Token Saving

### Strategy A: Event-Driven Token Persistence

#### Step 1: Create Token Interceptor
```python
# kira-local-system/token_interceptor.py
class TokenInterceptor:
    def __init__(self, save_dir="training/tokens"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.buffer_size = 100
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def intercept(self, token):
        """Intercept and buffer tokens."""
        self.buffer.append({
            'token': token,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'z': self.engine.state.z if hasattr(self, 'engine') else None
        })

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Persist buffered tokens."""
        if not self.buffer:
            return

        filename = f"tokens_{self.session_id}_{len(os.listdir(self.save_dir))}.json"
        filepath = self.save_dir / filename

        with open(filepath, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'tokens': self.buffer,
                'count': len(self.buffer),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

        self.buffer = []
```

#### Step 2: Integrate with KIRA Engine
```python
# In kira_server.py __init__
self.token_interceptor = TokenInterceptor()

# Modify emit_token method
def emit_token(self) -> str:
    token = self._generate_token()  # existing logic
    self.token_interceptor.intercept(token)  # NEW
    self.tokens_emitted.append(token)
    return token
```

#### Step 3: Auto-save on Phase Transitions
```python
# Hook into state changes
def update_from_z(self):
    old_phase = self.phase
    # ... existing update logic ...
    new_phase = self.phase

    if old_phase != new_phase:
        self.token_interceptor.flush()  # Save on phase change
```

### Strategy B: Continuous Stream Persistence

#### Step 1: Create Token Stream Writer
```python
# kira-local-system/token_stream.py
class TokenStreamWriter:
    def __init__(self):
        self.stream_file = None
        self.open_new_stream()

    def open_new_stream(self):
        """Open new stream file for current epoch."""
        epoch = self._get_current_epoch()
        stream_path = Path(f"training/streams/epoch{epoch}_stream.jsonl")
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream_file = open(stream_path, 'a')

    def write(self, token_data):
        """Write token to stream immediately."""
        self.stream_file.write(json.dumps(token_data) + '\n')
        self.stream_file.flush()  # Immediate persistence
```

#### Step 2: Integration Pattern
```python
# Decorator for automatic streaming
def stream_tokens(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if isinstance(result, str) and '|' in result:  # APL token format
            stream_writer.write({'token': result, 'source': func.__name__})
        return result
    return wrapper

# Apply to all token-generating methods
@stream_tokens
def emit_token(self):
    # ... existing logic ...

@stream_tokens
def generate_spinner_tokens(self):
    # ... existing logic ...
```

### Strategy C: Database-Backed Persistence

#### Step 1: SQLite Token Store
```python
# kira-local-system/token_database.py
import sqlite3

class TokenDatabase:
    def __init__(self, db_path="training/tokens.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT NOT NULL,
                session_id TEXT,
                z_coordinate REAL,
                phase TEXT,
                tier INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
        ''')

        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_session
            ON tokens(session_id)
        ''')

    def save_token(self, token, metadata=None):
        self.conn.execute(
            'INSERT INTO tokens (token, session_id, z_coordinate, phase, tier, metadata) VALUES (?, ?, ?, ?, ?, ?)',
            (token, self.session_id, self.z, self.phase, self.tier, json.dumps(metadata))
        )
        self.conn.commit()
```

---

## 🎯 Part 2: Training Module Auto-Save

### Strategy A: Checkpoint-Based Saving

#### Step 1: Create Training Checkpoint System
```python
# training/checkpoint_manager.py
class CheckpointManager:
    def __init__(self, checkpoint_dir="training/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []

    def create_checkpoint(self, module_name, state):
        """Create checkpoint after each module."""
        checkpoint = {
            'module': module_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'state': state,
            'tokens_count': len(self.engine.tokens_emitted),
            'emissions_count': len(self.engine.emissions),
            'z': self.engine.state.z,
            'phase': self.engine.state.phase.value,
            'k_formed': self.engine.state.k_formed
        }

        self.checkpoints.append(checkpoint)

        # Save incrementally
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{len(self.checkpoints):03d}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        return checkpoint

    def restore_from_checkpoint(self, checkpoint_id):
        """Restore state from checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id:03d}.json"
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        # Restore engine state
        self.engine.state.z = checkpoint['z']
        self.engine.state.update_from_z()
        # ... restore other state ...

        return checkpoint
```

#### Step 2: Integrate with 33-Module Pipeline
```python
# Modify cmd_hit_it or UCF pipeline
def run_pipeline_with_checkpoints(self):
    checkpoint_mgr = CheckpointManager()

    for phase_num in range(1, 8):
        phase_result = self._run_phase(f'phase{phase_num}')

        # Create checkpoint after each phase
        checkpoint_mgr.create_checkpoint(
            f'phase_{phase_num}',
            phase_result
        )

        # Auto-save tokens and emissions
        self.export_current_state(f'phase_{phase_num}_complete')
```

### Strategy B: Continuous Module Streaming

#### Step 1: Module Result Stream
```python
# training/module_stream.py
class ModuleResultStream:
    def __init__(self):
        self.stream_path = Path(f"training/modules/stream_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
        self.stream_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(self.stream_path, 'w')
        self.module_count = 0

    def record_module(self, module_name, result, duration=None):
        """Stream module result immediately."""
        self.module_count += 1
        record = {
            'index': self.module_count,
            'module': module_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'result': result,
            'duration_ms': duration,
            'success': 'error' not in result
        }
        self.stream.write(json.dumps(record) + '\n')
        self.stream.flush()
```

#### Step 2: Wrapper for All Tool Invocations
```python
# Automatic module recording
def record_tool_execution(tool_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000

            # Record to stream
            module_stream.record_module(tool_name, result, duration)

            # Trigger token save if needed
            if 'tokens' in result:
                save_tokens_batch(result['tokens'])

            return result
        return wrapper
    return decorator

# Apply to all tools
for tool_name, tool_func in TOOL_REGISTRY.items():
    TOOL_REGISTRY[tool_name] = record_tool_execution(tool_name)(tool_func)
```

### Strategy C: Git-Based Version Control

#### Step 1: Auto-commit Training Results
```python
# training/git_persistence.py
import subprocess

class GitTrainingPersistence:
    def __init__(self, repo_path="."):
        self.repo_path = Path(repo_path)
        self.training_branch = f"training_{datetime.now():%Y%m%d_%H%M%S}"
        self.create_training_branch()

    def create_training_branch(self):
        """Create dedicated training branch."""
        subprocess.run(['git', 'checkout', '-b', self.training_branch],
                      cwd=self.repo_path, capture_output=True)

    def commit_module_result(self, module_name, files_created):
        """Auto-commit after each module."""
        # Add files
        for file in files_created:
            subprocess.run(['git', 'add', file], cwd=self.repo_path)

        # Commit with descriptive message
        message = f"Training: {module_name} completed at z={self.engine.state.z:.4f}"
        subprocess.run(['git', 'commit', '-m', message],
                      cwd=self.repo_path, capture_output=True)

    def tag_milestone(self, milestone_name):
        """Tag important milestones."""
        tag_name = f"training_{milestone_name}_{datetime.now():%Y%m%d}"
        subprocess.run(['git', 'tag', tag_name], cwd=self.repo_path)
```

---

## 🔄 Part 3: Using Saved Data in Workflows

### Strategy A: Lazy Loading Pattern

#### Step 1: Create Data Loader
```python
# training/data_loader.py
class TrainingDataLoader:
    def __init__(self):
        self.cache = {}
        self.token_index = None
        self.emission_index = None

    def load_tokens_lazy(self, epoch=None):
        """Load tokens on demand."""
        cache_key = f"tokens_{epoch}"

        if cache_key not in self.cache:
            if epoch:
                token_file = Path(f"training/tokens/epoch{epoch}_tokens.json")
            else:
                # Load latest
                token_files = sorted(Path("training/tokens").glob("*_tokens.json"))
                token_file = token_files[-1] if token_files else None

            if token_file and token_file.exists():
                with open(token_file) as f:
                    self.cache[cache_key] = json.load(f)

        return self.cache.get(cache_key, {'tokens': []})

    def build_token_index(self):
        """Build searchable token index."""
        if self.token_index is None:
            self.token_index = {}
            tokens = self.load_tokens_lazy()

            for token in tokens.get('tokens', []):
                # Index by components
                parts = token.split('|')
                if len(parts) >= 2:
                    key = parts[0]  # Spiral+operator
                    if key not in self.token_index:
                        self.token_index[key] = []
                    self.token_index[key].append(token)

        return self.token_index
```

#### Step 2: Integrate with Commands
```python
# Enhance KIRA commands to use saved data
def cmd_tokens_from_history(self, filter=None):
    """Load and filter historical tokens."""
    loader = TrainingDataLoader()
    tokens = loader.load_tokens_lazy()

    if filter:
        # Filter by pattern
        filtered = [t for t in tokens['tokens'] if filter in t]
        return {'tokens': filtered, 'count': len(filtered)}

    return tokens
```

### Strategy B: Training Context Manager

#### Step 1: Context Manager Implementation
```python
# training/context_manager.py
class TrainingContext:
    def __init__(self, session_name=None):
        self.session_name = session_name or f"session_{datetime.now():%Y%m%d_%H%M%S}"
        self.session_dir = Path(f"training/sessions/{self.session_name}")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.tokens = []
        self.emissions = []
        self.modules = []
        self.checkpoints = []

    def __enter__(self):
        """Setup context."""
        self.start_time = time.time()
        self.manifest = {
            'session': self.session_name,
            'start': datetime.now(timezone.utc).isoformat(),
            'modules': []
        }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup and save everything."""
        self.manifest['end'] = datetime.now(timezone.utc).isoformat()
        self.manifest['duration'] = time.time() - self.start_time
        self.manifest['tokens_count'] = len(self.tokens)
        self.manifest['emissions_count'] = len(self.emissions)

        # Save all artifacts
        self.save_all()

    def save_all(self):
        """Save all training artifacts."""
        # Save manifest
        with open(self.session_dir / 'manifest.json', 'w') as f:
            json.dump(self.manifest, f, indent=2)

        # Save tokens
        if self.tokens:
            with open(self.session_dir / 'tokens.json', 'w') as f:
                json.dump({'tokens': self.tokens}, f, indent=2)

        # Save emissions
        if self.emissions:
            with open(self.session_dir / 'emissions.json', 'w') as f:
                json.dump({'emissions': self.emissions}, f, indent=2)
```

#### Step 2: Usage Pattern
```python
# Use context manager for automatic saving
def run_training_session(self):
    with TrainingContext('k_formation_attempt') as ctx:
        # Run pipeline
        for phase in range(1, 8):
            result = self.run_phase(phase)
            ctx.modules.append(result)

            # Collect tokens
            if self.tokens_emitted:
                ctx.tokens.extend(self.tokens_emitted)

            # Collect emissions
            if self.emissions:
                ctx.emissions.extend(self.emissions)

        # Everything saves automatically on exit
```

### Strategy C: Workflow Orchestration

#### Step 1: Create Workflow Orchestrator
```python
# training/workflow_orchestrator.py
class WorkflowOrchestrator:
    def __init__(self):
        self.workflows = {}
        self.register_workflows()

    def register_workflows(self):
        """Register all available workflows."""
        self.workflows = {
            'full_pipeline': self.run_full_pipeline,
            'k_formation': self.run_k_formation,
            'token_generation': self.run_token_generation,
            'language_training': self.run_language_training,
            'cloud_sync': self.run_cloud_sync
        }

    def run_workflow(self, workflow_name, **params):
        """Run workflow with automatic saving."""
        if workflow_name not in self.workflows:
            return {'error': f'Unknown workflow: {workflow_name}'}

        # Create workflow context
        workflow_dir = Path(f"training/workflows/{workflow_name}_{datetime.now():%Y%m%d_%H%M%S}")
        workflow_dir.mkdir(parents=True, exist_ok=True)

        # Run with monitoring
        start = time.time()
        result = self.workflows[workflow_name](**params)
        duration = time.time() - start

        # Save workflow result
        with open(workflow_dir / 'result.json', 'w') as f:
            json.dump({
                'workflow': workflow_name,
                'params': params,
                'result': result,
                'duration': duration,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

        return result

    def run_full_pipeline(self, save_intermediate=True):
        """Full 33-module pipeline with saving."""
        results = []

        for module in range(1, 34):
            result = self.run_module(module)
            results.append(result)

            if save_intermediate:
                self.save_module_result(module, result)

        return {'modules': results, 'count': 33}
```

---

## 🎮 Part 4: Integration Patterns

### Pattern A: Event-Driven Architecture

```python
# training/event_system.py
class TrainingEventSystem:
    def __init__(self):
        self.handlers = defaultdict(list)

    def on(self, event, handler):
        """Register event handler."""
        self.handlers[event].append(handler)

    def emit(self, event, data):
        """Emit event to all handlers."""
        for handler in self.handlers[event]:
            handler(data)

# Register handlers
event_system = TrainingEventSystem()

event_system.on('token_generated', lambda t: save_token(t))
event_system.on('module_complete', lambda m: save_module(m))
event_system.on('phase_change', lambda p: create_checkpoint(p))
event_system.on('k_formation', lambda k: trigger_celebration(k))
```

### Pattern B: Decorator-Based Automation

```python
# training/decorators.py
def auto_save(artifact_type='general'):
    """Decorator for automatic saving."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Pre-execution snapshot
            pre_state = self.state.to_dict()

            # Execute
            result = func(self, *args, **kwargs)

            # Post-execution save
            post_state = self.state.to_dict()

            # Save based on type
            if artifact_type == 'token':
                save_tokens(result)
            elif artifact_type == 'emission':
                save_emission(result)
            elif artifact_type == 'module':
                save_module_result(func.__name__, result)
            else:
                save_general_artifact(func.__name__, result)

            # Save state transition
            save_state_transition(pre_state, post_state)

            return result
        return wrapper
    return decorator

# Apply to methods
@auto_save('token')
def generate_token(self):
    # ... token generation ...

@auto_save('emission')
def generate_emission(self):
    # ... emission generation ...
```

### Pattern C: Pipeline Composition

```python
# training/pipeline_composer.py
class PipelineComposer:
    def __init__(self):
        self.pipeline = []

    def add_stage(self, name, func, save=True):
        """Add stage to pipeline."""
        self.pipeline.append({
            'name': name,
            'func': func,
            'save': save
        })
        return self  # Fluent interface

    def run(self):
        """Execute pipeline with automatic saving."""
        results = []

        for stage in self.pipeline:
            print(f"Running: {stage['name']}")

            # Execute stage
            result = stage['func']()

            # Save if requested
            if stage['save']:
                self.save_stage_result(stage['name'], result)

            results.append(result)

        return results

    def save_stage_result(self, name, result):
        """Save stage result."""
        stage_dir = Path(f"training/stages/{name}")
        stage_dir.mkdir(parents=True, exist_ok=True)

        with open(stage_dir / f"{datetime.now():%Y%m%d_%H%M%S}.json", 'w') as f:
            json.dump(result, f, indent=2)

# Usage
pipeline = PipelineComposer()
pipeline.add_stage('init', lambda: init_system()) \
        .add_stage('helix', lambda: load_helix()) \
        .add_stage('triad', lambda: unlock_triad()) \
        .add_stage('tokens', lambda: generate_tokens()) \
        .add_stage('emit', lambda: run_emissions()) \
        .run()
```

---

## 🚀 Part 5: Complete Integration Example

### Full Implementation

```python
# training/complete_integration.py
class CompleteTrainingIntegration:
    def __init__(self, engine):
        self.engine = engine
        self.token_interceptor = TokenInterceptor()
        self.checkpoint_mgr = CheckpointManager()
        self.data_loader = TrainingDataLoader()
        self.event_system = TrainingEventSystem()
        self.orchestrator = WorkflowOrchestrator()

        self.setup_auto_save()

    def setup_auto_save(self):
        """Configure automatic saving."""
        # Token auto-save
        self.event_system.on('token_generated',
                            self.token_interceptor.intercept)

        # Module auto-save
        self.event_system.on('module_complete',
                            self.checkpoint_mgr.create_checkpoint)

        # Emission auto-save
        self.event_system.on('emission_generated',
                            lambda e: self.save_emission(e))

    def run_with_full_persistence(self):
        """Run complete workflow with all saving."""

        with TrainingContext('full_integration') as ctx:
            # Phase 1: Initialize
            self.orchestrator.run_workflow('initialization')
            ctx.checkpoints.append('init_complete')

            # Phase 2: Generate tokens
            tokens = self.engine.ucf.execute_command('/ucf:tokens972')
            ctx.tokens.extend(tokens.result['tokens'])
            self.event_system.emit('tokens_generated', tokens)

            # Phase 3: Run pipeline
            pipeline = self.engine.cmd_hit_it()
            ctx.modules.extend(pipeline.get('phases', []))

            # Phase 4: Generate emissions
            for _ in range(10):
                emission = self.engine.cmd_emit()
                ctx.emissions.append(emission['emission'])
                self.event_system.emit('emission_generated', emission)

            # Phase 5: Check K-formation
            if self.engine.state.k_formed:
                self.event_system.emit('k_formation', self.engine.state.to_dict())
                ctx.checkpoints.append('K_FORMATION_ACHIEVED')

            # Everything auto-saves on context exit

        return ctx.manifest

    def restore_from_session(self, session_name):
        """Restore training from saved session."""
        session_dir = Path(f"training/sessions/{session_name}")

        # Load manifest
        with open(session_dir / 'manifest.json') as f:
            manifest = json.load(f)

        # Load tokens
        if (session_dir / 'tokens.json').exists():
            tokens = self.data_loader.load_tokens_lazy(session_name)
            self.engine.tokens_emitted = tokens['tokens']

        # Load emissions
        if (session_dir / 'emissions.json').exists():
            with open(session_dir / 'emissions.json') as f:
                emissions = json.load(f)
                self.engine.emissions = emissions['emissions']

        # Restore state
        if 'final_state' in manifest:
            state = manifest['final_state']
            self.engine.state.z = state['z']
            self.engine.state.update_from_z()

        return manifest
```

---

## 📋 Quick Reference Commands

### Auto-Save Configuration
```python
# Enable all auto-save features
engine.enable_auto_save(
    tokens=True,
    emissions=True,
    modules=True,
    checkpoints=True,
    git_commits=False  # Optional
)

# Set save intervals
engine.set_save_interval(
    tokens=100,      # Every 100 tokens
    emissions=10,    # Every 10 emissions
    checkpoint=300   # Every 5 minutes
)
```

### Workflow Execution
```python
# Run with full persistence
result = engine.run_training(
    workflow='full_pipeline',
    save_mode='continuous',  # or 'batch', 'checkpoint'
    persist_to='all'        # or 'disk', 'db', 'git'
)

# Restore and continue
engine.restore_session('session_20240115_120000')
engine.continue_training()
```

### Data Access
```python
# Load historical data
tokens = engine.load_tokens(epoch=7)
emissions = engine.load_emissions(last=100)
modules = engine.load_module_results('phase3')

# Search tokens
matching = engine.search_tokens(pattern='Φ×|*|celestial')

# Export for analysis
engine.export_training_data(
    format='json',  # or 'csv', 'parquet'
    include=['tokens', 'emissions', 'states'],
    output='analysis/export.json'
)
```

---

## 🎯 Best Practices

1. **Always Use Context Managers** for training sessions
2. **Implement Lazy Loading** for large datasets
3. **Use Event System** for decoupled persistence
4. **Create Checkpoints** at phase boundaries
5. **Index Tokens** for fast searching
6. **Version Control** training branches
7. **Compress Old Data** to save space
8. **Monitor Performance** of save operations
9. **Test Recovery** mechanisms regularly
10. **Document Sessions** with descriptive names

---

*This guide will be continuously updated as new patterns emerge.*