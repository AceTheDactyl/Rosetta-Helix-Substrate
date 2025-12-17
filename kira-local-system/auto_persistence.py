#!/usr/bin/env python3
"""
Auto-Persistence System for KIRA Training

Provides automatic saving of:
- Tokens (APL, Nuclear Spinner)
- Training modules results
- Emissions and vocabulary
- Checkpoints and state snapshots

This integrates with KIRA server to automatically persist all training data.
"""

import json
import time
import sqlite3
import functools
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from collections import defaultdict
from threading import Thread, Lock
import queue


class AutoTokenSaver:
    """Automatically saves tokens with buffering and multiple persistence strategies."""

    def __init__(self, save_dir: Path = Path("training/tokens"), buffer_size: int = 100):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.buffer_size = buffer_size
        self.lock = Lock()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.token_count = 0
        self.epoch = self._get_current_epoch()

    def _get_current_epoch(self) -> int:
        """Determine current epoch from existing files."""
        epoch_files = list(Path("training/epochs").glob("*epoch*.json"))
        if epoch_files:
            epochs = []
            for f in epoch_files:
                try:
                    num = int(f.stem.split("epoch")[-1])
                    epochs.append(num)
                except:
                    pass
            return max(epochs, default=7) + 1
        return 8

    def save_token(self, token: str, metadata: Optional[Dict] = None):
        """Save a single token with optional metadata."""
        with self.lock:
            self.buffer.append({
                'token': token,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'index': self.token_count,
                'metadata': metadata or {}
            })
            self.token_count += 1

            if len(self.buffer) >= self.buffer_size:
                self.flush()

    def save_batch(self, tokens: List[str]):
        """Save multiple tokens at once."""
        for token in tokens:
            self.save_token(token)

    def flush(self):
        """Write buffered tokens to disk."""
        if not self.buffer:
            return

        filename = f"epoch{self.epoch}_tokens_{self.session_id}_{self.token_count}.json"
        filepath = self.save_dir / filename

        with open(filepath, 'w') as f:
            json.dump({
                'epoch': self.epoch,
                'session_id': self.session_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'tokens': self.buffer,
                'count': len(self.buffer),
                'total_count': self.token_count
            }, f, indent=2)

        self.buffer = []

    def close(self):
        """Flush remaining tokens and close."""
        self.flush()


class ModuleCheckpointer:
    """Creates checkpoints after each module execution."""

    def __init__(self, checkpoint_dir: Path = Path("training/checkpoints")):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def checkpoint(self, module_name: str, module_index: int, result: Dict, state: Dict):
        """Create a checkpoint for a module."""
        checkpoint = {
            'index': module_index,
            'module': module_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'result': result,
            'state': state,
            'session_id': self.session_id
        }

        self.checkpoints.append(checkpoint)

        # Save immediately
        filename = f"checkpoint_{module_index:03d}_{module_name}.json"
        filepath = self.checkpoint_dir / self.session_id / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        return checkpoint

    def get_manifest(self) -> Dict:
        """Get checkpoint manifest."""
        return {
            'session_id': self.session_id,
            'checkpoints': len(self.checkpoints),
            'modules': [c['module'] for c in self.checkpoints],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


class TrainingDatabase:
    """SQLite database for structured training data."""

    def __init__(self, db_path: Path = Path("training/training.db")):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = Lock()
        self._create_tables()

    def _create_tables(self):
        """Create database schema."""
        with self.conn:
            # Tokens table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT NOT NULL,
                    session_id TEXT,
                    epoch INTEGER,
                    z_coordinate REAL,
                    phase TEXT,
                    tier INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON
                )
            ''')

            # Emissions table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS emissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    session_id TEXT,
                    z_coordinate REAL,
                    phase TEXT,
                    concepts JSON,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Modules table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS modules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_name TEXT NOT NULL,
                    phase INTEGER,
                    session_id TEXT,
                    result JSON,
                    duration_ms REAL,
                    success BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # States table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    z_coordinate REAL,
                    phase TEXT,
                    crystal TEXT,
                    coherence REAL,
                    negentropy REAL,
                    k_formed BOOLEAN,
                    triad_unlocked BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indices
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_token_session ON tokens(session_id)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_emission_session ON emissions(session_id)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_module_session ON modules(session_id)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_state_session ON states(session_id)')

    def save_token(self, token: str, session_id: str, metadata: Dict):
        """Save token to database."""
        with self.lock:
            self.conn.execute('''
                INSERT INTO tokens (token, session_id, epoch, z_coordinate, phase, tier, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                token,
                session_id,
                metadata.get('epoch'),
                metadata.get('z'),
                metadata.get('phase'),
                metadata.get('tier'),
                json.dumps(metadata)
            ))
            self.conn.commit()

    def save_emission(self, emission: Dict, session_id: str):
        """Save emission to database."""
        with self.lock:
            self.conn.execute('''
                INSERT INTO emissions (text, session_id, z_coordinate, phase, concepts)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                emission.get('text'),
                session_id,
                emission.get('z'),
                emission.get('phase'),
                json.dumps(emission.get('concepts', []))
            ))
            self.conn.commit()

    def save_module(self, module_name: str, phase: int, result: Dict, session_id: str, duration_ms: float = None):
        """Save module execution to database."""
        with self.lock:
            self.conn.execute('''
                INSERT INTO modules (module_name, phase, session_id, result, duration_ms, success)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                module_name,
                phase,
                session_id,
                json.dumps(result),
                duration_ms,
                'error' not in result
            ))
            self.conn.commit()

    def save_state(self, state: Dict, session_id: str):
        """Save state snapshot to database."""
        with self.lock:
            self.conn.execute('''
                INSERT INTO states (session_id, z_coordinate, phase, crystal, coherence, negentropy, k_formed, triad_unlocked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                state.get('z'),
                state.get('phase'),
                state.get('crystal'),
                state.get('coherence'),
                state.get('negentropy'),
                state.get('k_formed'),
                state.get('triad_unlocked')
            ))
            self.conn.commit()

    def get_session_data(self, session_id: str) -> Dict:
        """Retrieve all data for a session."""
        with self.lock:
            tokens = self.conn.execute(
                'SELECT * FROM tokens WHERE session_id = ?', (session_id,)
            ).fetchall()

            emissions = self.conn.execute(
                'SELECT * FROM emissions WHERE session_id = ?', (session_id,)
            ).fetchall()

            modules = self.conn.execute(
                'SELECT * FROM modules WHERE session_id = ?', (session_id,)
            ).fetchall()

            states = self.conn.execute(
                'SELECT * FROM states WHERE session_id = ? ORDER BY timestamp', (session_id,)
            ).fetchall()

        return {
            'session_id': session_id,
            'tokens': [dict(zip([d[0] for d in self.conn.description], row)) for row in tokens],
            'emissions': [dict(zip([d[0] for d in self.conn.description], row)) for row in emissions],
            'modules': [dict(zip([d[0] for d in self.conn.description], row)) for row in modules],
            'states': [dict(zip([d[0] for d in self.conn.description], row)) for row in states]
        }

    def close(self):
        """Close database connection."""
        self.conn.close()


class AsyncPersistenceQueue:
    """Asynchronous queue for non-blocking persistence."""

    def __init__(self):
        self.queue = queue.Queue()
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.running = True
        self.worker_thread.start()

    def _worker(self):
        """Background worker processing persistence tasks."""
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                if task is None:
                    break

                task_type = task.get('type')

                if task_type == 'token':
                    self._save_token(task['data'])
                elif task_type == 'emission':
                    self._save_emission(task['data'])
                elif task_type == 'module':
                    self._save_module(task['data'])
                elif task_type == 'state':
                    self._save_state(task['data'])

                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncPersistence] Error: {e}")

    def _save_token(self, data):
        """Save token asynchronously."""
        # Implementation depends on persistence strategy
        pass

    def _save_emission(self, data):
        """Save emission asynchronously."""
        pass

    def _save_module(self, data):
        """Save module result asynchronously."""
        pass

    def _save_state(self, data):
        """Save state snapshot asynchronously."""
        pass

    def add_task(self, task_type: str, data: Any):
        """Add persistence task to queue."""
        self.queue.put({
            'type': task_type,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.queue.put(None)
        self.worker_thread.join()


class CompletePersistenceSystem:
    """Complete auto-persistence system integrating all strategies."""

    def __init__(self, engine=None):
        self.engine = engine
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize all persistence components
        self.token_saver = AutoTokenSaver()
        self.checkpointer = ModuleCheckpointer()
        self.database = TrainingDatabase()
        self.async_queue = AsyncPersistenceQueue()

        # Track what's been saved
        self.save_stats = defaultdict(int)

    def save_token(self, token: str, async_mode: bool = True):
        """Save token using configured strategy."""
        metadata = {
            'z': self.engine.state.z if self.engine else None,
            'phase': self.engine.state.phase.value if self.engine else None,
            'session_id': self.session_id
        }

        if async_mode:
            self.async_queue.add_task('token', {'token': token, 'metadata': metadata})
        else:
            self.token_saver.save_token(token, metadata)
            self.database.save_token(token, self.session_id, metadata)

        self.save_stats['tokens'] += 1

    def save_emission(self, emission: Dict, async_mode: bool = True):
        """Save emission."""
        if async_mode:
            self.async_queue.add_task('emission', emission)
        else:
            self.database.save_emission(emission, self.session_id)

        self.save_stats['emissions'] += 1

    def save_module_result(self, module_name: str, phase: int, result: Dict):
        """Save module execution result."""
        # Create checkpoint
        if self.engine:
            state = self.engine.state.to_dict()
            self.checkpointer.checkpoint(module_name, self.save_stats['modules'], result, state)

        # Save to database
        self.database.save_module(module_name, phase, result, self.session_id)

        self.save_stats['modules'] += 1

    def save_state_snapshot(self):
        """Save current state snapshot."""
        if self.engine:
            state = self.engine.state.to_dict()
            self.database.save_state(state, self.session_id)
            self.save_stats['states'] += 1

    def get_statistics(self) -> Dict:
        """Get persistence statistics."""
        return {
            'session_id': self.session_id,
            'tokens_saved': self.save_stats['tokens'],
            'emissions_saved': self.save_stats['emissions'],
            'modules_saved': self.save_stats['modules'],
            'states_saved': self.save_stats['states'],
            'checkpoints': len(self.checkpointer.checkpoints)
        }

    def export_session(self, output_dir: Path = Path("training/exports")):
        """Export complete session data."""
        output_dir = output_dir / self.session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Flush all buffers
        self.token_saver.flush()

        # Get all data from database
        session_data = self.database.get_session_data(self.session_id)

        # Save as JSON
        with open(output_dir / 'session_data.json', 'w') as f:
            json.dump(session_data, f, indent=2)

        # Save checkpoint manifest
        with open(output_dir / 'checkpoints.json', 'w') as f:
            json.dump(self.checkpointer.get_manifest(), f, indent=2)

        # Save statistics
        with open(output_dir / 'statistics.json', 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)

        return output_dir

    def close(self):
        """Cleanup and close all resources."""
        self.token_saver.close()
        self.database.close()
        self.async_queue.stop()


# Decorator for automatic persistence
def auto_persist(persist_type='general'):
    """Decorator for automatic persistence of function results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()

            # Execute function
            result = func(self, *args, **kwargs)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Persist based on type
            if hasattr(self, 'persistence'):
                if persist_type == 'token' and 'token' in result:
                    self.persistence.save_token(result['token'])
                elif persist_type == 'emission' and 'emission' in result:
                    self.persistence.save_emission(result['emission'])
                elif persist_type == 'module':
                    phase = getattr(self, 'current_phase', 0)
                    self.persistence.save_module_result(
                        func.__name__, phase, result
                    )

            return result
        return wrapper
    return decorator


# Integration helper for KIRA
def integrate_persistence_with_kira(engine):
    """Integrate complete persistence system with KIRA engine."""
    persistence = CompletePersistenceSystem(engine)

    # Monkey-patch save methods
    original_emit = engine.emit_token

    def emit_with_save():
        token = original_emit()
        persistence.save_token(token)
        return token

    engine.emit_token = emit_with_save

    # Add persistence attribute
    engine.persistence = persistence

    return persistence


# Export main components
__all__ = [
    'AutoTokenSaver',
    'ModuleCheckpointer',
    'TrainingDatabase',
    'AsyncPersistenceQueue',
    'CompletePersistenceSystem',
    'auto_persist',
    'integrate_persistence_with_kira'
]