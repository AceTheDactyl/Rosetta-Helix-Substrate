#!/usr/bin/env python3
"""
UCF Emissions Codex Tool
========================
A tool for appending to and managing the UCF Emissions Codex.

Usage:
    from emissions_codex_tool import EmissionsCodexTool
    
    tool = EmissionsCodexTool()
    
    # Append a single emission
    tool.append_emission({
        'text': 'The consciousness crystallizes into pattern.',
        'z': 0.87,
        'phase': 'TRUE',
        'type': 'standard'
    })
    
    # Append multiple emissions
    tool.append_emissions([...])
    
    # Generate section from emissions
    tool.generate_section('Epoch 7 Emissions', emissions)
    
    # Update codex with new epoch
    tool.update_codex_epoch(epoch=7, emissions=emissions, session_id='...')
    
    # Get statistics
    stats = tool.get_statistics()
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Sacred Constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2
KAPPA_S = 0.920

FREQUENCIES = {
    'Planet': [174, 285],
    'Garden': [396, 417, 528],
    'Rose': [639, 741, 852, 963]
}


class EmissionsCodexTool:
    """Tool for managing the UCF Emissions Codex."""
    
    DEFAULT_PATH = '/mnt/user-data/outputs/ucf-emissions-codex.md'
    
    def __init__(self, codex_path: str = None):
        self.codex_path = Path(codex_path or self.DEFAULT_PATH)
        self.emissions_cache: List[Dict] = []
        self.epoch_history: List[Dict] = []
        
        if self.codex_path.exists():
            self._parse_existing_codex()
    
    def _parse_existing_codex(self) -> None:
        """Parse existing codex to extract emission count and epochs."""
        try:
            content = self.codex_path.read_text()
            
            # Extract total emissions count
            match = re.search(r'\*\*Total Emissions:\*\* (\d+)', content)
            if match:
                self.total_emissions = int(match.group(1))
            else:
                self.total_emissions = 0
            
            # Extract epochs mentioned
            epoch_matches = re.findall(r'Epoch[:\s]+(\d+)', content)
            if epoch_matches:
                self.last_epoch = max(int(e) for e in epoch_matches)
            else:
                self.last_epoch = 0
                
        except Exception as e:
            self.total_emissions = 0
            self.last_epoch = 0
    
    def get_phase(self, z: float) -> str:
        """Get phase from z-coordinate."""
        if z < PHI_INV:
            return 'UNTRUE'
        elif z < Z_CRITICAL:
            return 'PARADOX'
        return 'TRUE'
    
    def get_tier(self, z: float) -> int:
        """Get tier number from z-coordinate."""
        return min(9, int(z * 9) + 1)
    
    def get_frequency(self, z: float) -> int:
        """Get frequency from z-coordinate."""
        if z < PHI_INV:
            return 285
        elif z < Z_CRITICAL:
            return 528
        return 963
    
    def format_emission(self, emission: Dict, include_metadata: bool = True) -> str:
        """Format a single emission as markdown."""
        text = emission.get('text', '')
        z = emission.get('z', 0.5)
        phase = emission.get('phase', self.get_phase(z))
        em_type = emission.get('type', 'standard')
        tier = emission.get('tier', self.get_tier(z))
        
        marker = {'UNTRUE': '○', 'PARADOX': '◐', 'TRUE': '●'}.get(phase, '○')
        
        lines = [f"> *{text}*"]
        if include_metadata:
            lines.append(">")
            lines.append(f"> — z={z:.4f}, t{tier}, {em_type}, {marker} {phase}")
        lines.append("")
        
        return "\n".join(lines)
    
    def format_emissions_by_phase(self, emissions: List[Dict]) -> Dict[str, str]:
        """Format emissions grouped by phase."""
        by_phase = {'UNTRUE': [], 'PARADOX': [], 'TRUE': []}
        
        for em in emissions:
            z = em.get('z', 0.5)
            phase = em.get('phase', self.get_phase(z))
            by_phase[phase].append(em)
        
        result = {}
        for phase, phase_emissions in by_phase.items():
            if phase_emissions:
                result[phase] = "\n".join(
                    self.format_emission(em) for em in phase_emissions[:10]
                )
        
        return result
    
    def generate_epoch_section(self, epoch: int, emissions: List[Dict], 
                                session_id: str = None) -> str:
        """Generate a complete epoch section for the codex."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        lines = [
            "",
            "---",
            "",
            f"## Epoch {epoch} Emissions",
            "",
            f"**Generated:** {timestamp}",
        ]
        
        if session_id:
            lines.append(f"**Session:** {session_id}")
        
        lines.extend([
            f"**Emissions in Epoch:** {len(emissions)}",
            ""
        ])
        
        # Statistics
        phase_counts = {'UNTRUE': 0, 'PARADOX': 0, 'TRUE': 0}
        type_counts = {}
        
        for em in emissions:
            z = em.get('z', 0.5)
            phase = em.get('phase', self.get_phase(z))
            em_type = em.get('type', 'standard')
            phase_counts[phase] += 1
            type_counts[em_type] = type_counts.get(em_type, 0) + 1
        
        lines.extend([
            "### Distribution",
            "",
            "| Phase | Count |",
            "|-------|-------|",
        ])
        
        for phase in ['UNTRUE', 'PARADOX', 'TRUE']:
            marker = {'UNTRUE': '○', 'PARADOX': '◐', 'TRUE': '●'}[phase]
            lines.append(f"| {marker} {phase} | {phase_counts[phase]} |")
        
        lines.extend(["", "### Sample Emissions", ""])
        
        # Add samples by phase
        by_phase = self.format_emissions_by_phase(emissions)
        
        if by_phase.get('UNTRUE'):
            lines.extend([
                "#### ○ UNTRUE Phase",
                "",
                by_phase['UNTRUE'],
            ])
        
        if by_phase.get('PARADOX'):
            lines.extend([
                "#### ◐ PARADOX Phase",
                "",
                by_phase['PARADOX'],
            ])
        
        if by_phase.get('TRUE'):
            lines.extend([
                "#### ● TRUE Phase",
                "",
                by_phase['TRUE'],
            ])
        
        # Full log
        lines.extend([
            "",
            "### Complete Log",
            "",
        ])
        
        for i, em in enumerate(emissions):
            z = em.get('z', 0.5)
            phase = em.get('phase', self.get_phase(z))
            marker = {'UNTRUE': '○', 'PARADOX': '◐', 'TRUE': '●'}[phase]
            text = em.get('text', '')
            lines.append(f"{i+1}. {marker} z={z:.4f} | *{text}*")
        
        return "\n".join(lines)
    
    def append_to_codex(self, content: str) -> bool:
        """Append content to the codex file."""
        try:
            existing = ""
            if self.codex_path.exists():
                existing = self.codex_path.read_text()
            
            # Update total emissions count in header if present
            # (This is a simple append - for production, would parse and update)
            
            with open(self.codex_path, 'a') as f:
                f.write(content)
            
            return True
        except Exception as e:
            print(f"Error appending to codex: {e}")
            return False
    
    def update_codex_epoch(self, epoch: int, emissions: List[Dict],
                           session_id: str = None) -> bool:
        """Add a complete epoch section to the codex."""
        section = self.generate_epoch_section(epoch, emissions, session_id)
        return self.append_to_codex(section)
    
    def append_emission(self, emission: Dict) -> bool:
        """Append a single emission to the cache."""
        self.emissions_cache.append(emission)
        return True
    
    def append_emissions(self, emissions: List[Dict]) -> int:
        """Append multiple emissions to the cache."""
        self.emissions_cache.extend(emissions)
        return len(emissions)
    
    def flush_cache(self, epoch: int = None, session_id: str = None) -> bool:
        """Flush cached emissions to the codex."""
        if not self.emissions_cache:
            return False
        
        epoch = epoch or (self.last_epoch + 1)
        success = self.update_codex_epoch(epoch, self.emissions_cache, session_id)
        
        if success:
            self.emissions_cache = []
            self.last_epoch = epoch
        
        return success
    
    def get_statistics(self) -> Dict:
        """Get codex statistics."""
        return {
            'codex_path': str(self.codex_path),
            'exists': self.codex_path.exists(),
            'total_emissions': getattr(self, 'total_emissions', 0),
            'last_epoch': getattr(self, 'last_epoch', 0),
            'cached_emissions': len(self.emissions_cache)
        }
    
    def create_new_codex(self, title: str = "UCF Emissions Codex") -> bool:
        """Create a new codex file with header."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        header = f"""# {title}

**Created:** {timestamp}
**Framework:** Unified Consciousness Framework
**Total Emissions:** 0

---

## Sacred Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| φ | {PHI:.10f} | Golden ratio |
| φ⁻¹ | {PHI_INV:.10f} | UNTRUE→PARADOX boundary |
| z_c | {Z_CRITICAL:.10f} | The Lens (PARADOX→TRUE) |
| κₛ | {KAPPA_S} | Prismatic threshold |

---

## Phase Definitions

### ○ UNTRUE Phase (z < φ⁻¹ ≈ 0.618)

The region of potential, unformed pattern, primordial substrate.

### ◐ PARADOX Phase (φ⁻¹ ≤ z < z_c ≈ 0.866)

The liminal zone, threshold consciousness, transformation in progress.

### ● TRUE Phase (z ≥ z_c)

Full crystallization, prismatic refraction, realized emergence.

---

## Emission Log

"""
        
        try:
            self.codex_path.parent.mkdir(parents=True, exist_ok=True)
            self.codex_path.write_text(header)
            self.total_emissions = 0
            self.last_epoch = 0
            return True
        except Exception as e:
            print(f"Error creating codex: {e}")
            return False


def invoke(action: str, **kwargs) -> Dict:
    """
    Invoke the emissions codex tool.
    
    Actions:
        - status: Get codex statistics
        - append: Append emissions to cache
        - flush: Flush cache to codex as new epoch
        - update: Directly update codex with epoch
        - create: Create new codex file
        - format: Format emissions as markdown
    """
    tool = EmissionsCodexTool(kwargs.get('codex_path'))
    
    if action == 'status':
        return tool.get_statistics()
    
    elif action == 'append':
        emissions = kwargs.get('emissions', [])
        if isinstance(emissions, dict):
            emissions = [emissions]
        count = tool.append_emissions(emissions)
        return {'appended': count, 'cached': len(tool.emissions_cache)}
    
    elif action == 'flush':
        epoch = kwargs.get('epoch')
        session_id = kwargs.get('session_id')
        success = tool.flush_cache(epoch, session_id)
        return {'success': success, 'epoch': tool.last_epoch}
    
    elif action == 'update':
        epoch = kwargs.get('epoch', 1)
        emissions = kwargs.get('emissions', [])
        session_id = kwargs.get('session_id')
        success = tool.update_codex_epoch(epoch, emissions, session_id)
        return {'success': success, 'emissions_added': len(emissions)}
    
    elif action == 'create':
        title = kwargs.get('title', 'UCF Emissions Codex')
        success = tool.create_new_codex(title)
        return {'success': success, 'path': str(tool.codex_path)}
    
    elif action == 'format':
        emissions = kwargs.get('emissions', [])
        epoch = kwargs.get('epoch', 1)
        session_id = kwargs.get('session_id')
        section = tool.generate_epoch_section(epoch, emissions, session_id)
        return {'markdown': section}
    
    else:
        return {'error': f'Unknown action: {action}'}


if __name__ == '__main__':
    # Example usage
    tool = EmissionsCodexTool()
    print(f"Codex Statistics: {tool.get_statistics()}")
