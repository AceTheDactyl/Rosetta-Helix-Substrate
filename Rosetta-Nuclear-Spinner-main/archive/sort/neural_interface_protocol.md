# Neural Interface Protocol: Grid Cell Coupling at z_c

## Overview

Test whether grid cell ensembles show enhanced coupling to external drives 
operating at z = z_c = √3/2 = 0.866025... compared to other z values.

**Core insight:** Grid cells fire at 60° intervals. sin(60°) = √3/2 = z_c.
If the intrinsic geometry of grid cells is "tuned" to √3/2, an external 
signal structured around this value may couple more efficiently.

---

## 1. Signal Generation (Nuclear Spinner → Neural Interface)

The spinner generates a z-modulated signal. For neural interfacing, we 
don't use the magnetic field directly. Instead, we extract the spinner's 
state as an **electrical stimulus waveform**.

### 1.1 z-to-Waveform Mapping

```
z(t) → stimulus waveform parameters:

  Frequency:     f(z) = f_base + (f_max - f_base) * z
                 f_base = 4 Hz (theta band low)
                 f_max = 12 Hz (theta band high)
                 At z_c: f = 4 + 8 * 0.866 = 10.93 Hz

  Amplitude:     A(z) = A_max * ΔS_neg(z)
                 Peaks at z_c where ΔS_neg = 1.0

  Phase offset:  φ(z) = 2π * (z / φ) mod 2π
                 At z_c: φ = 2π * (0.866 / 1.618) = 1.065π rad = 191.7°
```

The stimulus waveform is:
```
V(t) = A(z(t)) * sin(2π * f(z(t)) * t + φ(z(t)))
```

### 1.2 Hexagonal Phase Modulation

Grid cells respond to movement through 2D space. To engage them without 
actual locomotion, we simulate hexagonal traversal via **phase cycling**:

```
Six-phase protocol (one period = 6 phases at 60° intervals):

  Phase 0:   0°    → baseline
  Phase 1:  60°    → sin(60°) = √3/2 = z_c  ← KEY
  Phase 2: 120°    → sin(120°) = √3/2 = z_c ← KEY  
  Phase 3: 180°    → sin(180°) = 0
  Phase 4: 240°    → sin(240°) = -√3/2
  Phase 5: 300°    → sin(300°) = -√3/2

The drive cycles through these phases, dwelling at each.
Prediction: Neural response strongest at phases 1, 2 (where sin = z_c).
```

### 1.3 Firmware Integration

Add to Nuclear Spinner firmware:

```c
typedef struct {
    float z;                    // Current z coordinate
    float delta_s_neg;          // Negentropy at current z
    float hex_phase_deg;        // Current hexagonal phase (0-360)
    float stimulus_freq_hz;     // Output frequency
    float stimulus_amplitude;   // Output amplitude (0-1)
    float stimulus_phase_rad;   // Output phase offset
} NeuralDriveState_t;

// Generate stimulus parameters from spinner state
NeuralDriveState_t compute_neural_drive(float z) {
    NeuralDriveState_t drive;
    drive.z = z;
    drive.delta_s_neg = compute_delta_s_neg(z);
    drive.stimulus_freq_hz = 4.0f + 8.0f * z;  // 4-12 Hz theta band
    drive.stimulus_amplitude = drive.delta_s_neg;  // Peaks at z_c
    drive.stimulus_phase_rad = 2.0f * M_PI * (z / PHI);
    return drive;
}

// Hexagonal phase cycling
void hex_phase_cycle(float dwell_time_s) {
    float phases[] = {0, 60, 120, 180, 240, 300};
    for (int i = 0; i < 6; i++) {
        float z_equiv = fabsf(sinf(phases[i] * M_PI / 180.0f));
        RotorControl_SetZ(z_equiv);
        // Hold and record
        HAL_Delay((uint32_t)(dwell_time_s * 1000));
    }
}
```

---

## 2. Neural Recording

### 2.1 Target Structure

**Medial Entorhinal Cortex (MEC), Layer II**

Grid cells are concentrated here. Recording options:
- **Acute:** Tetrode array, single session, anesthetized or head-fixed
- **Chronic:** Implanted electrode array, behaving animal, multiple sessions
- **Non-invasive (human):** High-density EEG/MEG over temporal cortex
  (lower resolution but ethically simpler)

### 2.2 Recording Parameters

```
Sampling rate:     30 kHz (spike detection)
Bandpass:          300 Hz - 6 kHz (spikes), 1-100 Hz (LFP)
Channels:          32+ for ensemble recording
Reference:         Cerebellar or skull screw (low neural activity)
```

### 2.3 Spike Sorting

Identify grid cells by:
1. Spatial firing rate map (if locomotion available) → hexagonal pattern
2. Gridness score > 0.3 (Sargolini et al. criteria)
3. Theta-phase precession (spikes shift earlier in theta cycle)

---

## 3. Experimental Protocol

### 3.1 Session Structure

```
Total duration: 90 minutes

Block 1 (Baseline, 15 min):
  - No external drive
  - Record spontaneous activity
  - Establish gridness scores

Block 2 (z-Sweep, 30 min):
  - Sweep z from 0.5 to 0.95 in 0.05 steps
  - 3 minutes per z value
  - Record spike counts, timing, LFP coherence

Block 3 (z_c Dwell, 15 min):
  - Hold at z = z_c = 0.866
  - Sustained K-formation
  - Test for entrainment saturation

Block 4 (Hexagonal Cycling, 20 min):
  - 6-phase protocol, 30s per phase
  - 10 complete cycles
  - Compare response at 60°/120° vs other phases

Block 5 (Recovery, 10 min):
  - No drive
  - Test for aftereffects
  - Compare to Block 1 baseline
```

### 3.2 Control Conditions

```
Control A: Drive at z = 0.5 (sub-threshold, MEMORY tier)
Control B: Drive at z = 0.95 (past z_c, in META tier)
Control C: Random z jitter (same mean, no structure)
Control D: Non-grid cells (same region, different cell type)
```

### 3.3 Predictions

| Condition | Predicted Outcome |
|-----------|-------------------|
| z = z_c | Maximum phase-locking, highest MI, gridness preserved |
| z < z_c | Weak coupling, gridness unaffected |
| z > z_c | Coupling present but weaker than z_c (past resonance peak) |
| Random z | No systematic coupling, gridness may degrade |
| Non-grid cells | No differential response to z_c (negative control) |

---

## 4. Analysis Pipeline

### 4.1 Phase-Locking Value (PLV)

```python
def compute_plv(spike_times, drive_phase_at_spike):
    """
    PLV = |mean(exp(i * phase))|
    Range: 0 (uniform) to 1 (perfect locking)
    """
    phases = drive_phase_at_spike
    plv = np.abs(np.mean(np.exp(1j * phases)))
    return plv
```

### 4.2 Mutual Information

```python
def compute_mi(z_values, spike_counts, n_bins=10):
    """
    I(Z; Spikes) = H(Spikes) - H(Spikes | Z)
    """
    # Discretize
    z_binned = np.digitize(z_values, np.linspace(0.5, 1.0, n_bins))
    spike_binned = np.digitize(spike_counts, 
                               np.percentile(spike_counts, np.linspace(0, 100, n_bins)))
    
    # Joint and marginal distributions
    p_joint = joint_histogram(z_binned, spike_binned)
    p_z = p_joint.sum(axis=1)
    p_spike = p_joint.sum(axis=0)
    
    # MI calculation
    mi = entropy(p_spike) - conditional_entropy(p_joint, p_z)
    return mi
```

### 4.3 Resonance Curve Fitting

```python
def fit_resonance(z_values, coupling_metric):
    """
    Fit Gaussian centered at z_c to test resonance hypothesis
    
    Model: C(z) = A * exp(-σ * (z - z_c)²) + baseline
    
    If best-fit center ≈ 0.866, resonance hypothesis supported.
    """
    from scipy.optimize import curve_fit
    
    def gaussian(z, A, center, sigma, baseline):
        return A * np.exp(-sigma * (z - center)**2) + baseline
    
    popt, pcov = curve_fit(gaussian, z_values, coupling_metric,
                           p0=[1.0, 0.866, 36, 0.1])  # Initial guess: z_c
    
    fitted_center = popt[1]
    center_std = np.sqrt(pcov[1, 1])
    
    return {
        'center': fitted_center,
        'center_std': center_std,
        'matches_zc': abs(fitted_center - 0.866025) < 2 * center_std
    }
```

---

## 5. Hardware Requirements

### 5.1 Stimulus Isolation

The spinner's electrical output must be galvanically isolated from 
the neural recording system to prevent artifacts.

```
Spinner DAC → Optical isolator → Stimulus isolator → Electrode/TMS coil
                                          ↓
                                    Current source
                                    (constant current)
```

### 5.2 Timing Synchronization

```
Master clock: Spinner MCU (STM32H7 @ 480 MHz)
  ↓
Sync pulse: 1 PPS (pulse per second) → Recording system
  ↓
Timestamp alignment: < 100 μs jitter requirement
```

### 5.3 Stimulus Modalities (choose based on preparation)

| Modality | Invasiveness | Spatial precision | Temporal precision |
|----------|--------------|-------------------|-------------------|
| Electrical (μA) | High | ~100 μm | < 1 ms |
| Optogenetic | High (requires viral transfection) | ~50 μm | < 1 ms |
| TMS | Non-invasive | ~1 cm | ~1 ms |
| tACS | Non-invasive | ~2-3 cm | < 1 ms |

For initial human studies: **tACS (transcranial alternating current stimulation)**
For rodent studies: **Optogenetic or electrical**

---

## 6. Ethical Considerations

### 6.1 Animal Studies
- IACUC approval required
- Minimize animal numbers (power analysis)
- Humane endpoints defined

### 6.2 Human Studies (if applicable)
- IRB approval required
- Informed consent with clear description of:
  - What spinner does (generates oscillating signal)
  - What z_c means (specific frequency/phase combination)
  - What we're testing (neural coupling hypothesis)
- tACS safety limits: < 2 mA, < 30 min continuous
- Exclusion: epilepsy, implants, pregnancy

### 6.3 Consent Protocol (Helix Framework)

Per helix-kira skill:
1. Invoke consent_protocol before any neural interfacing
2. Record explicit YES
3. Honor conditions precisely
4. Allow revocation at any time

**This is non-negotiable.**

---

## 7. Expected Outcomes

### 7.1 If Hypothesis Confirmed

Resonance curve peaks at z_c = √3/2:
- Grid cells show enhanced coupling to drives structured around √3/2
- The geometric constant embedded in grid cell organization has 
  functional significance for external interfacing
- Opens path to: optimized brain-machine interfaces, 
  targeted neuromodulation, information-theoretic models of spatial coding

### 7.2 If Hypothesis Rejected

Coupling is independent of z:
- √3/2 appearance in grid cells is geometric consequence without 
  functional resonance implications
- The spinner still works as designed; the neural coupling 
  hypothesis was simply wrong
- Science proceeds

### 7.3 If Results Are Mixed

Coupling shows z-dependence but peak ≠ z_c:
- Indicates some structure but not the predicted one
- Requires model revision
- Most likely outcome (reality is usually messier than theory)

---

## 8. Firmware Extensions Required

New files to add to nuclear_spinner_firmware:

```
src/neural_interface.c      - Drive generation, hex cycling
include/neural_interface.h  - API definitions
src/stimulus_output.c       - DAC waveform generation
include/stimulus_output.h   - Waveform parameters
```

Key functions:
```c
// Initialize neural interface subsystem
HAL_Status_t NeuralInterface_Init(void);

// Set stimulus parameters from current z
HAL_Status_t NeuralInterface_UpdateFromZ(float z);

// Run hexagonal phase cycling protocol
HAL_Status_t NeuralInterface_HexCycle(float dwell_time_s, int n_cycles);

// Get current drive state for logging
NeuralDriveState_t NeuralInterface_GetState(void);

// Emergency stop (immediately zero output)
void NeuralInterface_EmergencyStop(void);
```

---

## 9. Summary

We're testing whether the mathematical constant √3/2—which appears both in 
grid cell geometry (60° periodicity) and quantum spin (spin-½ magnitude)—
has functional significance for neural coupling.

The Nuclear Spinner provides the drive signal.
The Helix coordinate system provides the framework (z_c = THE LENS).
The experiment provides the test.

**Null hypothesis:** z_c is just a number. No special coupling.
**Alternative:** z_c is a resonance point. Enhanced coupling at √3/2.

This is falsifiable. That makes it science.

---

Δ|neural-interface-protocol|z_c=0.866|Ω
