import types


class _FakeWUMBOAPLTrainingEngine:
    def __init__(self, n_oscillators=60):
        class _Cycle:
            z = 0.5

        self.wumbo_cycle = _Cycle()
        self._k = 0

    def training_step(self, val):
        self._k += 1
        self.wumbo_cycle.z = min(0.9, self.wumbo_cycle.z + 0.01)
        return {
            'wumbo_phase': 'A',
            'z': self.wumbo_cycle.z,
            'kappa': 0.5,
            'delta_s_neg': 0.1,
            'phase': 'PARADOX',
        }

    def get_session_summary(self):
        return {
            'wumbo_cycles': 1,
            'k_formations': 0,
            'final_z': self.wumbo_cycle.z,
            'final_phase': 'PARADOX',
        }


class _FakeWumboTrainer:
    def validate_coupling_conservation(self):
        return {'all_valid': True}

    def validate_threshold_ordering(self):
        return {'all_valid': True}

    def validate_gaussian_physics(self):
        return {'all_valid': True}

    def validate_kuramoto_dynamics(self):
        return {'all_coherence_valid': True}

    def validate_free_energy_principle(self):
        return {'free_energy_positive': True}

    def validate_phase_transitions(self):
        return {'all_valid': True}

    def train_all_sentences(self, steps_per_sentence=2):
        return {
            'sentence_results': {},
            'total_k_formations': 0,
            'sentences_trained': 7,
        }


class _FakeUnifiedOperatorEngine:
    def __init__(self, initial_z=0.5):
        self.z = initial_z
        self.scalar_state = 0.0

    def apply_operator(self, op, val):
        self.scalar_state += 0.01
        self.z = min(0.95, self.z + 0.01)
        return {'new_state': self.scalar_state, 'z': self.z, 'kappa_field': 0.6}

    def run_prs_cycle(self, ops):
        return {'cycle_count': len(ops), 'final_state': self.scalar_state, 'final_z': self.z, 'kappa_final': 0.6, 'conservation_error': 0.0}

    def validate_state(self):
        return {'all_valid': True, 'kappa_field_state': {'phase': 'ok', 'kappa': 0.6}}

    def get_summary(self):
        return {'ok': True}


class _FakeKappaFieldState:
    def __init__(self, z=0.5):
        self.z = z
        self.kappa = 0.6
        self.lambda_ = 0.4
        self.at_golden_balance = True

    def evolve(self, dz):
        self.z = max(0.0, min(1.0, self.z + dz))

    def get_phase(self):
        return 'PARADOX'


class _FakeKuramotoOscillator:
    def __init__(self, n_oscillators=60):
        pass

    def compute_coherence(self):
        return 0.1

    def evolve(self, steps=5):
        return [0.1] * steps


class _FakeFreeEnergyState:
    def __init__(self):
        self.F_history = []
        self.PE_history = []

    def step(self, obs):
        self.F_history.append(1.0)
        self.PE_history.append(0.1)


class _FakePhaseTransitionState:
    def __init__(self):
        self.transition_events = []
        self.current_phase = 'PARADOX'
        self.order_parameter = 0.1

    def update(self, z):
        pass


def test_wumbo_apl_runner_smoke(monkeypatch):
    # Seed fake dependency modules before importing runner
    import sys, types

    # training.wumbo_apl_automated_training
    mod_apl = types.ModuleType('training.wumbo_apl_automated_training')
    mod_apl.WUMBOAPLTrainingEngine = _FakeWUMBOAPLTrainingEngine
    mod_apl.WUMBOPhase = types.SimpleNamespace()
    mod_apl.WUMBO_PHASES = []
    mod_apl.TokenCategory = object
    mod_apl.compute_delta_s_neg = lambda z: 0.1
    mod_apl.get_phase = lambda z: 'PARADOX'
    mod_apl.PHI = 1.618
    mod_apl.PHI_INV = 0.618
    mod_apl.PHI_INV_SQ = 0.382
    mod_apl.Z_CRITICAL = 0.866
    mod_apl.SIGMA = 36
    mod_apl.COUPLING_CONSERVATION = 1.0
    sys.modules['training.wumbo_apl_automated_training'] = mod_apl

    # training.wumbo_integrated_training
    mod_wumbo = types.ModuleType('training.wumbo_integrated_training')
    mod_wumbo.WumboTrainer = _FakeWumboTrainer
    mod_wumbo.WumboTrainingState = object
    mod_wumbo.KuramotoOscillator = _FakeKuramotoOscillator
    mod_wumbo.FreeEnergyState = _FakeFreeEnergyState
    mod_wumbo.PhaseTransitionState = _FakePhaseTransitionState
    sys.modules['training.wumbo_integrated_training'] = mod_wumbo

    # src.n0_operator_integration
    mod_n0 = types.ModuleType('src.n0_operator_integration')
    mod_n0.UnifiedN0Validator = lambda: types.SimpleNamespace(validate_all=lambda: {"validations": {}, "all_valid": True})
    mod_n0.UnifiedOperatorEngine = _FakeUnifiedOperatorEngine
    mod_n0.KappaFieldState = _FakeKappaFieldState
    mod_n0.PRSCycleState = object
    sys.modules['src.n0_operator_integration'] = mod_n0

    import run_wumbo_apl_integrated as mod

    # Patch heavy classes with fakes
    monkeypatch.setattr(mod, 'WUMBOAPLTrainingEngine', _FakeWUMBOAPLTrainingEngine)
    monkeypatch.setattr(mod, 'WumboTrainer', _FakeWumboTrainer)
    monkeypatch.setattr(mod, 'UnifiedOperatorEngine', _FakeUnifiedOperatorEngine)
    monkeypatch.setattr(mod, 'KappaFieldState', _FakeKappaFieldState)
    monkeypatch.setattr(mod, 'KuramotoOscillator', _FakeKuramotoOscillator)
    monkeypatch.setattr(mod, 'FreeEnergyState', _FakeFreeEnergyState)
    monkeypatch.setattr(mod, 'PhaseTransitionState', _FakePhaseTransitionState)

    summary = mod.run_integrated_session()
    assert 'physics' in summary
    assert 'n0_laws_valid' in summary
    assert 'wumbo_apl_training' in summary


def test_wumbo_n0_runner_smoke(monkeypatch):
    import sys, types

    # Seed dependency modules before import
    mod_wumbo = types.ModuleType('training.wumbo_integrated_training')
    mod_wumbo.WumboTrainer = _FakeWumboTrainer
    mod_wumbo.PHI = 1.618
    mod_wumbo.PHI_INV = 0.618
    mod_wumbo.PHI_INV_SQ = 0.382
    mod_wumbo.Z_CRITICAL = 0.866
    mod_wumbo.SIGMA_NEG_ENTROPY = 36
    mod_wumbo.COUPLING_CONSERVATION = 1.0
    mod_wumbo.compute_delta_s_neg = lambda z: 0.1
    sys.modules['training.wumbo_integrated_training'] = mod_wumbo

    mod_n0 = types.ModuleType('src.n0_operator_integration')
    mod_n0.UnifiedN0Validator = lambda: types.SimpleNamespace(validate_all=lambda: {"validations": {}, "all_valid": True})
    mod_n0.UnifiedOperatorEngine = _FakeUnifiedOperatorEngine
    mod_n0.KappaFieldState = _FakeKappaFieldState
    mod_n0.PRSCycleState = object
    mod_n0.OPERATOR_SYMBOLS = {}
    sys.modules['src.n0_operator_integration'] = mod_n0

    mod_adapt = types.ModuleType('src.adaptive_operator_matrix')
    mod_adapt.AdaptiveOperatorMatrix = object
    mod_adapt.create_adaptive_matrix = lambda: None  # patched later
    mod_adapt.Operator = object
    mod_adapt.OperatorType = object
    mod_adapt.OperatorDomain = object
    sys.modules['src.adaptive_operator_matrix'] = mod_adapt

    import run_wumbo_n0_integrated as mod

    monkeypatch.setattr(mod, 'WumboTrainer', _FakeWumboTrainer)
    monkeypatch.setattr(mod, 'UnifiedOperatorEngine', _FakeUnifiedOperatorEngine)
    # Adaptive matrix create function may be heavy; monkeypatch to a simple fake
    class _FakeMatrix:
        num_rows = 3
        num_operators = 3
        global_kappa = 0.6
        global_lambda = 0.4
        coupling_conservation_error = 0.0
        evolution_count = 1

        def evolve(self, cycles=1):
            return {
                'initial_phase': 'PARADOX',
                'final_phase': 'TRUE',
                'balance_before': 0.5,
                'balance_after': 0.6,
                'conservation_error_after': 0.0,
            }

    monkeypatch.setattr(mod, 'create_adaptive_matrix', lambda: _FakeMatrix())

    res = mod.run_integrated_wumbo_n0()
    assert 'n0_validation' in res
    assert 'engine_summary' in res
    assert 'matrix_stats' in res
