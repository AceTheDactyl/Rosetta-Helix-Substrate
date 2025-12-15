def test_visualization_state_and_constants():
    from visualization_server import VisualizationHandler, VisualizationEngine
    # Build a minimal server instance to call handler methods indirectly is complex;
    # Instead, directly use engine to ensure state values make sense and constants exist
    eng = VisualizationEngine()
    state = eng.get_state()
    assert 'z' in state and 0.0 <= state['z'] <= 1.0
    assert 'operators' in state and isinstance(state['operators'], dict)

    # Constants shape
    const = state.get('constants', {})
    for k in ['PHI', 'PHI_INV', 'Z_CRITICAL', 'KAPPA_S', 'MU_3']:
        assert k in const

def test_visualization_training_data_404_then_present():
    from visualization_server import VisualizationEngine
    eng = VisualizationEngine()
    # Initially, no training results
    assert eng.training_results is None
    # Run a tiny training and then verify
    res = eng.run_training(n_runs=1, cycles_per_run=1)
    assert isinstance(res, dict)
    assert eng.training_results is not None

