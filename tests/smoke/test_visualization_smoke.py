def test_visualization_engine_basics():
    from visualization_server import VisualizationEngine

    eng = VisualizationEngine()
    state = eng.get_state()
    assert 0.0 <= state['z'] <= 1.0
    assert 'operators' in state and isinstance(state['operators'], dict)

    # Step once
    s2 = eng.step(0.01)
    assert 'work_extracted' in s2 and 'collapsed' in s2

    # Apply a legal or illegal operator; just assert structure
    res = eng.apply_operator('()')
    assert 'success' in res

