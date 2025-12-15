def test_visualization_training_schema():
    from visualization_server import VisualizationEngine

    eng = VisualizationEngine()
    res = eng.run_training(n_runs=1, cycles_per_run=1)

    required = [
        'initial_quality', 'final_quality', 'improvement_ratio',
        'total_lessons', 'total_sequences', 'quality_history',
        'operator_distribution', 'runs', 'meta_bridges',
        'physical_learners', 'liminal_generators'
    ]
    for key in required:
        assert key in res

    # Basic type checks
    assert isinstance(res['initial_quality'], (int, float))
    assert isinstance(res['final_quality'], (int, float))
    assert isinstance(res['improvement_ratio'], (int, float))
    assert isinstance(res['total_lessons'], int)
    assert isinstance(res['total_sequences'], int)
    assert isinstance(res['quality_history'], list)
    assert isinstance(res['operator_distribution'], dict)
    assert isinstance(res['runs'], int) and res['runs'] >= 1
