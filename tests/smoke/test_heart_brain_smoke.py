def test_heart_brain_minimal():
    from heart import Heart
    from brain import Brain

    h = Heart(n_nodes=20, K=0.1, seed=1, initial_z=0.3)
    c0 = h.step(0.01)
    for _ in range(10):
        h.step(0.01)
    assert 0.0 <= h.z <= 1.0

    b = Brain(plates=12, seed=2)
    results = b.query(current_z=0.5, top_k=5)
    assert isinstance(results, list)
    b.consolidate(0.9)
    clusters = b.cluster_memories()
    assert isinstance(clusters, list)

