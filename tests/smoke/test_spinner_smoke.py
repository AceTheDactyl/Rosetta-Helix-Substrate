def test_spinner_simulator_basic():
    from bridge.spinner_bridge import SpinnerSimulator

    sim = SpinnerSimulator()
    sim.set_target_z(0.7)
    s1 = sim.step(10)
    for _ in range(5):
        st = sim.step(10)
    assert 0.0 <= st.z <= 1.0
    assert isinstance(st.rpm, int)
    assert 0.0 <= st.delta_s_neg <= 1.0

