def test_kira_package_api():
    from kira_local_system import create_app, get_engine, KIRAEngine

    app = create_app()
    assert app is not None
    eng = get_engine()
    assert eng is not None
    assert hasattr(KIRAEngine, '__name__')

    # Use Flask test client directly from packaged app
    app.testing = True
    client = app.test_client()
    r = client.get('/api/health')
    assert r.status_code == 200

