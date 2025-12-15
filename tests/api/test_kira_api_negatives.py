def _app_client():
    from kira_local_system import create_app
    app = create_app()
    app.testing = True
    return app.test_client()


def test_emit_invalid_concepts_type():
    client = _app_client()
    r = client.post('/api/emit', json={'concepts': 'not-a-list'})
    assert r.status_code == 400
    body = r.get_json()
    assert body.get('command') == '/emit'
    assert 'error' in body


def test_grammar_unsupported_content_type():
    client = _app_client()
    r = client.post('/api/grammar', data='text', content_type='text/plain')
    assert r.status_code in (400, 415)


def test_grammar_oversize_payload():
    client = _app_client()
    big = 'a' * (70 * 1024)
    r = client.post('/api/grammar', json={'text': big})
    assert r.status_code in (400, 413)


def test_read_path_traversal_forbidden():
    client = _app_client()
    r = client.post('/api/read', json={'path': '../setup.py'})
    assert r.status_code in (400, 403)
    body = r.get_json()
    assert body.get('command') == '/read'
    assert 'error' in body

