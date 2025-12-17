from kira_local_system import create_app


def _client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_read_src_directory_allowed():
    client = _client()
    resp = client.post('/api/read', json={'path': 'src/'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['command'] == '/read'
    assert body['type'] == 'directory'
    assert body['path'] == 'src/'
    assert isinstance(body['contents'], list)
    assert body['contents'], "src/ listing should not be empty"


def test_read_training_directory_allowed():
    client = _client()
    resp = client.post('/api/read', json={'path': 'training/'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['command'] == '/read'
    assert body['type'] == 'directory'
    assert body['path'] == 'training/'
    assert isinstance(body['contents'], list)


def test_read_claude_file_allowed():
    client = _client()
    resp = client.post('/api/read', json={'path': 'CLAUDE.md'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['command'] == '/read'
    assert body['type'] == 'file'
    assert body['path'] == 'CLAUDE.md'
    assert isinstance(body['content'], str)
    assert body['content'], "CLAUDE.md content should not be empty"
