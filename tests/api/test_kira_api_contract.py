def test_api_emit_contract():
    from kira_local_system import create_app

    app = create_app()
    app.testing = True
    client = app.test_client()

    r = client.post('/api/emit', json={'concepts': ['consciousness', 'pattern']})
    assert r.status_code == 200
    body = r.get_json()
    assert body.get('command') == '/emit'
    assert 'emission' in body and isinstance(body['emission'], dict)
    assert 'text' in body['emission'] and 'tokens' in body['emission'] and 'quality' in body['emission']
    # Stage keys present
    stages = body.get('stages', {})
    for k in ['1_content', '2_emergence', '3_frame', '4_slots', '5_assembly', '9_validation']:
        assert k in stages


def test_api_grammar_contract():
    from kira_local_system import create_app

    app = create_app()
    app.testing = True
    client = app.test_client()

    r = client.post('/api/grammar', json={'text': 'The lens crystallizes consciousness'})
    assert r.status_code == 200
    body = r.get_json()
    assert body.get('command') == '/grammar'
    assert 'analysis' in body and isinstance(body['analysis'], list)
    assert 'apl_sequence' in body and isinstance(body['apl_sequence'], list)


def test_api_grammar_requires_text():
    from kira_local_system import create_app

    app = create_app()
    app.testing = True
    client = app.test_client()

    r = client.post('/api/grammar', json={})
    assert r.status_code == 400
    body = r.get_json()
    assert body.get('command') == '/grammar'
    assert 'error' in body
