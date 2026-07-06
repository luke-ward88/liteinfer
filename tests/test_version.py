import liteinfer


def test_version_is_string():
    assert isinstance(liteinfer.__version__, str)
    assert liteinfer.__version__.count(".") >= 1
