from types import SimpleNamespace

import pytest

from llm import client as c


def fake_response(text="hi", model="model-x"):
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    msg = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage, model=model)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


MSG = [{"role": "user", "content": "hello"}]


def test_chat_returns_parsed_response():
    r = c.chat(FakeClient([fake_response("ok", "qwen")]), "qwen", MSG)
    assert r.text == "ok"
    assert r.total_tokens == 5
    assert r.served_model == "qwen"


def test_chat_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(c.time, "sleep", lambda s: None)
    fake = FakeClient([RuntimeError("busy"), RuntimeError("busy"), fake_response()])
    assert c.chat(fake, "qwen", MSG).text == "hi"
    assert fake.calls == 3


def test_chat_raises_after_all_retries(monkeypatch):
    monkeypatch.setattr(c.time, "sleep", lambda s: None)
    fake = FakeClient([RuntimeError("x")] * 4)
    with pytest.raises(c.LLMError):
        c.chat(fake, "qwen", MSG, retries=4)
    assert fake.calls == 4


def test_non_retryable_status_stops_immediately(monkeypatch):
    monkeypatch.setattr(c.time, "sleep", lambda s: None)
    err = RuntimeError("bad key")
    err.status_code = 401
    fake = FakeClient([err, fake_response()])
    with pytest.raises(c.LLMError):
        c.chat(fake, "qwen", MSG)
    assert fake.calls == 1


@pytest.mark.parametrize("name", ["auto", "freellmapi/auto", "freellmapi/auto-smart", "openrouter/auto"])
def test_router_aliases_rejected(name):
    assert c.is_router_alias(name)
    with pytest.raises(ValueError):
        c.chat(FakeClient([fake_response()]), name, MSG)


def test_router_alias_allowed_when_explicit():
    r = c.chat(FakeClient([fake_response()]), "freellmapi/auto", MSG, allow_router_alias=True)
    assert r.text == "hi"


def test_pinned_models_are_not_aliases():
    assert not c.is_router_alias("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    assert not c.is_router_alias("autoformer-model")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        c.get_client("openrouter")


def test_local_needs_no_key(monkeypatch):
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)
    assert c.get_client("local") is not None


def test_unknown_backend():
    with pytest.raises(ValueError):
        c.get_client("nope")
