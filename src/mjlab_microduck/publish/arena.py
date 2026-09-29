"""Entering a published policy on the Microduck Arena: one POST to its event's submit route.

The Arena races the repo at the revision just uploaded, on its own machine, and answers
with the time and the entry's page. It knows who is entering from the Hugging Face token
sent with the request, which it checks with the Hub and does not keep; so the token only
ever travels over https, or to this machine for a local `arena serve`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

ARENA = "https://pollen-robotics-microduck-arena.hf.space"
# The Arena answers once it has raced a whole sweep: half a minute, more when it is busy.
ENTER_TIMEOUT_S = 600.0
_THIS_MACHINE = {"127.0.0.1", "localhost"}


class EnterError(RuntimeError):
    """The Arena did not enter the policy. The message is its answer."""


@dataclass(frozen=True)
class Entry:
    run_id: str
    score: float
    seeds_finished: int
    seeds_total: int
    command_vx: float
    page_url: str
    already: bool


def check_arena(url: str) -> str:
    """The Arena's address without a trailing slash, or ValueError where a token must not go."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _THIS_MACHINE):
        return url.rstrip("/")
    raise ValueError(f"--arena {url}: your token travels with the entry, so the Arena must be https, "
                     "or http on 127.0.0.1 for a local one")


def enter(arena: str, event: str, token: str, *, repo: str, revision: str, name: str, livery: str,
          speed: float | None, code_url: str | None) -> Entry:
    fields = {"repo_id": repo, "revision": revision, "name": name, "livery": livery}
    if speed is not None:
        fields["command_vx"] = f"{speed:g}"
    if code_url:
        fields["env_url"] = code_url
    request = urllib.request.Request(
        f"{arena}/api/events/{urllib.parse.quote(event)}/submit",
        data=urllib.parse.urlencode(fields).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=ENTER_TIMEOUT_S) as answer:
            body = json.load(answer)
    except urllib.error.HTTPError as e:
        raise EnterError(f"{e.code}: {_detail(e)}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise EnterError(f"{arena} did not answer: {getattr(e, 'reason', e)}") from None
    return Entry(run_id=body["run_id"], score=body["score"], seeds_finished=body["seeds_finished"],
                 seeds_total=body["seeds_total"], command_vx=body["command_vx"], page_url=body["page_url"],
                 already=bool(body.get("already_entered")))


def _detail(e: urllib.error.HTTPError) -> str:
    """The Arena's own sentence; a gateway's HTML error page has none, only its status."""
    try:
        detail = json.loads(e.read()).get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail) if detail else e.reason


def entered_line(event: str, entry: Entry) -> str:
    said = "already entered" if entry.already else "entered"
    return (f"[publish] {said} on {event}: {entry.score:.3f} s ({entry.seeds_finished}/{entry.seeds_total} "
            f"seeds) at {entry.command_vx:.2f} m/s → {entry.page_url}")
