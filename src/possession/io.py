"""Readers for the challenge assets. Raw ATD is parsed once and cached as .npy."""
import collections
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import paths

# Smart Tagging codes that are an exact mirror of another code (same spans, seen from the other
# team). The key is dropped, the value is kept.
TAG_MIRRORS = {
    "DEFENDING IN MIDDLE THIRD": "PROGRESSION",
    "DEFENDING IN ATTACKING THIRD": "BUILD UP",
    "DEFENDING IN DEFENSIVE THIRD": "CREATING CHANCES",
    "DEFENSIVE TRANSITION": "ATTACKING TRANSITION",
}


def load_atd_raw():
    """(n_frames, n_slots + 1, 3) float32 of normalised x, y and s; the last slot is the ball.

    Row i is video frame i (the text file has one line per frame, frame_id = i + 1).
    """
    npy = paths.cache("atd_raw.npy")
    if os.path.exists(npy):
        return np.load(npy)
    n_slots = len(load_meta()["slot_team"])
    rows = []
    with open(paths.ATD_TXT) as f:
        for line in f:
            tok = line.rstrip("\n").replace(":", ",").replace(";", ",").split(",")
            rows.append(np.array(tok[1:], dtype=np.float32))
    a = np.stack(rows).reshape(len(rows), n_slots + 1, 3)
    assert a.shape[0] == paths.N_FRAMES, a.shape
    np.save(npy, a)
    return a


def _params(el):
    return {pp.findtext("Name"): pp.findtext("Value") for pp in el.iter("ProviderParameter")}


def load_meta():
    root = ET.parse(paths.ATD_XML).getroot()
    teams = list(root.iter("Team"))
    team_ids = [t.get("id") for t in teams]
    players = list(root.iter("Player"))
    pp = [_params(p) for p in players]
    ptypes = sorted({p.get("position_type") or "" for p in pp})
    return {
        "team_names": [t.findtext("Name") for t in teams],
        "team_params": [_params(t) for t in teams],
        "slot_team": np.array([team_ids.index(p.get("teamId")) for p in players], np.int8),
        "slot_ptype": np.array([ptypes.index(p.get("position_type") or "") for p in pp], np.int8),
        "ptype_names": ptypes,
        "slot_gk": np.array(["goal" in (p.get("position_type") or "").lower() for p in pp], bool),
    }


def load_homography():
    """(n, 3, 3) matrices mapping normalised pitch coordinates to image pixels."""
    df = pd.read_csv(paths.HOMOGRAPHY_CSV)
    cols = ["h%d%d" % (r, c) for r in range(3) for c in range(3)]
    assert (df["frame"].values == np.arange(len(df))).all()
    return df[cols].values.reshape(-1, 3, 3), df["valid"].values.astype(bool)


def load_broadcast_ball():
    """The broadcast recording, on its own timeline: ball position xy [NB, 2] (normalised 0-1, NaN when
    the ball is not tracked) and wide [NB], True where a pitch homography exists, i.e. the wide shot
    is on air (False: replay, close-up, crowd)."""
    xy = []
    with open(paths.BC_ATD_TXT) as f:
        for line in f:
            tok = line.strip().rsplit(":", 1)[-1].split(",")
            xy.append((float(tok[0]), float(tok[1])))
    xy = np.array(xy, np.float32)
    df = pd.read_csv(paths.BC_HOMOGRAPHY_CSV, usecols=["frame", "valid"])
    wide = np.zeros(len(xy), bool)
    ok = df["frame"].values < len(xy)
    wide[df["frame"].values[ok]] = df["valid"].values[ok].astype(bool)
    return xy, wide


def load_tags():
    """Deduplicated Smart Tagging instances as a list of dicts with frame spans and team."""
    root = ET.parse(paths.TAGS_XML).getroot()
    out, seen = [], set()
    for ins in root.iter("instance"):
        code = ins.findtext("code")
        if code in TAG_MIRRORS:
            continue
        labels = {}
        for lab in ins.findall("label"):
            labels.setdefault(lab.findtext("group"), lab.findtext("text"))
        s, e = float(ins.findtext("start")), float(ins.findtext("end"))
        key = (code, round(s, 2), round(e, 2), labels.get("Team"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"code": code, "f0": int(round(s * paths.FPS)), "f1": int(round(e * paths.FPS)),
                    "team": labels.get("Team"), "labels": labels})
    return out


def load_gt_public():
    """Labels (array of 'H'/'A'/'D'/'X') of the public window, indexed by video frame."""
    df = pd.read_csv(paths.GT_PUBLIC_CSV)
    lab = np.full(paths.N_FRAMES, "X", dtype="<U1")
    lab[df["frame"].values] = df["label"].values
    return lab


def tag_team_index(name, team_names):
    """Map a tag's Team text to 0 (home) / 1 (away) / -1 (none)."""
    if not name or name.upper() == "N/A":
        return -1
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalpha())  # noqa: E731
    for i, t in enumerate(team_names):
        a, b = norm(name), norm(t)
        if a and b and (a in b or b in a or a[:5] == b[:5]):
            return i
    return -1


def counts(a):
    return dict(collections.Counter(a))
