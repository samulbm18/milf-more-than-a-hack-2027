"""Locations and per-match constants. Everything else imports from here."""
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(REPO, "configs", "paths.yaml"), encoding="utf-8") as _f:
    _P = yaml.safe_load(_f)

MTAH_ROOT = os.path.abspath(os.environ.get("MTAH_ROOT") or os.path.join(REPO, _P["mtah_root"]))
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(MTAH_ROOT, "data")
CACHE = os.path.join(REPO, _P["cache_dir"])
RUNS = os.path.join(REPO, _P["runs_dir"])
OUTPUTS = os.path.join(REPO, _P["outputs_dir"])
for _d in (CACHE, RUNS, OUTPUTS):
    os.makedirs(_d, exist_ok=True)

GAME = "BAR-ATH"
STEM = os.path.join(DATA_DIR, "tactical", GAME + "_tactical")
ATD_TXT = STEM + "_atd.txt"
ATD_XML = STEM + "_atd.xml"
HOMOGRAPHY_CSV = STEM + "_homography.csv"
TAGS_XML = STEM + "_smart_tagging.xml"
# broadcast recording (its own timeline; sync.py puts it on the tactical one)
BC_STEM = os.path.join(DATA_DIR, "broadcast", GAME + "_broadcast")
BC_ATD_TXT = BC_STEM + "_atd.txt"
BC_HOMOGRAPHY_CSV = BC_STEM + "_homography.csv"
GT_PUBLIC_CSV = os.path.join(MTAH_ROOT, "evaluation", "gt", GAME + "_possession_gt_public.csv")

N_FRAMES = 199319
FPS = 25
PITCH_L, PITCH_W = 105.3, 68.0
HALF_L, HALF_W = PITCH_L / 2, PITCH_W / 2
IMG_W, IMG_H = 1920, 1080

# graded blocks (video frames, inclusive)
GRADED = {1: (11185, 90651), 2: (110808, 190197)}
# second-half kickoff; frames [GRADED[2][0], KICKOFF_2) are dead ball before it
KICKOFF_2 = 113742
# public ground-truth window (inclusive) and the 2-fold split used for selection
PUBLIC = (11185, 18684)
PUBLIC_SPLIT = 14935


def cache(name):
    return os.path.join(CACHE, name)
