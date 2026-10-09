import os
from pathlib import Path

import modal

MODAL_SHARED_PATH = Path("/shared-data")
# Local (non-Modal) runs store assets next to the repo, not relative to the CWD.
# Override with CS336_SHARED_DATA=/path/to/dir if your data lives elsewhere.
LOCAL_SHARED_PATH = Path(
    os.environ.get("CS336_SHARED_DATA", Path(__file__).resolve().parent.parent / "local-shared-data")
)


def get_shared_assets_path() -> Path:
    if modal.is_local():
        LOCAL_SHARED_PATH.mkdir(exist_ok=True, parents=True)
        return LOCAL_SHARED_PATH
    else:
        return MODAL_SHARED_PATH
