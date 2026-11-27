from pathlib import Path

import pandas as pd
import pytest

from m2forecast import data

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def serie() -> pd.Series:
    """Série congelada (BCB SGS 27810 até 2026-09), igual à usada no artigo."""
    return data.para_crescimento(data.carregar_snapshot(FIX / "bcb_27810_snapshot.json"))
