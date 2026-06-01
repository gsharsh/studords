from __future__ import annotations

import pandas as pd

MATERIAL_ACTIVITY_TYPES = ["oucontent", "resource", "page", "subpage", "url"]


def merge_vle_activity_types(student_vle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    vle_cols = ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"]
    available_cols = [col for col in vle_cols if col in vle.columns]
    return student_vle.merge(
        vle[available_cols],
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )
