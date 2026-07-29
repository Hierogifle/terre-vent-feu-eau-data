"""Calendrier — variables temporelles row-local.

Aucun risque de fuite ici : chaque valeur ne dépend que de la date de sa
propre ligne, jamais d'une autre observation ni de y.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def construire(debut: str, fin: str) -> pd.DataFrame:
    dates = pd.date_range(debut, fin, freq="D")
    feries = holidays.France(years=range(dates.year.min(), dates.year.max() + 1))

    doy = dates.dayofyear.values
    mois = dates.month.values
    df = pd.DataFrame(
        {
            "date": dates,
            "annee": dates.year,
            "mois": mois,
            "doy": doy,
            "jour_semaine": dates.dayofweek,          # 0 = lundi
            "est_weekend": dates.dayofweek >= 5,
            "est_ferie": [d.date() in feries for d in dates],
            "nom_ferie": [feries.get(d.date()) for d in dates],
            "est_14_juillet": (mois == 7) & (dates.day.values == 14),
            "est_15_aout": (mois == 8) & (dates.day.values == 15),
            "vacances": False,   # ⚠️ à remplir depuis data.education.gouv.fr
            # encodage cyclique : le 31/12 et le 01/01 doivent être voisins
            "sin_doy": np.sin(2 * np.pi * doy / 365.25),
            "cos_doy": np.cos(2 * np.pi * doy / 365.25),
            "sin_mois": np.sin(2 * np.pi * mois / 12),
            "cos_mois": np.cos(2 * np.pi * mois / 12),
        }
    )
    df["date"] = df.date.dt.date
    return df
