"""Étape 16b — assemble les archives téléchargées en cartes de FWI.

    python -m tvfed.cartes_projections

Une carte par (scénario, horizon), plus la carte historique de référence.
Chacune est la moyenne du FWI de saison sur les 15 saisons de la fenêtre —
une climatologie, pas une année.

⚠️ POURQUOI 15 SAISONS ET NON 5. Un premier essai sur la seule période
2046-2050 donnait RCP8.5 EN DESSOUS de RCP4.5, ce qui n'a aucun sens
physique. C'était l'échantillon : la variabilité interannuelle du FWI est
énorme — 4,34 en 2021 contre 6,90 en 2020 sur les observations du projet.
"""
from __future__ import annotations

import zipfile

import numpy as np
import xarray as xr

from .paths import RACINE
from .projections import HORIZONS, SCENARIOS

PROJ = RACINE / "data" / "projections"
VAR = "fwi-mean-jjas"


def _carte(nom: str) -> tuple[np.ndarray, int]:
    """Moyenne du FWI de saison sur toutes les années de l'archive."""
    z = PROJ / f"fwi_{nom}.zip"
    if not z.exists():
        raise FileNotFoundError(f"{z.name} absent — lancer --telecharger")
    dossier = PROJ / nom
    if not dossier.exists():
        with zipfile.ZipFile(z) as a:
            a.extractall(dossier)
    fs = sorted(dossier.rglob("*.nc"))
    x = np.concatenate([xr.open_dataset(f)[VAR].values for f in fs], axis=0)
    return np.nanmean(x, axis=0), len(x)


def main() -> None:
    cartes, infos = {}, []

    c, n = _carte("historical")
    cartes["historique"] = c
    infos.append(("historique", n, float(np.nanmean(c))))

    for s in SCENARIOS:
        for h, (_, centre) in HORIZONS.items():
            nom = f"{s}__{h}"
            try:
                c, n = _carte(nom)
            except FileNotFoundError:
                print(f"  ⚠️ {nom} absent — ignoré")
                continue
            cartes[f"{s}_{centre}"] = c
            infos.append((f"{s} · centré {centre}", n, float(np.nanmean(c))))

    np.savez_compressed(PROJ / "cartes_fwi.npz", **cartes)

    f0 = sorted((PROJ / "historical").rglob("*.nc"))[0]
    d0 = xr.open_dataset(f0)
    np.savez_compressed(PROJ / "grille.npz", lat=d0.lat.values, lon=d0.lon.values)

    ref = infos[0][2]
    print(f"{'carte':26s} {'saisons':>8s} {'FWI moyen':>10s} {'vs 1986-2005':>13s}")
    print("─" * 62)
    for nom, n, m in infos:
        ecart = "" if nom == "historique" else f"{100 * (m / ref - 1):+11.1f} %"
        print(f"{nom:26s} {n:>8d} {m:>10.3f} {ecart:>13s}")
    print(f"\n✅ cartes_fwi.npz — {len(cartes)} cartes")


if __name__ == "__main__":
    main()
