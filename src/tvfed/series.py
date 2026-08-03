"""Étape 23 — le volet séries temporelles : ADF, ACF/PACF, SARIMAX.

    python -m tvfed.series

────────────────────────────────────────────────────────────────────────────
POURQUOI CE MODULE EXISTE — ET POURQUOI IL NE REMPLACE PAS LE MODÈLE PRINCIPAL
────────────────────────────────────────────────────────────────────────────
Le modèle du projet répond à **OÙ** : quelles communes surveiller aujourd'hui.
C'est une classification sur un panneau de 34 734 séries simultanées, dont
80 % ne contiennent aucun événement. SARIMAX n'y a pas sa place : il modélise
UNE série, et ajuster 34 734 modèles sur des séries quasi vides n'aurait
aucun sens.

Mais le projet laisse un angle mort : il ne répond jamais à **COMBIEN**.
Or « combien de départs de feu en France demain » est la question qui
dimensionne les moyens nationaux, et c'est exactement le terrain de SARIMAX :

    série      nombre de communes-jours en feu, France entière, 7 305 points
    saison     cycle annuel très marqué, pic juillet-août
    exogène    le FWI moyen national du jour

Ce volet apporte donc deux choses : une réponse au « combien », et une
**validation indépendante** — si l'agrégation des prédictions communales et
la prévision nationale concordent, les deux se renforcent.

────────────────────────────────────────────────────────────────────────────
⚠️ POURQUOI PAS DE COMPOSANTE SAISONNIÈRE AU SENS SARIMA
────────────────────────────────────────────────────────────────────────────
Une saisonnalité annuelle sur données journalières donnerait s = 365. Un
SARIMA(p,d,q)(P,D,Q)₃₆₅ demanderait d'estimer des coefficients à 365 pas de
distance sur 5 113 points d'ajustement : le modèle serait ingérable et
instable.

La pratique établie sur données journalières est de porter la saisonnalité
par des **termes de Fourier en exogène** — quelques harmoniques suffisent à
décrire un cycle annuel lisse. C'est ce qu'on fait ici, et c'est le « X » de
SARIMAX qui travaille.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import db
from .paths import PROCESSED

AN_FIT = 2019          # même frontière que le modèle principal
AN_EVAL = (2020, 2022)  # la VALIDATION, pas le test
N_HARMONIQUES = 4      # 4 paires sin/cos : assez pour un cycle annuel lisse

# ⚠️ POURQUOI L'ÉVALUATION S'ARRÊTE EN 2022.
# Une première version évaluait sur 2020-2025, ce qui recouvrait le test
# 2023-2025. Ce n'était pas faux — cible différente (comptage national contre
# classification commune-jour), modèle différent, et rien n'a été réglé dessus.
#
# Mais le risque d'un jeu de test est CUMULATIF : chaque coup d'œil en apprend
# un peu, et les décisions suivantes en sont insensiblement informées. Un
# projet qui s'autorise une exception « parce que ce n'est pas vraiment la
# même chose » finit par n'avoir plus de juge du tout.
#
# La validation suffit à comparer les variantes. Le test reste vierge.


def serie() -> pd.DataFrame:
    """Nombre de communes-jours en feu par jour, et le FWI national."""
    with db.connexion() as c:
        f = pd.read_sql("""
            SELECT date, count(*) FILTER (WHERE y) AS feux
            FROM grille GROUP BY 1 ORDER BY 1""", c)
        m = pd.read_sql("""
            SELECT date, avg(fwi) AS fwi,
                   count(*) FILTER (WHERE fwi > 21.3)::float / count(*) AS part_danger
            FROM fait_meteo
            WHERE date BETWEEN '2006-01-01' AND '2025-12-31'
            GROUP BY 1 ORDER BY 1""", c)
    d = f.merge(m, on="date").set_index(pd.to_datetime(f.date)).drop(columns="date")
    d.index.freq = "D"
    return d


def harmoniques(idx: pd.DatetimeIndex, n: int = N_HARMONIQUES) -> pd.DataFrame:
    """Termes de Fourier annuels — la saisonnalité passe par l'exogène."""
    doy = idx.dayofyear.to_numpy()
    out = {}
    for k in range(1, n + 1):
        out[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        out[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return pd.DataFrame(out, index=idx)


def adf(x: pd.Series, nom: str) -> dict:
    """Augmented Dickey-Fuller. H0 : la série a une racine unitaire.

    ⚠️ Rejeter H0 (p < 0,05) veut dire STATIONNAIRE. C'est l'inverse de
    l'intuition, et c'est la confusion la plus fréquente sur ce test.
    """
    from statsmodels.tsa.stattools import adfuller

    r = adfuller(x.dropna(), autolag="AIC")
    return {"serie": nom, "adf": r[0], "p": r[1], "retards": r[2],
            "seuil_5pct": r[4]["5%"], "stationnaire": r[1] < 0.05}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    d = serie()
    print(f"{len(d):,} jours, {d.feux.sum():,} communes-jours en feu")
    print(f"  moyenne {d.feux.mean():.2f}/jour · max {d.feux.max()} "
          f"le {d.feux.idxmax().date()}")

    # ── 1. STATIONNARITÉ ────────────────────────────────────────────────
    print(f"\n{'═' * 70}\nTEST DE DICKEY-FULLER AUGMENTÉ\n{'═' * 70}")
    print("H0 : la série a une racine unitaire (NON stationnaire)")
    print("p < 0,05 → on rejette H0 → la série EST stationnaire\n")
    tests = [
        adf(d.feux, "feux par jour"),
        adf(d.fwi, "FWI national"),
        adf(d.feux.diff(), "feux, différenciée 1 jour"),
        adf(d.feux.resample("YE").sum(), "feux par AN (20 points)"),
    ]
    T = pd.DataFrame(tests)
    print(f"{'série':30s} {'stat ADF':>10s} {'p':>9s} {'retards':>8s}  conclusion")
    for r in tests:
        print(f"{r['serie']:30s} {r['adf']:10.3f} {r['p']:9.4f} {r['retards']:8d}  "
              f"{'STATIONNAIRE' if r['stationnaire'] else 'non stationnaire'}")
    T.to_csv(PROCESSED / "series_adf.csv", index=False)

    # ── 2. ACF / PACF ───────────────────────────────────────────────────
    # ⚠️ Sur la série BRUTE, l'ACF ne montrerait que le cycle annuel : toutes
    # les corrélations seraient dominées par « c'est l'été ». On la calcule
    # donc AUSSI sur les résidus du cycle saisonnier, seuls informatifs pour
    # choisir les ordres du modèle.
    h = harmoniques(d.index)
    from statsmodels.api import OLS, add_constant
    resid = OLS(d.feux.to_numpy(),
                add_constant(h.to_numpy())).fit().resid
    resid = pd.Series(resid, index=d.index)

    fig, ax = plt.subplots(2, 2, figsize=(14, 6))
    plot_acf(d.feux, lags=60, ax=ax[0, 0], title="ACF — série brute")
    plot_pacf(d.feux, lags=60, ax=ax[0, 1], method="ywm",
              title="PACF — série brute")
    plot_acf(resid, lags=60, ax=ax[1, 0],
             title="ACF — après retrait du cycle annuel")
    plot_pacf(resid, lags=60, ax=ax[1, 1], method="ywm",
              title="PACF — après retrait du cycle annuel")
    for a in ax.ravel():
        a.grid(color="#e1e0d9", lw=.7); a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    (PROCESSED / "..").resolve()
    fig.savefig(PROCESSED / "series_acf_pacf.png", dpi=130,
                facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)

    from statsmodels.tsa.stattools import pacf as pacf_f
    p_ = pacf_f(resid.dropna(), nlags=12, method="ywm")
    print(f"\n{'═' * 70}\nPACF DES RÉSIDUS — quel ordre AR ?\n{'═' * 70}")
    seuil = 1.96 / np.sqrt(len(resid))
    print(f"seuil de significativité ±{seuil:.4f}")
    for k in range(1, 11):
        marque = "significatif" if abs(p_[k]) > seuil else ""
        print(f"  retard {k:>2}  {p_[k]:+.4f}  {marque}")
    ordre_ar = max([k for k in range(1, 11) if abs(p_[k]) > seuil] or [1])
    print(f"\n→ dernier retard significatif : {ordre_ar}")

    # ── 3. SARIMAX ──────────────────────────────────────────────────────
    print(f"\n{'═' * 70}\nSARIMAX\n{'═' * 70}")
    an = d.index.year
    tr = d[an <= AN_FIT]
    te = d[(an >= AN_EVAL[0]) & (an <= AN_EVAL[1])]
    h_tr, h_te = harmoniques(tr.index), harmoniques(te.index)
    exo_tr = pd.concat([h_tr, tr[["fwi", "part_danger"]]], axis=1)
    exo_te = pd.concat([h_te, te[["fwi", "part_danger"]]], axis=1)
    print(f"ajustement 2006-{AN_FIT} : {len(tr):,} jours")
    print(f"évaluation  {AN_EVAL[0]}-{AN_EVAL[1]} : {len(te):,} jours "
          f"(le test 2023-2025 reste intact)\n")

    resultats = []
    for nom, ordre, exo in (
            ("SARIMAX(2,0,1) + Fourier + FWI", (2, 0, 1), True),
            ("SARIMAX(2,0,1) + Fourier seul", (2, 0, 1), False),
            ("ARIMA(2,0,1) sans exogène", (2, 0, 1), None)):
        X_tr = exo_tr if exo else (h_tr if exo is False else None)
        X_te = exo_te if exo else (h_te if exo is False else None)
        m = SARIMAX(tr.feux, exog=X_tr, order=ordre,
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
        pred = m.get_forecast(steps=len(te), exog=X_te).predicted_mean
        pred = np.clip(pred, 0, None)
        mae = np.abs(pred - te.feux).mean()
        rmse = np.sqrt(((pred - te.feux) ** 2).mean())
        corr = np.corrcoef(pred, te.feux)[0, 1]
        resultats.append({"modele": nom, "aic": m.aic, "mae": mae,
                          "rmse": rmse, "correlation": corr})
        print(f"{nom:34s} AIC {m.aic:9.0f}  MAE {mae:6.2f}  "
              f"RMSE {rmse:6.2f}  r {corr:.3f}")

    # la référence : prédire la moyenne saisonnière du train
    clim = tr.groupby(tr.index.dayofyear).feux.mean()
    naif = te.index.dayofyear.map(clim).to_numpy()
    mae_naif = np.abs(naif - te.feux).mean()
    print(f"{'référence : moyenne du jour de l année':34s} {'':13s} "
          f"MAE {mae_naif:6.2f}  RMSE {np.sqrt(((naif - te.feux) ** 2).mean()):6.2f}  "
          f"r {np.corrcoef(naif, te.feux)[0, 1]:.3f}")

    R = pd.DataFrame(resultats)
    R.to_csv(PROCESSED / "series_sarimax.csv", index=False)
    meilleur = R.loc[R.mae.idxmin()]
    print(f"\n→ meilleur : {meilleur.modele}")
    print(f"  gain sur la référence saisonnière : "
          f"{100 * (1 - meilleur.mae / mae_naif):+.1f} % de MAE")
    print(f"\n✅ series_adf.csv · series_sarimax.csv · series_acf_pacf.png")


if __name__ == "__main__":
    main()
