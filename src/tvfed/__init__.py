"""Terre, Vent, Feu, Eau, Data.

⚠️ LA SORTIE CONSOLE EST FORCÉE EN UTF-8, ET CE N'EST PAS COSMÉTIQUE.
Sous Windows, `sys.stdout.encoding` vaut cp1252 par défaut. Trente-cinq
modules du paquet terminent leur travail par un `print("✅ …")` : sans cette
reconfiguration, ils lèvent UnicodeEncodeError *après* avoir écrit leurs
fichiers, ce qui donne un code de sortie non nul pour un traitement qui a
pourtant réussi. Un pipeline qui teste le code de retour conclurait à un
échec.

Mesuré sur cette machine : `python -m tvfed.vitrine` échouait sur la dernière
ligne alors que `docs/index.html` était déjà écrit.
"""
from __future__ import annotations

import sys

for _flux in (sys.stdout, sys.stderr):
    # `reconfigure` n'existe que sur les TextIOWrapper ; un flux capturé par
    # pytest ou redirigé vers un tube ne l'expose pas toujours.
    _rec = getattr(_flux, "reconfigure", None)
    if _rec is not None:
        try:
            _rec(encoding="utf-8")
        except (ValueError, OSError):
            pass
