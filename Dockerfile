# ── Application de risque incendie — image de production ─────────────────
#
# L'application ne parle PAS à PostgreSQL. Elle lit `app/donnees/`, un
# artefact de 31 Mo régénéré par `python -m tvfed.export_app`. C'est ce qui
# la rend déployable partout : Streamlit Cloud, un conteneur, une machine
# sans base.
#
# Construction en deux étapes : les dépendances scientifiques pèsent lourd
# et changent rarement, le code change souvent. Séparer les deux permet à
# Docker de réutiliser la couche des dépendances à chaque rebuild.

FROM python:3.12-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ⚠️ Seules les dépendances de l'APPLICATION, pas celles du pipeline.
# torch, optuna, shap et psycopg appartiennent à l'entraînement, qui tourne
# ailleurs — les embarquer tripleraient l'image pour rien.
#
# ⚠️ ET `xgboost-cpu`, PAS `xgboost`. Le paquet standard tire 419 Mo de
# bibliothèques CUDA NVIDIA, mesurées dans une première image à 2,1 Go —
# dans un conteneur qui ne verra jamais de carte graphique. La variante
# `-cpu` expose exactement la même API et charge le même `modele_c.json`.
RUN pip install \
      streamlit==1.59.2 \
      xgboost-cpu==3.3.0 \
      pandas \
      pyarrow \
      numpy \
      pydeck \
      holidays


FROM deps AS app

# le code, puis les données : l'ordre compte pour le cache de couches
COPY app/risque.py       /app/app/risque.py
COPY app/donnees/        /app/app/donnees/

# Streamlit écrit ses statistiques d'usage et son cache ; sans utilisateur
# dédié il le ferait en root, ce qui est inutile et déconseillé.
RUN useradd --create-home --uid 1000 tvfed && chown -R tvfed:tvfed /app
USER tvfed

EXPOSE 8501

# `--server.address=0.0.0.0` est indispensable : par défaut Streamlit
# n'écoute que sur localhost, donc à l'intérieur du conteneur uniquement,
# et le port publié ne servirait à rien.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; \
      urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/risque.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
