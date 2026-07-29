"""Connexion PostgreSQL et chargement en masse."""
from __future__ import annotations

import io
import os

import pandas as pd
import psycopg

DSN = os.environ.get(
    "TVFED_DSN", "host=localhost port=5433 dbname=tvfed user=tvfed password=tvfed"
)


def connexion() -> psycopg.Connection:
    return psycopg.connect(DSN)


def copier(df: pd.DataFrame, table: str, conn: psycopg.Connection | None = None) -> int:
    """Charge un DataFrame par COPY — largement plus rapide que des INSERT.

    Les colonnes du DataFrame doivent porter le nom des colonnes cibles.
    """
    fermer = conn is None
    conn = conn or connexion()
    tampon = io.StringIO()
    df.to_csv(tampon, index=False, header=False, na_rep="\\N")
    tampon.seek(0)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    try:
        with conn.cursor() as cur:
            with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
                cp.write(tampon.read())
        conn.commit()
    finally:
        if fermer:
            conn.close()
    return len(df)


def compter(table: str, conn: psycopg.Connection | None = None) -> int:
    fermer = conn is None
    conn = conn or connexion()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            return cur.fetchone()[0]
    finally:
        if fermer:
            conn.close()


def executer(sql: str, conn: psycopg.Connection | None = None) -> None:
    fermer = conn is None
    conn = conn or connexion()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    finally:
        if fermer:
            conn.close()
