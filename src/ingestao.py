"""
Ingestao de dados brutos, com cache local em disco (nao rebaixa a cada
refresh do app) e timestamp de atualizacao visivel na interface.

Fontes (todas publicas, sem chave de API):
  - FRED: propano Mont Belvieu (WPROPANEMBTX), Brent (DCOILBRENTEU)
  - BCB SGS serie 1: cambio PTAX venda
  - ANP/SDC: tabelas historicas de preco (ja extraidas por coordenada de
    caractere via pdfplumber - ver parse_anp_*.py na pasta pai)
  - ANP: volume mensal de GLP por distribuidora (xlsx)
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[2]      # PITCH_GLP_PASSTHROUGH/
CACHE = Path(__file__).resolve().parents[1] / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
INICIO = "2001-11-01"


def _cache_path(nome: str) -> Path:
    return CACHE / f"{nome}.parquet"


def _meta_path(nome: str) -> Path:
    return CACHE / f"{nome}.meta.json"


def _ler_cache(nome: str, max_idade_dias: float):
    p, m = _cache_path(nome), _meta_path(nome)
    if not (p.exists() and m.exists()):
        return None, None
    meta = json.loads(m.read_text(encoding="utf-8"))
    ts = datetime.fromisoformat(meta["atualizado_em"])
    if datetime.now() - ts > timedelta(days=max_idade_dias):
        return None, ts          # vencido, mas devolve o ts para fallback
    return pd.read_parquet(p), ts


def _gravar_cache(nome: str, df: pd.DataFrame):
    ts = datetime.now()
    df.to_parquet(_cache_path(nome), index=False)
    _meta_path(nome).write_text(
        json.dumps({"atualizado_em": ts.isoformat()}), encoding="utf-8"
    )
    return ts


def _fallback(nome: str):
    """Ultimo recurso: devolve o cache mesmo vencido, em vez de quebrar o app."""
    p, m = _cache_path(nome), _meta_path(nome)
    if p.exists() and m.exists():
        ts = datetime.fromisoformat(json.loads(m.read_text(encoding="utf-8"))["atualizado_em"])
        return pd.read_parquet(p), ts
    return None, None


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
def _fred(serie: str, nome_col: str, nome_cache: str, max_idade=1.0):
    df, ts = _ler_cache(nome_cache, max_idade)
    if df is not None:
        return df, ts
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}&cosd={INICIO}"
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = ["data", nome_col]
        d["data"] = pd.to_datetime(d["data"])
        d[nome_col] = pd.to_numeric(d[nome_col], errors="coerce")
        d = d.dropna().reset_index(drop=True)
        return d, _gravar_cache(nome_cache, d)
    except Exception:
        return _fallback(nome_cache)


def propano():
    return _fred("WPROPANEMBTX", "propano_usd_gal", "propano")


def brent():
    return _fred("DCOILBRENTEU", "brent_usd_bbl", "brent")


# ---------------------------------------------------------------------------
# BCB SGS (cambio PTAX) - em blocos de 9 anos (limite da API), com retry.
# A API do BCB falha de forma intermitente; sem retry um bloco inteiro some
# em silencio e abre um buraco de anos na serie. Aqui, se um bloco falhar
# apos 3 tentativas, o app registra o gap em vez de fingir que nao houve.
# ---------------------------------------------------------------------------
def cambio():
    df, ts = _ler_cache("cambio", 1.0)
    if df is not None:
        return df, ts
    try:
        fim = pd.Timestamp.today().normalize()
        bordas = pd.date_range(INICIO, fim, freq="9YS").tolist()
        if not bordas or bordas[-1] < fim:
            bordas.append(fim)

        partes, falhas = [], []
        for i in range(len(bordas) - 1):
            ini, f = bordas[i], min(bordas[i + 1], fim)
            url = (
                "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
                f"?formato=json&dataInicial={ini:%d/%m/%Y}&dataFinal={f:%d/%m/%Y}"
            )
            bloco = None
            for _ in range(3):
                try:
                    r = requests.get(url, headers=UA, timeout=60)
                    if r.status_code == 200 and r.text.strip().startswith("["):
                        bloco = pd.DataFrame(r.json())
                        if len(bloco):
                            break
                except Exception:
                    pass
                time.sleep(1)
            if bloco is None or not len(bloco):
                falhas.append(f"{ini:%Y-%m}..{f:%Y-%m}")
            else:
                partes.append(bloco)

        if not partes:
            return _fallback("cambio")

        d = pd.concat(partes, ignore_index=True)
        d["data"] = pd.to_datetime(d["data"], format="%d/%m/%Y")
        d["ptax"] = pd.to_numeric(d["valor"].str.replace(",", "."), errors="coerce")
        d = d[["data", "ptax"]].dropna().drop_duplicates("data").sort_values("data")
        d.attrs["falhas"] = falhas
        return d.reset_index(drop=True), _gravar_cache("cambio", d)
    except Exception:
        return _fallback("cambio")


# ---------------------------------------------------------------------------
# ANP/SDC - series historicas ja extraidas dos PDFs (pasta dados_brutos/)
# ---------------------------------------------------------------------------
def decomposicao_anp():
    """Decomposicao mensal completa do preco (produtor -> consumidor).

    Procura primeiro uma copia DENTRO do proprio repo (dados_estaticos/) -
    e' o que existe quando o app roda publicado, como repositorio autonomo,
    sem a pasta-mae com os dados brutos das outras etapas do projeto. Se nao
    achar, cai para o caminho de desenvolvimento local (RAIZ/dados_brutos),
    onde o arquivo e' gerado por parse_anp_decomposicao.py.
    """
    candidatos = [
        Path(__file__).resolve().parents[1] / "dados_estaticos" / "anp_decomposicao_nacional_2001_2026.csv",
        RAIZ / "dados_brutos" / "anp_decomposicao_nacional_2001_2026.csv",
    ]
    p = next((c for c in candidatos if c.exists()), None)
    if p is None:
        raise FileNotFoundError(
            "Falta anp_decomposicao_nacional_2001_2026.csv em "
            f"{candidatos[0]} nem em {candidatos[1]}. "
            "Rode parse_anp_decomposicao.py na pasta pai (extracao dos 26 PDFs)."
        )
    d = pd.read_csv(p)
    d["mes"] = pd.to_datetime(d["mes"])
    for c in d.columns:
        if c != "mes":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["tributos"] = d[["cide", "pis_cofins", "icms"]].fillna(0).sum(axis=1)
    ts = datetime.fromtimestamp(p.stat().st_mtime)
    return d.sort_values("mes").reset_index(drop=True), ts


# ---------------------------------------------------------------------------
# ANP - volume mensal por distribuidora (para HHI / CR4)
# ---------------------------------------------------------------------------
URL_VENDAS = ("https://www.gov.br/anp/pt-br/assuntos/distribuicao-e-revenda/"
              "distribuidor/distr/dm-glp/relatorio-vendas-recipiente.xlsx")


def distribuidoras():
    df, ts = _ler_cache("distribuidoras", 7.0)
    if df is not None:
        return df, ts
    try:
        r = requests.get(URL_VENDAS, headers=UA, timeout=90)
        r.raise_for_status()
        bruto = pd.read_excel(io.BytesIO(r.content), sheet_name=1, header=None)

        # acha a linha de cabecalho pelo conteudo ("MES"), em vez de contar
        # linhas de skip na mao - o topo tem celulas mescladas que deslocam
        # a contagem de um jeito nao obvio
        col0 = bruto[0].astype(str).str.strip().str.upper()
        linha_cab = col0[col0.str.match(r"^M[ÊE]S$", na=False)].index
        if len(linha_cab) == 0:
            raise ValueError("cabecalho 'MES' nao encontrado na planilha da ANP")
        i = int(linha_cab[0])

        d = bruto.iloc[i + 1:, :4].copy()
        d.columns = ["mes_bruto", "distribuidora", "p13_kg", "outros_kg"]

        # a coluna de mes vem ora como datetime (openpyxl ja converte), ora
        # como serial do Excel — dependendo de como a celula foi formatada.
        # Trata os dois casos; o que nao virar data em nenhum deles e descartado.
        como_data = pd.to_datetime(d["mes_bruto"], errors="coerce")
        serial = pd.to_numeric(d["mes_bruto"], errors="coerce")
        como_serial = pd.to_datetime("1899-12-30") + pd.to_timedelta(serial, unit="D")
        d["mes"] = como_data.fillna(como_serial)
        d = d[d["mes"].notna()]
        d["p13_kg"] = pd.to_numeric(d["p13_kg"], errors="coerce")
        d["outros_kg"] = pd.to_numeric(d["outros_kg"], errors="coerce")
        d = d[["mes", "distribuidora", "p13_kg", "outros_kg"]].dropna(subset=["distribuidora"])
        d["distribuidora"] = d["distribuidora"].astype(str).str.strip()
        return d.reset_index(drop=True), _gravar_cache("distribuidoras", d)
    except Exception:
        return _fallback("distribuidoras")
