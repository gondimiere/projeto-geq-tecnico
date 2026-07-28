"""
Vendas de GLP por UF e distribuidora (ANP/SIMP) — base da aba de Demanda &
Market Share.

Decisoes de agregacao, todas verificadas contra o dado antes de codar:

1. Filtra embalagem "P 13" e mercado "REVENDA DE GLP" — botijao residencial
   vendido via revendedor, consistente com o resto do painel. "CONSUMIDOR
   FINAL" (venda direta/B2B) fica de fora.

2. Soma sobre UF Origem: um agente pode suprir a mesma UF a partir de varias
   bases. Sem essa soma, a participacao sairia fragmentada.

3. NACIONAL GAS = "NACIONAL GAS BUTANO DISTRIBUIDORA LTDA" + "NGC
   DISTRIBUIDORA DE GAS LTDA." nos anos em que a NGC aparece (2021-2023).
   Isso NAO e escolha: a propria ANP instrui a somar as duas no periodo de
   transicao da aquisicao da Liquigas (aprovada pelo CADE em 18/11/2020), e
   a NGC foi formalmente incorporada em mai/2023. Sem somar, a serie mostra
   uma queda artificial em 2021 seguida de recuperacao — um degrau que e
   contabil, nao de mercado.

4. Quantidades negativas (21 linhas em 52.599, somando -0,061 de 79.079 mil
   toneladas) sao estornos. Ficam na soma: liquidam corretamente e o efeito
   e da ordem de 0,0001%.

Checagem de sanidade executada: a soma das participacoes de todos os agentes
fecha em 1,000000 nas 5.508 celulas (UF x mes) — sem excecao.

LIMITE DE COBERTURA: o arquivo historico granular por distribuidora vai ate
DEZ/2023. Os meses posteriores existem apenas agregados (sem distribuidora).
Isso e restricao de defesa da concorrencia da propria ANP e precisa aparecer
na interface — nao no rodape.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CSV = RAIZ / "GLP" / "GLP_Vendas_Historico.csv"
CACHE = Path(__file__).resolve().parents[1] / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

COLUNAS = ["ano", "mes", "agente", "cod_produto", "produto", "regiao_origem",
           "uf_origem", "regiao_dest", "uf_destino", "mercado", "embalagem",
           "qtd_mil_ton"]

NACIONAL = "NACIONAL GAS BUTANO DISTRIBUIDORA LTDA"
NGC = "NGC DISTRIBUIDORA DE GAS LTDA."

KG_POR_BOTIJAO = 13
MIL_TON_PARA_BOTIJOES = 1_000_000 / KG_POR_BOTIJAO   # 1 mil ton = 1e6 kg


def carregar(force=False) -> tuple[pd.DataFrame, datetime]:
    """Devolve o painel (ano, mes, uf, agente, volume, share) com cache."""
    cache = CACHE / "vendas_uf_agente.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache), datetime.fromtimestamp(cache.stat().st_mtime)

    if not CSV.exists():
        raise FileNotFoundError(f"CSV de vendas nao encontrado: {CSV}")

    d = pd.read_csv(CSV, sep=";", encoding="latin-1", decimal=",")
    d.columns = COLUNAS

    f = d[(d["embalagem"] == "P 13") & (d["mercado"] == "REVENDA DE GLP")].copy()

    # consolida NGC dentro da Nacional Gas (ver nota 3 no topo)
    f["agente"] = f["agente"].replace({NGC: NACIONAL})

    g = (f.groupby(["ano", "mes", "uf_destino", "agente"], as_index=False)["qtd_mil_ton"]
           .sum())

    tot = (g.groupby(["ano", "mes", "uf_destino"], as_index=False)["qtd_mil_ton"]
             .sum().rename(columns={"qtd_mil_ton": "total_uf"}))
    g = g.merge(tot, on=["ano", "mes", "uf_destino"])
    g["share"] = np.where(g["total_uf"] > 0, g["qtd_mil_ton"] / g["total_uf"], np.nan)

    g["data"] = pd.to_datetime(dict(year=g["ano"], month=g["mes"], day=1))
    g["botijoes"] = g["qtd_mil_ton"] * MIL_TON_PARA_BOTIJOES
    g["botijoes_uf"] = g["total_uf"] * MIL_TON_PARA_BOTIJOES

    g = g.sort_values(["uf_destino", "agente", "data"]).reset_index(drop=True)
    g.to_parquet(cache, index=False)
    return g, datetime.now()


def painel_nacional(g: pd.DataFrame) -> pd.DataFrame:
    """Serie mensal da Nacional Gas por UF, com o mercado total da UF."""
    nac = g[g["agente"] == NACIONAL][
        ["data", "ano", "mes", "uf_destino", "qtd_mil_ton", "botijoes", "share",
         "total_uf", "botijoes_uf"]
    ].copy()
    nac = nac.rename(columns={"qtd_mil_ton": "vol_nacional",
                              "botijoes": "botijoes_nacional"})

    # UFs onde a Nacional Gas nao vendeu em certos meses: share = 0, nao NA —
    # ausencia de venda e informacao, nao dado faltante
    grade = (g[["data", "uf_destino", "total_uf", "botijoes_uf"]]
             .drop_duplicates(subset=["data", "uf_destino"]))
    out = grade.merge(
        nac[["data", "uf_destino", "vol_nacional", "botijoes_nacional", "share"]],
        on=["data", "uf_destino"], how="left")
    out[["vol_nacional", "botijoes_nacional", "share"]] = \
        out[["vol_nacional", "botijoes_nacional", "share"]].fillna(0.0)
    out["ano"] = out["data"].dt.year
    out["mes"] = out["data"].dt.month
    return out.sort_values(["uf_destino", "data"]).reset_index(drop=True)


def concentracao_uf(g: pd.DataFrame) -> pd.DataFrame:
    """HHI e CR4 por UF e mes, a partir das participacoes ja calculadas."""
    def _agg(sub):
        s = sub["share"].to_numpy()
        s = s[s > 0]
        return pd.Series({
            "hhi": float(((s * 100) ** 2).sum()),
            "cr4": float(np.sort(s)[::-1][:4].sum() * 100),
            "n_agentes": int(len(s)),
        })

    out = (g.groupby(["data", "uf_destino"])[["share"]]
             .apply(_agg, include_groups=False).reset_index())
    return out


def cobertura(g: pd.DataFrame) -> dict:
    """Metadados de cobertura, para o aviso de defasagem na interface."""
    ultima = g["data"].max()
    hoje = pd.Timestamp.today().normalize().replace(day=1)
    meses_atraso = (hoje.year - ultima.year) * 12 + (hoje.month - ultima.month)
    return dict(
        inicio=g["data"].min(), fim=ultima,
        meses_atraso=int(meses_atraso),
        n_ufs=int(g["uf_destino"].nunique()),
        n_agentes=int(g["agente"].nunique()),
        n_celulas=int(g.groupby(["data", "uf_destino"]).ngroups),
    )
