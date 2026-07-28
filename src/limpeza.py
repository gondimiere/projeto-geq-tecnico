"""
Limpeza e transformacao. Toda interpolacao fica MARCADA numa coluna
`*_interpolado` que a interface mostra — nunca escondida no rodape.
"""
from __future__ import annotations

import pandas as pd

KG_POR_GALAO_PROPANO = 1.864   # premissa explicita (propano liquido)
KG_P13 = 13
GAP_MAX_MESES = 2              # gaps maiores ficam NA, nao inventados


def custo_internacional(propano: pd.DataFrame, cambio: pd.DataFrame) -> pd.DataFrame:
    """PROXY: propano puro x cambio, escalado para R$/botijao P13.

    Simplificacao consciente: o GLP brasileiro e mistura propano/butano, e o
    PPI real embute frete e tancagem. Serve para medir co-movimento, nao para
    reproduzir o PPI centavo a centavo.
    """
    p = propano.copy()
    p["mes"] = p["data"].dt.to_period("M").dt.to_timestamp()
    p = p.groupby("mes", as_index=False)["propano_usd_gal"].mean()

    c = cambio.copy()
    c["mes"] = c["data"].dt.to_period("M").dt.to_timestamp()
    c = c.groupby("mes", as_index=False)["ptax"].mean()

    d = p.merge(c, on="mes", how="inner")
    d["custo_intl_brl_p13"] = (
        d["propano_usd_gal"] / KG_POR_GALAO_PROPANO * d["ptax"] * KG_P13
    )
    return d[["mes", "custo_intl_brl_p13"]]


CAMADAS = ["preco_produtor", "tributos", "margem_distribuicao", "margem_revenda"]


def preparar_decomposicao(decomp: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Fecha gaps pontuais nas camadas do stack e devolve quais meses foram
    interpolados, para a interface declarar.

    Sem isso, um unico componente faltando (set/2020: margem de revenda) faz o
    grafico empilhado desabar ~R$18 naquele mes — uma queda que NAO existiu.
    Um buraco desenhado como despencada e pior que uma interpolacao declarada.
    """
    d = decomp.copy().sort_values("mes")
    faltando = d[d[CAMADAS].isna().any(axis=1)]["mes"].tolist()

    d = d.set_index("mes")
    for c in CAMADAS:
        d[c] = d[c].interpolate(method="time", limit=GAP_MAX_MESES, limit_area="inside")
    d = d.reset_index()

    # recalcula o total para ficar coerente com as camadas interpoladas
    d["preco_consumidor"] = d["preco_consumidor"].fillna(d[CAMADAS].sum(axis=1))

    interpolados = [m for m in faltando if not d.loc[d["mes"] == m, CAMADAS].isna().any(axis=1).iloc[0]]
    return d, interpolados


def painel_mensal(custo: pd.DataFrame, decomp: pd.DataFrame) -> pd.DataFrame:
    """Junta o custo internacional (proxy) com produtor e consumidor (reais)."""
    base = decomp[["mes", "preco_produtor", "preco_consumidor"]].copy()
    d = base.merge(custo, on="mes", how="outer").sort_values("mes")

    grade = pd.DataFrame({"mes": pd.date_range(d["mes"].min(), d["mes"].max(), freq="MS")})
    d = grade.merge(d, on="mes", how="left")

    # marca ANTES de interpolar
    for col in ["custo_intl_brl_p13", "preco_produtor", "preco_consumidor"]:
        d[f"{col}_interpolado"] = d[col].isna()

    d = d.set_index("mes")
    for col in ["custo_intl_brl_p13", "preco_produtor", "preco_consumidor"]:
        d[col] = d[col].interpolate(method="time", limit=GAP_MAX_MESES, limit_area="inside")
    d = d.reset_index()

    # se sobrou NA (gap maior que o limite), a marca vira False de novo:
    # nao foi interpolado, ficou faltando mesmo
    for col in ["custo_intl_brl_p13", "preco_produtor", "preco_consumidor"]:
        d.loc[d[col].isna(), f"{col}_interpolado"] = False
    return d


def indexar(d: pd.DataFrame, colunas: list[str], base_100=True) -> pd.DataFrame:
    """Indexa colunas a 100 na primeira observacao valida da janela."""
    out = d.copy()
    for c in colunas:
        serie = out[c].dropna()
        if len(serie) == 0:
            continue
        base = serie.iloc[0]
        out[f"idx_{c}"] = 100 * out[c] / base if base_100 else out[c]
    return out


def hhi_cr4(d_mes: pd.DataFrame, coluna="p13_kg"):
    """HHI (0-10000) e CR4 (%) a partir de volumes por distribuidora."""
    v = d_mes[d_mes[coluna].notna() & (d_mes[coluna] > 0)].copy()
    if v.empty:
        return None, None, v
    v["share"] = v[coluna] / v[coluna].sum()
    v = v.sort_values("share", ascending=False).reset_index(drop=True)
    hhi = float(((v["share"] * 100) ** 2).sum())
    cr4 = float(v["share"].head(4).sum() * 100)
    return hhi, cr4, v


def classificar_hhi(hhi: float) -> tuple[str, str]:
    """Faixas do guia de concentracao horizontal (DOJ/FTC, adotadas pelo CADE)."""
    if hhi >= 2500:
        return "altamente concentrado", "warn"
    if hhi >= 1500:
        return "moderadamente concentrado", "accent"
    return "desconcentrado", ""
