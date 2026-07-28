"""
Elasticidade-preco da demanda agregada de GLP P13 (revenda).

    log(Volume_total) = a + eps * log(Preco_real) + sazonalidade + tendencia + u

Decisoes que mudam o resultado, todas deliberadas:

1. PRECO REAL, nao nominal. Entre 2007 e 2023 o preco nominal do botijao mais
   que triplicou por inflacao geral. Regredir volume contra preco nominal mede
   inflacao, nao elasticidade. Deflacionado pelo IPCA (BCB/SGS 433).

2. VOLUME AGREGADO DE TODOS OS DISTRIBUIDORES, nao so da Nacional Gas. A
   pergunta e sobre demanda de mercado; a participacao de cada empresa e outro
   problema (aba 6).

3. ENDOGENEIDADE ASSUMIDA E DECLARADA. Preco e quantidade sao determinados em
   equilibrio: um choque de demanda move os dois. O coeficiente de MQO e
   portanto uma mistura de elasticidade de demanda e de oferta, com vies de
   simultaneidade de sinal ambiguo. NAO tentamos corrigir com variavel
   instrumental aqui — o candidato natural (custo internacional) e o mesmo
   choque que ja alimenta o simulador, e um instrumento mal validado e pior
   que nenhum. O numero e ORDEM DE GRANDEZA declarada, nao elasticidade
   estrutural.

4. Teste de estacionariedade ANTES do MQO. Se as series forem I(1), a
   regressao em nivel pode ser espuria; nesse caso a especificacao em
   primeira diferenca e a que vale, e e ela que reportamos como principal.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# IPCA (deflator)
# ---------------------------------------------------------------------------
def ipca(force=False) -> pd.DataFrame:
    """Indice IPCA mensal (base = 1 no primeiro mes), via BCB/SGS 433."""
    import requests

    cache = CACHE / "ipca.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    url = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
           "?formato=json&dataInicial=01/01/2006&dataFinal=31/12/2030")
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    d["mes"] = pd.to_datetime(d["data"], format="%d/%m/%Y")
    d["var_pct"] = pd.to_numeric(d["valor"], errors="coerce")
    d = d.dropna(subset=["var_pct"]).sort_values("mes")
    d["ipca_indice"] = (1 + d["var_pct"] / 100).cumprod()
    out = d[["mes", "ipca_indice"]].reset_index(drop=True)
    out.to_parquet(cache, index=False)
    return out


# ---------------------------------------------------------------------------
# Painel de demanda
# ---------------------------------------------------------------------------
# A cobertura do reporte da ANP/SIMP so estabiliza em 2012. Antes disso o
# volume agregado cresce +33% ao ano (2008-2010) e +6,2% em 2011 — o mercado
# brasileiro de GLP nao quadruplicou nesse periodo, isso e ampliacao
# progressiva da base de declarantes. Usar 2007-2011 inflaciona artificialmente
# a elasticidade, porque a subida de cobertura fica correlacionada com a
# trajetoria do preco real. De 2012 em diante o crescimento anual cai para
# +0,3% / +1,3% / +2,7%, compativel com um mercado maduro.
INICIO_COBERTURA_ESTAVEL = "2012-01-01"


def painel_demanda(g_vendas: pd.DataFrame, decomp: pd.DataFrame,
                   inicio: str | None = INICIO_COBERTURA_ESTAVEL) -> pd.DataFrame:
    """Volume nacional total (todos os distribuidores) x preco real."""
    vol = (g_vendas.groupby("data", as_index=False)["qtd_mil_ton"].sum()
                   .rename(columns={"data": "mes", "qtd_mil_ton": "volume_mil_ton"}))
    if inicio is not None:
        vol = vol[vol["mes"] >= pd.Timestamp(inicio)]

    pr = decomp[["mes", "preco_consumidor"]].copy()
    d = vol.merge(pr, on="mes", how="inner").merge(ipca(), on="mes", how="left")
    d = d.dropna(subset=["volume_mil_ton", "preco_consumidor", "ipca_indice"])
    d = d[d["volume_mil_ton"] > 0].sort_values("mes").reset_index(drop=True)

    # preco real na moeda do ultimo mes da amostra
    base = d["ipca_indice"].iloc[-1]
    d["preco_real"] = d["preco_consumidor"] * base / d["ipca_indice"]

    d["log_vol"] = np.log(d["volume_mil_ton"])
    d["log_preco"] = np.log(d["preco_real"])
    d["log_preco_nominal"] = np.log(d["preco_consumidor"])
    d["mes_num"] = d["mes"].dt.month
    d["tendencia"] = np.arange(len(d))
    d["sin"] = np.sin(2 * np.pi * d["mes_num"] / 12)
    d["cos"] = np.cos(2 * np.pi * d["mes_num"] / 12)
    return d


# ---------------------------------------------------------------------------
# Diagnostico e estimacao
# ---------------------------------------------------------------------------
def diagnostico_raiz_unitaria(d: pd.DataFrame) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss

    out = {}
    for nome, serie in [("log_volume", d["log_vol"]), ("log_preco_real", d["log_preco"])]:
        x = serie.dropna().to_numpy()
        adf_p = float(adfuller(x, regression="c", autolag="AIC")[1])
        try:
            kpss_p = float(kpss(x, regression="c", nlags="auto")[1])
        except Exception:
            kpss_p = np.nan
        dx = np.diff(x)
        adf_d_p = float(adfuller(dx, regression="c", autolag="AIC")[1])
        out[nome] = dict(
            adf_p=adf_p, kpss_p=kpss_p, adf_diff_p=adf_d_p,
            # I(1) tipico: ADF nao rejeita em nivel, KPSS rejeita, ADF rejeita na diferenca
            i1=(adf_p > 0.05) and (adf_d_p < 0.05),
        )
    return out


def estimar(d: pd.DataFrame, hac_lags=12) -> dict:
    """Estima a elasticidade em nivel e em primeira diferenca.

    Erros-padrao HAC (Newey-West): a serie tem autocorrelacao residual obvia,
    e MQO simples subestimaria o intervalo de confianca.
    """
    import statsmodels.api as sm

    # --- especificacao em NIVEL -------------------------------------------
    X = sm.add_constant(d[["log_preco", "tendencia", "sin", "cos"]])
    m_niv = sm.OLS(d["log_vol"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    # --- especificacao em PRIMEIRA DIFERENCA -------------------------------
    dd = d.copy()
    dd["d_vol"] = dd["log_vol"].diff()
    dd["d_preco"] = dd["log_preco"].diff()
    dd = dd.dropna(subset=["d_vol", "d_preco"])
    Xd = sm.add_constant(dd[["d_preco", "sin", "cos"]])
    m_dif = sm.OLS(dd["d_vol"], Xd).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    def _resumo(m, chave):
        ic = m.conf_int(alpha=0.05)
        lo, hi = float(ic.loc[chave, 0]), float(ic.loc[chave, 1])
        return dict(eps=float(m.params[chave]), se=float(m.bse[chave]),
                    t=float(m.tvalues[chave]), p=float(m.pvalues[chave]),
                    ic=(lo, hi), r2=float(m.rsquared), n=int(m.nobs))

    return dict(
        nivel=_resumo(m_niv, "log_preco"),
        diferenca=_resumo(m_dif, "d_preco"),
        modelo_nivel=m_niv, modelo_diferenca=m_dif,
        dados=d, dados_dif=dd,
    )


def avaliar_plausibilidade(eps: float) -> tuple[str, str]:
    """Confere o numero contra o que a teoria e a literatura esperam.

    GLP de botijao e bem de necessidade sem substituto facil para domicilio
    sem gas encanado: espera-se demanda INELASTICA (|eps| < 1) e sinal
    NEGATIVO. Fora disso, o numero nao deve ser usado sem investigacao.
    """
    if eps > 0:
        return ("implausivel", "Sinal POSITIVO: preço sobe e quantidade sobe. "
                "Sintoma clássico de viés de simultaneidade — a regressão está "
                "capturando a curva de OFERTA, não a de demanda.")
    if abs(eps) > 1.0:
        return ("suspeito", "Demanda elástica (|ε| > 1) para um bem de necessidade "
                "sem substituto fácil. Fora do esperado pela literatura.")
    if abs(eps) < 0.05:
        return ("fraco", "Elasticidade praticamente nula — possível, mas verificar "
                "se o preço tem variação real suficiente para identificar o efeito.")
    return ("plausivel", "Sinal negativo e demanda inelástica (|ε| < 1), como se "
            "espera de um bem de necessidade sem substituto fácil.")
