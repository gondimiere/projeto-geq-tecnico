"""
Funcoes de resposta a impulso (IRF) para o elo custo internacional -> produtor.

DUAS versoes, porque o modelo estimado e assimetrico e a IRF linear padrao
NAO representa isso:

1) IRF linear (VECM ortogonalizado, Cholesky) — o padrao dos livros-texto.
   Serve de referencia e de sanity check, mas por construcao devolve a MESMA
   resposta para choque de alta e de baixa, so trocando o sinal. Como os
   testes rejeitaram simetria (M-TAR, p=0,003), ela subestima a lentidao do
   ajuste depois de um choque de alta.

2) GIRF (Generalized IRF, Koop, Pesaran & Potter 1996) sobre o modelo de
   correcao de erro com limiar. E o instrumento correto para modelo nao
   linear: a resposta passa a depender do TAMANHO e do SINAL do choque e do
   estado em que o sistema estava. E obtida por simulacao Monte Carlo:
       GIRF(h, d, w) = E[y_{t+h} | e_t = d, w_{t-1}] - E[y_{t+h} | w_{t-1}]
   com as duas trajetorias partindo das MESMAS historias e dos MESMOS
   choques futuros reamostrados — a diferenca isola o efeito do choque.

Identificacao: o custo internacional e ordenado primeiro (Cholesky) e tratado
como fracamente exogeno. Isso nao e conveniencia: o Brasil e tomador de preco
no mercado internacional de GLP, e o proprio VECM estimado no R mostrou
alpha ~ 0 na equacao do custo. Um choque em Mont Belvieu afeta o preco
domestico no mesmo mes; o contrario nao vale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260727)


# ---------------------------------------------------------------------------
# Estimacao
# ---------------------------------------------------------------------------
def preparar(painel: pd.DataFrame, inicio="2017-08-01") -> pd.DataFrame:
    """Recorta o regime confirmado por Gregory-Hansen e devolve as log-series."""
    d = painel[painel["mes"] >= pd.Timestamp(inicio)].copy()
    d = d.dropna(subset=["custo_intl_brl_p13", "preco_produtor"])
    d["lc"] = np.log(d["custo_intl_brl_p13"].astype(float))
    d["lp"] = np.log(d["preco_produtor"].astype(float))
    return d.reset_index(drop=True)


def estimar_mtar(d: pd.DataFrame, n_lags=1, grid=np.arange(0.15, 0.86, 0.05)):
    """Engle-Granger + correcao de erro com limiar (momentum-TAR).

    Devolve os parametros da equacao do produtor, que e o que a simulacao
    precisa. Reproduz o script 18 em R — os numeros sao comparados na
    interface como checagem cruzada entre as duas implementacoes.
    """
    lc, lp = d["lc"].to_numpy(), d["lp"].to_numpy()

    # passo 1: relacao de longo prazo
    X = np.column_stack([np.ones_like(lc), lc])
    coef, *_ = np.linalg.lstsq(X, lp, rcond=None)
    a, beta = coef
    e = lp - (a + beta * lc)

    # passo 2: limiar por grid-search minimizando SSR (Chan, 1993)
    de = np.diff(e)
    cand = np.quantile(de[:-1], grid)
    melhor = None
    for tau in cand:
        Z, y = _montar_mtar(e, n_lags, tau)
        if Z is None:
            continue
        b, *_ = np.linalg.lstsq(Z, y, rcond=None)
        ssr = float(((y - Z @ b) ** 2).sum())
        if melhor is None or ssr < melhor["ssr"]:
            melhor = dict(ssr=ssr, tau=float(tau), beta_mtar=b, Z=Z, y=y)

    b = melhor["beta_mtar"]
    resid = melhor["y"] - melhor["Z"] @ b
    gl = len(melhor["y"]) - Z.shape[1]
    return dict(
        a=float(a), beta=float(beta), tau=melhor["tau"],
        rho1=float(b[0]), rho2=float(b[1]),
        gamas=b[2:].astype(float),
        sigma=float(np.sqrt((resid ** 2).sum() / gl)),
        resid=resid, e=e, n_lags=n_lags,
    )


def _montar_mtar(e, n_lags, tau):
    """Monta a regressao De_t = rho1*M*e_{t-1} + rho2*(1-M)*e_{t-1} + gamas*De_{t-l}."""
    ini = n_lags + 2
    if ini >= len(e):
        return None, None
    idx = np.arange(ini, len(e))
    y = e[idx] - e[idx - 1]
    elag = e[idx - 1]
    de_lag = e[idx - 1] - e[idx - 2]
    M = (de_lag >= tau).astype(float)
    cols = [M * elag, (1 - M) * elag]
    for l in range(1, n_lags + 1):
        cols.append(e[idx - l] - e[idx - l - 1])
    return np.column_stack(cols), y


def meia_vida(rho: float) -> float:
    """Meses para metade do desvio ser corrigido, dado o coeficiente rho."""
    if rho >= 0 or rho <= -2:
        return float("inf")
    return float(np.log(0.5) / np.log(1 + rho))


# ---------------------------------------------------------------------------
# GIRF por simulacao
# ---------------------------------------------------------------------------
def girf(par: dict, d: pd.DataFrame, choque_pct: float, horizonte=24,
         n_sim=600, n_hist=60):
    """Resposta generalizada do PRECO DO PRODUTOR a um choque no custo.

    O choque entra como um deslocamento permanente de `choque_pct` no custo
    internacional (o custo se comporta como passeio aleatorio: um choque nao
    se reverte sozinho). Isso desloca o equilibrio de longo prazo em
    beta * log(1 + choque), e a velocidade com que o produtor caminha ate la
    depende do regime do limiar — que e exatamente o ponto do modelo.

    Devolve mediana e banda de 80% da trajetoria, em % sobre o preco do
    produtor.
    """
    e = par["e"]
    n_lags = par["n_lags"]
    resid = par["resid"]
    rho1, rho2, tau = par["rho1"], par["rho2"], par["tau"]
    gamas = par["gamas"]

    d_log_choque = np.log1p(choque_pct / 100.0)
    desloc = par["beta"] * d_log_choque   # deslocamento do equilibrio

    # historias iniciais: janelas reais observadas (nao inventadas)
    inicio_min = n_lags + 2
    historias = np.arange(inicio_min, len(e))
    if len(historias) > n_hist:
        historias = RNG.choice(historias, n_hist, replace=False)

    trajs = np.zeros((n_sim * len(historias), horizonte + 1))
    k = 0
    for h0 in historias:
        hist = e[h0 - n_lags - 2: h0].copy()
        for _ in range(n_sim):
            choques = RNG.choice(resid, horizonte + 1, replace=True)
            # baseline e cenario com choque compartilham as MESMAS historias
            # e os MESMOS choques futuros: a diferenca isola o efeito
            base = _simular(hist, 0.0, choques, horizonte, rho1, rho2, tau, gamas, n_lags)
            alt = _simular(hist, desloc, choques, horizonte, rho1, rho2, tau, gamas, n_lags)
            # lp_t = a + beta*lc_t + e_t. Com o choque, o equilibrio sobe
            # `desloc` e o desvio parte de e_0 - desloc. Logo:
            #   resposta_h = desloc + (e_h^choque - e_h^base)
            # Em h=0 isso da 0 (preco ainda nao reagiu) e em h->inf tende a
            # `desloc` (repasse integral de beta) — as duas ancoras corretas.
            trajs[k] = desloc + (alt - base)
            k += 1

    trajs = trajs[:k]
    resp_pct = (np.exp(trajs) - 1) * 100
    return dict(
        h=np.arange(horizonte + 1),
        mediana=np.median(resp_pct, axis=0),
        p10=np.percentile(resp_pct, 10, axis=0),
        p90=np.percentile(resp_pct, 90, axis=0),
        equilibrio_pct=(np.exp(desloc) - 1) * 100,
    )


def _simular(hist, desloc, choques, horizonte, rho1, rho2, tau, gamas, n_lags):
    """Propaga o desvio do equilibrio (e_t) e devolve a trajetoria em NIVEL.

    `desloc` > 0 significa que o equilibrio subiu: o preco ainda nao se moveu,
    entao o desvio no instante 0 passa a ser e_0 - desloc (produtor abaixo do
    novo equilibrio) e o termo de correcao puxa de volta a partir dali.
    """
    e = list(hist)
    e[-1] = e[-1] - desloc          # desloc = 0 no cenario base

    saida = np.empty(horizonte + 1)
    saida[0] = e[-1]
    for t in range(1, horizonte + 1):
        e_lag = e[-1]
        de_lag = e[-1] - e[-2]
        rho = rho1 if de_lag >= tau else rho2      # regime pelo limiar
        de = rho * e_lag + choques[t]
        for l in range(1, n_lags + 1):
            de += gamas[l - 1] * (e[-l] - e[-l - 1])
        e.append(e_lag + de)
        saida[t] = e[-1]
    return saida


# ---------------------------------------------------------------------------
# IRF linear (VECM) — referencia
# ---------------------------------------------------------------------------
def irf_linear(d: pd.DataFrame, choque_pct: float, horizonte=24, n_boot=300):
    """IRF ortogonalizada de um VECM(rank=1) com bandas por bootstrap.

    Ordem de Cholesky: custo internacional primeiro (fracamente exogeno).
    """
    from statsmodels.tsa.vector_ar.vecm import VECM

    y = d[["lc", "lp"]].to_numpy()
    mod = VECM(y, k_ar_diff=1, coint_rank=1, deterministic="ci")
    res = mod.fit()
    irfs = res.irf(horizonte).orth_irfs        # (h+1, neqs, neqs)
    resp = irfs[:, 1, 0]                        # resposta de lp a choque em lc

    # escala: choque de 1 desvio-padrao -> choque desejado
    sd_lc = float(np.sqrt(res.sigma_u[0, 0]))
    escala = np.log1p(choque_pct / 100.0) / sd_lc
    central = (np.exp(resp * escala) - 1) * 100

    # bootstrap residual para a banda
    boot = np.zeros((n_boot, horizonte + 1))
    resid = res.resid
    fit_vals = y[-len(resid):] - resid
    for b in range(n_boot):
        idx = RNG.integers(0, len(resid), len(resid))
        y_b = fit_vals + resid[idx]
        try:
            r_b = VECM(y_b, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
            rb = r_b.irf(horizonte).orth_irfs[:, 1, 0]
            sd_b = float(np.sqrt(r_b.sigma_u[0, 0]))
            boot[b] = (np.exp(rb * np.log1p(choque_pct / 100.0) / sd_b) - 1) * 100
        except Exception:
            boot[b] = np.nan

    boot = boot[~np.isnan(boot).any(axis=1)]
    return dict(
        h=np.arange(horizonte + 1),
        central=central,
        p10=np.nanpercentile(boot, 10, axis=0) if len(boot) else central,
        p90=np.nanpercentile(boot, 90, axis=0) if len(boot) else central,
        beta=float(res.beta[1, 0] / res.beta[0, 0]) if res.beta[0, 0] != 0 else np.nan,
    )
