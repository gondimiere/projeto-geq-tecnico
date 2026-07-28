"""
MIDAS (Mixed Data Sampling — Ghysels, Santa-Clara & Valkanov) para usar o
cambio DIARIO e o propano SEMANAL sem reduzi-los a media mensal.

O problema que resolve: hoje o painel calcula a media do mes e joga fora a
trajetoria dentro do mes. Um dolar que sobe 5% no dia 28 e um que sobe 5% no
dia 2 entram identicos na media — mas nao carregam a mesma informacao sobre
o mes seguinte. MIDAS mantem cada observacao de alta frequencia e estima
QUANTO cada defasagem pesa, com poucos parametros.

Esquema de pesos (Beta normalizada, o padrao da literatura):

    w(k; t1, t2) = f(k/K; t1, t2) / soma,   f(x) = x^(t1-1) * (1-x)^(t2-1)

Com K defasagens de alta frequencia, o custo em parametros e 2 (t1, t2) em
vez de K. Com t1=1 os pesos decaem monotonicamente — a forma que se espera
para dado financeiro (o mais recente pesa mais).

Modelo estimado (correcao de erro + MIDAS):

    dlp_t = c + a*ECT_{t-1} + b_fx * X_fx(t; theta_fx)
                            + b_pr * X_pr(t; theta_pr)
                            + phi * dlp_{t-1} + e_t

    onde X(t; theta) = soma_k w(k; theta) * (variacao de alta frequencia)

Referencia (benchmark) para comparacao honesta: o MESMO modelo com media
mensal simples no lugar do termo MIDAS. Se o MIDAS nao ganhar, isso e
reportado — a comparacao existe justamente para poder perder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Pesos
# ---------------------------------------------------------------------------
def pesos_beta(K: int, t1: float, t2: float) -> np.ndarray:
    """Pesos Beta normalizados. Indice 0 = defasagem mais RECENTE."""
    x = (np.arange(K) + 1.0) / (K + 1.0)
    t1 = max(t1, 1e-4)
    t2 = max(t2, 1e-4)
    with np.errstate(over="ignore", invalid="ignore"):
        f = x ** (t1 - 1.0) * (1.0 - x) ** (t2 - 1.0)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    s = f.sum()
    return f / s if s > 0 else np.full(K, 1.0 / K)


# ---------------------------------------------------------------------------
# Montagem das matrizes de alta frequencia
# ---------------------------------------------------------------------------
def matriz_hf(serie: pd.DataFrame, col_data: str, col_valor: str,
              meses: pd.Series, K: int, em_variacao=True) -> np.ndarray:
    """Para cada mes, pega as K observacoes de alta frequencia mais recentes
    ate o fim daquele mes. Devolve matriz (n_meses, K), coluna 0 = mais recente.

    Sem olhar para o futuro: so entram observacoes com data <= fim do mes t.
    """
    s = serie[[col_data, col_valor]].dropna().sort_values(col_data)
    datas = s[col_data].to_numpy()
    vals = s[col_valor].to_numpy(dtype=float)
    if em_variacao:
        vals = np.r_[np.nan, np.diff(np.log(vals))]

    out = np.full((len(meses), K), np.nan)
    for i, m in enumerate(meses):
        fim = (pd.Timestamp(m) + pd.offsets.MonthEnd(0)).to_datetime64()
        idx = np.searchsorted(datas, fim, side="right")
        ini = max(0, idx - K)
        janela = vals[ini:idx][::-1]          # inverte: [0] = mais recente
        if len(janela) == K:
            out[i] = janela
    return out


# ---------------------------------------------------------------------------
# Estimacao
# ---------------------------------------------------------------------------
def _monta_xy(painel, X_fx, X_pr, ect, n_pula=1):
    dlp = np.r_[np.nan, np.diff(np.log(painel["preco_produtor"].to_numpy(dtype=float)))]
    y, linhas = [], []
    for t in range(n_pula + 1, len(painel)):
        if not (np.isfinite(dlp[t]) and np.isfinite(dlp[t - 1]) and np.isfinite(ect[t - 1])):
            continue
        if not (np.all(np.isfinite(X_fx[t])) and np.all(np.isfinite(X_pr[t]))):
            continue
        y.append(dlp[t])
        linhas.append(t)
    return np.asarray(y), np.asarray(linhas), dlp


def estimar_midas(painel: pd.DataFrame, cambio: pd.DataFrame, propano: pd.DataFrame,
                  K_fx=44, K_pr=10, inicio="2017-08-01", theta1_livre=False):
    """Estima o EC-MIDAS e o benchmark de media mensal na mesma amostra.

    K_fx=44 dias uteis (~2 meses de cambio), K_pr=10 semanas (~2,5 meses).
    """
    d = painel[painel["mes"] >= pd.Timestamp(inicio)].copy()
    d = d.dropna(subset=["preco_produtor", "custo_intl_brl_p13"]).reset_index(drop=True)

    lc = np.log(d["custo_intl_brl_p13"].to_numpy(dtype=float))
    lp = np.log(d["preco_produtor"].to_numpy(dtype=float))
    A = np.column_stack([np.ones_like(lc), lc])
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    ect = lp - (coef[0] + coef[1] * lc)

    X_fx = matriz_hf(cambio, "data", "ptax", d["mes"], K_fx)
    X_pr = matriz_hf(propano, "data", "propano_usd_gal", d["mes"], K_pr)

    y, linhas, dlp = _monta_xy(d, X_fx, X_pr, ect)
    if len(y) < 30:
        raise ValueError(f"amostra insuficiente para MIDAS: {len(y)} observacoes")

    Xfx = X_fx[linhas]
    Xpr = X_pr[linhas]
    ect_l = ect[linhas - 1]
    dlp_l = dlp[linhas - 1]

    def _residuo(theta):
        t1f, t2f, t1p, t2p = theta
        zf = Xfx @ pesos_beta(Xfx.shape[1], t1f, t2f)
        zp = Xpr @ pesos_beta(Xpr.shape[1], t1p, t2p)
        Z = np.column_stack([np.ones_like(y), ect_l, dlp_l, zf, zp])
        b, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return y - Z @ b, Z, b

    def _ssr(theta):
        r, _, _ = _residuo(theta)
        return float((r ** 2).sum())

    # theta1 fixo em 1 => pesos monotonicamente decrescentes (o padrao).
    # theta1 livre => permite pesos em corcova, mais flexivel e mais sujeito
    # a sobreajuste: por isso a comparacao fora da amostra roda nas duas.
    b1 = (0.5, 20.0) if theta1_livre else (1.0, 1.0)
    melhor, melhor_ssr = None, np.inf
    for t1 in ((1.0, 3.0) if theta1_livre else (1.0,)):
        for t2f in (1.5, 3.0, 6.0):
            for t2p in (1.5, 3.0, 6.0):
                r = minimize(_ssr, x0=[t1, t2f, t1, t2p],
                             bounds=[b1, (1.01, 30.0), b1, (1.01, 30.0)],
                             method="L-BFGS-B")
                if r.fun < melhor_ssr:
                    melhor_ssr, melhor = r.fun, r.x

    resid, Z, b = _residuo(melhor)
    n, k = len(y), Z.shape[1] + 2          # +2 = os dois theta estimados
    ssr = float((resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())

    # --- benchmark: media mensal simples (o que o painel usava) -------------
    zf_m = np.nanmean(Xfx, axis=1)
    zp_m = np.nanmean(Xpr, axis=1)
    Zb = np.column_stack([np.ones_like(y), ect_l, dlp_l, zf_m, zp_m])
    bb, *_ = np.linalg.lstsq(Zb, y, rcond=None)
    resid_b = y - Zb @ bb
    ssr_b = float((resid_b ** 2).sum())
    kb = Zb.shape[1]

    def _r2adj(ssr_, k_):
        return 1 - (ssr_ / (n - k_)) / (sst / (n - 1))

    return dict(
        n=n,
        theta=dict(fx=(melhor[0], melhor[1]), propano=(melhor[2], melhor[3])),
        pesos_fx=pesos_beta(Xfx.shape[1], melhor[0], melhor[1]),
        pesos_pr=pesos_beta(Xpr.shape[1], melhor[2], melhor[3]),
        coef=dict(const=b[0], ect=b[1], dlp_lag=b[2], fx=b[3], propano=b[4]),
        midas=dict(ssr=ssr, r2adj=_r2adj(ssr, k), rmse=float(np.sqrt(ssr / n)),
                   aic=n * np.log(ssr / n) + 2 * k),
        benchmark=dict(ssr=ssr_b, r2adj=_r2adj(ssr_b, kb), rmse=float(np.sqrt(ssr_b / n)),
                       aic=n * np.log(ssr_b / n) + 2 * kb),
        K_fx=Xfx.shape[1], K_pr=Xpr.shape[1],
        ganho_rmse_pct=100 * (1 - np.sqrt(ssr / n) / np.sqrt(ssr_b / n)),
    )


def validacao_fora_amostra(painel, cambio, propano, K_fx=44, K_pr=10,
                            inicio="2017-08-01", n_treino=60, theta1_livre=False):
    """Comparacao honesta fora da amostra: janela expansiva, 1 passo a frente.

    Re-estima os dois modelos a cada mes e compara o erro de previsao. E o
    unico teste que importa para a pergunta "usar alta frequencia ajuda?".
    """
    d = painel[painel["mes"] >= pd.Timestamp(inicio)].copy()
    d = d.dropna(subset=["preco_produtor", "custo_intl_brl_p13"]).reset_index(drop=True)

    erros_m, erros_b = [], []
    for fim in range(n_treino, len(d)):
        treino = d.iloc[:fim]
        try:
            r = estimar_midas(treino, cambio, propano, K_fx, K_pr, inicio=inicio,
                              theta1_livre=theta1_livre)
        except Exception:
            continue

        lc = np.log(d["custo_intl_brl_p13"].to_numpy(dtype=float))
        lp = np.log(d["preco_produtor"].to_numpy(dtype=float))
        A = np.column_stack([np.ones_like(lc[:fim]), lc[:fim]])
        coef, *_ = np.linalg.lstsq(A, lp[:fim], rcond=None)
        ect = lp - (coef[0] + coef[1] * lc)
        dlp = np.r_[np.nan, np.diff(lp)]

        Xfx1 = matriz_hf(cambio, "data", "ptax", d["mes"].iloc[[fim]], K_fx)[0]
        Xpr1 = matriz_hf(propano, "data", "propano_usd_gal", d["mes"].iloc[[fim]], K_pr)[0]
        if not (np.all(np.isfinite(Xfx1)) and np.all(np.isfinite(Xpr1))):
            continue

        c = r["coef"]
        prev_m = (c["const"] + c["ect"] * ect[fim - 1] + c["dlp_lag"] * dlp[fim - 1]
                  + c["fx"] * float(Xfx1 @ r["pesos_fx"])
                  + c["propano"] * float(Xpr1 @ r["pesos_pr"]))
        prev_b = (c["const"] + c["ect"] * ect[fim - 1] + c["dlp_lag"] * dlp[fim - 1]
                  + c["fx"] * float(np.nanmean(Xfx1))
                  + c["propano"] * float(np.nanmean(Xpr1)))
        real = dlp[fim]
        if np.isfinite(real):
            erros_m.append(real - prev_m)
            erros_b.append(real - prev_b)

    em, eb = np.asarray(erros_m), np.asarray(erros_b)
    if len(em) < 5:
        return None
    rmse_m, rmse_b = float(np.sqrt((em ** 2).mean())), float(np.sqrt((eb ** 2).mean()))
    return dict(n=len(em), rmse_midas=rmse_m, rmse_benchmark=rmse_b,
                ganho_pct=100 * (1 - rmse_m / rmse_b))
