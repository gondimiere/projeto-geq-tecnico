"""
Local Projections (Jordà, 2005) para a resposta do preco do produtor a um
choque no custo internacional.

Por que trocar a IRF do VECM/GIRF por LP:

  - A IRF de um VAR/VECM propaga o MESMO modelo h passos a frente. Se a
    dinamica estiver mal especificada em algum ponto, o erro se compoe ao
    longo do horizonte. LP estima uma regressao SEPARADA para cada horizonte,
    entao um erro em h=6 nao contamina h=12.
  - Assimetria em LP e trivial: basta interagir o choque com um indicador de
    estado (Ramey & Zubairy, 2018). Nao precisa de modelo de limiar, nem de
    simulacao Monte Carlo — e sai com inferencia (erro-padrao) de verdade,
    nao banda condicional ao modelo estimado.
  - O preco: LP e menos eficiente (erros-padrao maiores) e as bandas ficam
    mais largas. Isso e honestidade, nao defeito — a incerteza sempre esteve
    la; o VAR e que a escondia.

Especificacao, para cada horizonte h = 0..H:

    lp_{t+h} - lp_{t-1} = a_h + b_h * s_t + controles_{t-1} + u_{t+h}

O coeficiente b_h E a resposta a impulso no horizonte h. Como as janelas se
sobrepoem, o residuo e serialmente correlacionado por construcao: usa-se
erro-padrao HAC (Newey-West) com defasagem h+1, que e a pratica padrao.

Controles: defasagens de Δlp e Δlc e o termo de correcao de erro (ECT), que
e o que preserva a relacao de cointegracao ja confirmada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _base(d: pd.DataFrame, n_lags=2):
    """Monta as series de trabalho: log-niveis, diferencas e o ECT."""
    lc, lp = d["lc"].to_numpy(), d["lp"].to_numpy()

    # relacao de longo prazo (a mesma do Engle-Granger)
    X = np.column_stack([np.ones_like(lc), lc])
    coef, *_ = np.linalg.lstsq(X, lp, rcond=None)
    ect = lp - (coef[0] + coef[1] * lc)

    dlc = np.r_[np.nan, np.diff(lc)]
    dlp = np.r_[np.nan, np.diff(lp)]
    return dict(lc=lc, lp=lp, ect=ect, dlc=dlc, dlp=dlp, beta=float(coef[1]), n_lags=n_lags)


def _controles(B, t, n_lags):
    """Vetor de controles conhecidos em t-1."""
    out = [B["ect"][t - 1]]
    for l in range(1, n_lags + 1):
        out.append(B["dlp"][t - l])
        out.append(B["dlc"][t - l])
    return out


def projecao_local(d: pd.DataFrame, horizonte=24, n_lags=2, choque_pct=30.0):
    """LP linear. Devolve a resposta (%) a um choque de `choque_pct` no custo."""
    B = _base(d, n_lags)
    n = len(d)
    escala = np.log1p(choque_pct / 100.0)

    hs, betas, se = [], [], []
    for h in range(horizonte + 1):
        y, Xr = [], []
        for t in range(n_lags + 1, n - h):
            ctrl = _controles(B, t, n_lags)
            if not np.all(np.isfinite(ctrl)) or not np.isfinite(B["dlc"][t]):
                continue
            y.append(B["lp"][t + h] - B["lp"][t - 1])
            Xr.append([B["dlc"][t]] + ctrl)
        if len(y) < 20:
            break
        y = np.asarray(y)
        X = sm.add_constant(np.asarray(Xr), has_constant="add")
        # HAC (Newey-West): janelas sobrepostas geram autocorrelacao por
        # construcao; defasagem h+1 e a escolha padrao na literatura de LP
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        hs.append(h)
        betas.append(res.params[1])
        se.append(res.bse[1])

    b = np.asarray(betas) * escala
    s = np.asarray(se) * abs(escala)
    return dict(
        h=np.asarray(hs),
        central=(np.exp(b) - 1) * 100,
        p10=(np.exp(b - 1.2816 * s) - 1) * 100,   # faixa de 80%
        p90=(np.exp(b + 1.2816 * s) - 1) * 100,
        beta_lp=B["beta"],
        equilibrio_pct=(np.exp(B["beta"] * escala) - 1) * 100,
    )


def projecao_local_estado(d: pd.DataFrame, horizonte=24, n_lags=2, choque_pct=30.0):
    """LP com dependencia de estado (Ramey & Zubairy, 2018).

    Separa a amostra pelo SINAL do choque de custo no periodo t e estima um
    coeficiente de resposta para cada regime. Testa diretamente a hipotese
    "sobe rapido / desce devagar" (ou o contrario) com inferencia formal, em
    vez de por simulacao.
    """
    B = _base(d, n_lags)
    n = len(d)
    escala = np.log1p(abs(choque_pct) / 100.0)

    hs, b_alta, s_alta, b_baixa, s_baixa, p_dif = [], [], [], [], [], []
    for h in range(horizonte + 1):
        y, Xr = [], []
        for t in range(n_lags + 1, n - h):
            ctrl = _controles(B, t, n_lags)
            if not np.all(np.isfinite(ctrl)) or not np.isfinite(B["dlc"][t]):
                continue
            s_t = B["dlc"][t]
            ind = 1.0 if s_t > 0 else 0.0      # estado: custo subindo ou caindo
            y.append(B["lp"][t + h] - B["lp"][t - 1])
            Xr.append([ind * s_t, (1 - ind) * s_t, ind] + ctrl)
        if len(y) < 25:
            break
        y = np.asarray(y)
        X = sm.add_constant(np.asarray(Xr), has_constant="add")
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})

        hs.append(h)
        b_alta.append(res.params[1]); s_alta.append(res.bse[1])
        b_baixa.append(res.params[2]); s_baixa.append(res.bse[2])
        # teste formal de simetria: b_alta == b_baixa
        R = np.zeros((1, X.shape[1])); R[0, 1] = 1; R[0, 2] = -1
        p_dif.append(float(res.f_test(R).pvalue))

    def _pct(b, s, escala_regime):
        """Converte o coeficiente na resposta % a um choque do TAMANHO e do
        SINAL coerentes com o regime. O regime de queda tem de ser avaliado
        num choque negativo — avalia-lo num choque positivo nao significa nada.
        """
        b = np.asarray(b) * escala_regime
        s = np.asarray(s) * abs(escala_regime)
        return ((np.exp(b) - 1) * 100,
                (np.exp(b - 1.2816 * s) - 1) * 100,
                (np.exp(b + 1.2816 * s) - 1) * 100)

    esc_alta = np.log1p(abs(choque_pct) / 100.0)      # ex.: +30%
    esc_baixa = np.log1p(-abs(choque_pct) / 100.0)    # ex.: -30%
    ca, la, ua = _pct(b_alta, s_alta, esc_alta)
    cb, lb, ub = _pct(b_baixa, s_baixa, esc_baixa)
    return dict(
        h=np.asarray(hs),
        alta=dict(central=ca, p10=la, p90=ua),
        baixa=dict(central=cb, p10=lb, p90=ub),
        p_simetria=np.asarray(p_dif),
        equilibrio_pct=(np.exp(B["beta"] * escala) - 1) * 100,
    )
