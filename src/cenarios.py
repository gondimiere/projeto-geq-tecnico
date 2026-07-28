"""
Simulador de cenarios: oferta -> demanda -> participacao -> receita.

Encadeamento, por horizonte h:

  (a) OFERTA   Choque no custo internacional -> preco do produtor, pela GIRF
      assimetrica ja estimada e validada em irf.py. Nao reestima nada aqui.

  (b) REPASSE  Fracao do movimento do produtor que chega ao consumidor. E
      variavel de ESCOLHA, nao estimada: as abas 4 e 5 mostraram que o elo
      produtor->consumidor nao segue equilibrio competitivo (nao cointegra em
      janela alguma). Modelar isso como parametro livre e mais honesto que
      fingir que existe uma regra estimavel.

      O repasse e aplicado sobre a variacao ABSOLUTA em R$, nao percentual:
      tributos e margens sao valores por botijao (R$/P13), nao percentuais do
      preco do produtor. Um produtor +R$3,64 com repasse de 50% vira
      +R$1,82 no preco final.

  (c) DEMANDA  Elasticidade-preco estimada em elasticidade.py:
      V_novo = V_base * (P_novo / P_base) ** eps

  (d) PARTICIPACAO  Constante no cenario central (a aba 6 mostrou que
      participacao de mercado e dominada por persistencia), com choque
      opcional para cenarios de M&A ou entrada regulatoria.

  (e) RECEITA  Da Nacional Gas, ao PRECO DE DISTRIBUICAO — nao ao preco final
      ao consumidor. A margem de revenda fica com o revendedor, nao com a
      distribuidora; usar o preco ao consumidor superestimaria a receita em
      ~40%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 20260727


# ---------------------------------------------------------------------------
# Distribuicao da elasticidade
# ---------------------------------------------------------------------------
def distribuicao_elasticidade(especificacoes: list[tuple[float, float]],
                              n: int, rng: np.random.Generator) -> np.ndarray:
    """Amostra a elasticidade de uma MISTURA de especificacoes.

    Usar so o IC de uma especificacao subestima a incerteza: ja sabemos que o
    numero se move com o corte de amostra (-0,187 a -0,297) e que a versao em
    primeira diferenca tem intervalo muito mais largo. A mistura sorteia
    primeiro QUAL especificacao vale e so entao o valor dentro dela — o que
    propaga incerteza de especificacao, nao so de amostragem.

    `especificacoes`: lista de (ponto, erro_padrao).
    """
    idx = rng.integers(0, len(especificacoes), n)
    pontos = np.array([e[0] for e in especificacoes])[idx]
    erros = np.array([e[1] for e in especificacoes])[idx]
    return rng.normal(pontos, erros)


# ---------------------------------------------------------------------------
# Cenario deterministico
# ---------------------------------------------------------------------------
def cenario(base: dict, resposta_produtor_pct: np.ndarray,
            repasse: float, eps: float,
            choque_share_pp: float = 0.0) -> pd.DataFrame:
    """Encadeia oferta -> demanda -> receita para uma trajetoria de produtor.

    base: dict com preco_produtor, tributos, margem_distribuicao,
          margem_revenda, volume_mercado (botijoes/mes), share.
    resposta_produtor_pct: variacao % do preco do produtor por horizonte.
    repasse: fracao [0,1] do movimento em R$ repassada ao preco final.
    eps: elasticidade-preco da demanda.
    """
    h = np.arange(len(resposta_produtor_pct))

    d_produtor_rs = base["preco_produtor"] * resposta_produtor_pct / 100.0
    d_final_rs = d_produtor_rs * repasse

    p_cons_base = (base["preco_produtor"] + base["tributos"]
                   + base["margem_distribuicao"] + base["margem_revenda"])
    p_cons = p_cons_base + d_final_rs

    # o que a distribuidora recebe: preco final menos a margem do revendedor
    p_dist_base = p_cons_base - base["margem_revenda"]
    p_dist = p_cons - base["margem_revenda"]

    volume = base["volume_mercado"] * (p_cons / p_cons_base) ** eps
    share = np.clip(base["share"] + choque_share_pp / 100.0, 0.0, 1.0)

    receita = p_dist * volume * share

    # MARGEM BRUTA — o numero que decide, nao a receita.
    # O pedaco do aumento do produtor que NAO e repassado (1 - repasse) e
    # absorvido pela cadeia. Aqui atribuimos essa absorcao a distribuidora:
    # a margem por botijao encolhe exatamente nesse valor. Receita pode subir
    # com repasse alto e ainda assim a margem cair — por isso as duas saem.
    margem_unit = base["margem_distribuicao"] - d_produtor_rs * (1.0 - repasse)
    margem_total = margem_unit * volume * share

    receita_base = p_dist_base * base["volume_mercado"] * base["share"]
    margem_base = base["margem_distribuicao"] * base["volume_mercado"] * base["share"]
    return pd.DataFrame({
        "h": h,
        "preco_produtor": base["preco_produtor"] + d_produtor_rs,
        "preco_consumidor": p_cons,
        "preco_distribuicao": p_dist,
        "volume_mercado": volume,
        "volume_nacional": volume * share,
        "share": share,
        "receita": receita,
        "receita_var_pct": 100 * (receita / receita_base - 1),
        "margem_unit": margem_unit,
        "margem_total": margem_total,
        "margem_var_pct": 100 * (margem_total / margem_base - 1),
    })


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo(base: dict, girf_mediana: np.ndarray,
                girf_p10: np.ndarray, girf_p90: np.ndarray,
                especificacoes_eps: list[tuple[float, float]],
                repasse: float, choque_share_pp: float = 0.0,
                sigma_share_pp: float = 1.0, n_sim: int = 5000) -> dict:
    """Propaga a incerteza de cada peca ate a distribuicao de receita.

    Fontes de incerteza propagadas:
      - elasticidade: mistura de especificacoes (ver distribuicao_elasticidade)
      - repasse do produtor: a propria banda da GIRF, tratada como normal
        cuja dispersao vem de (p90 - p10) / 2,563
      - participacao de mercado: normal em torno do valor base, com desvio
        `sigma_share_pp` em pontos percentuais (default 1 pp, da ordem do MAE
        fora da amostra do modelo da aba 6)
    """
    rng = np.random.default_rng(RNG_SEED)
    H = len(girf_mediana)

    eps_draws = distribuicao_elasticidade(especificacoes_eps, n_sim, rng)
    sigma_girf = np.maximum((girf_p90 - girf_p10) / 2.5631, 1e-9)
    z = rng.standard_normal((n_sim, 1))          # um fator por replica: a
    girf_draws = girf_mediana + z * sigma_girf   # trajetoria sobe ou desce junta
    share_draws = np.clip(
        base["share"] + choque_share_pp / 100.0
        + rng.normal(0, sigma_share_pp / 100.0, n_sim), 0.0, 1.0)

    p_cons_base = (base["preco_produtor"] + base["tributos"]
                   + base["margem_distribuicao"] + base["margem_revenda"])
    p_dist_base = p_cons_base - base["margem_revenda"]
    receita_base = p_dist_base * base["volume_mercado"] * base["share"]

    d_prod = base["preco_produtor"] * girf_draws / 100.0
    p_cons = p_cons_base + d_prod * repasse
    p_dist = p_cons - base["margem_revenda"]
    volume = base["volume_mercado"] * (p_cons / p_cons_base) ** eps_draws[:, None]
    receita = p_dist * volume * share_draws[:, None]

    margem_unit = base["margem_distribuicao"] - d_prod * (1.0 - repasse)
    margem = margem_unit * volume * share_draws[:, None]
    margem_base = base["margem_distribuicao"] * base["volume_mercado"] * base["share"]

    pct = np.percentile(receita, [10, 50, 90], axis=0)
    pctm = np.percentile(margem, [10, 50, 90], axis=0)
    return dict(
        h=np.arange(H),
        p10=pct[0], p50=pct[1], p90=pct[2],
        receita_base=receita_base,
        var_p10=100 * (pct[0] / receita_base - 1),
        var_p50=100 * (pct[1] / receita_base - 1),
        var_p90=100 * (pct[2] / receita_base - 1),
        m_p10=pctm[0], m_p50=pctm[1], m_p90=pctm[2],
        margem_base=margem_base,
        mvar_p10=100 * (pctm[0] / margem_base - 1),
        mvar_p50=100 * (pctm[1] / margem_base - 1),
        mvar_p90=100 * (pctm[2] / margem_base - 1),
        eps_draws=eps_draws,
        frac_eps_positiva=float((eps_draws > 0).mean()),
        n_sim=n_sim,
    )


def base_atual(decomp: pd.DataFrame, volume_mercado_botijoes: float,
               share: float) -> dict:
    """Monta o ponto de partida a partir do ultimo mes observado."""
    u = decomp.dropna(subset=["preco_consumidor"]).iloc[-1]
    return dict(
        mes=u["mes"],
        preco_produtor=float(u["preco_produtor"]),
        tributos=float(u["tributos"]),
        margem_distribuicao=float(u["margem_distribuicao"]),
        margem_revenda=float(u["margem_revenda"]),
        preco_consumidor=float(u["preco_consumidor"]),
        volume_mercado=float(volume_mercado_botijoes),
        share=float(share),
    )
