"""
Modelo preditivo da participacao de mercado da Nacional Gas por UF.

Desenho da validacao — o ponto mais importante deste arquivo:

  O corte e TEMPORAL, nunca k-fold aleatorio. Em painel de serie temporal,
  embaralhar linhas coloca o futuro no treino e o passado no teste; o modelo
  aprende a interpolar dentro do periodo e a metrica sai otimista sem
  significar nada. Aqui: treina ate uma data de corte, testa nos meses
  seguintes, sem sobreposicao.

  O modelo tambem e comparado contra um BASELINE INGENUO ("mesma participacao
  do mesmo mes do ano anterior"). Um modelo de ML que nao bate um baseline
  ingenuo nao esta pronto para o slide, e a comparacao existe justamente para
  poder perder.

Cuidado com vazamento: as defasagens (1 e 3 meses) sao construidas por UF
com shift() dentro de cada grupo, e o corte temporal e aplicado DEPOIS.
Nenhuma feature usa informacao do proprio mes-alvo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "share_lag1", "share_lag3", "share_lag12",
    "hhi", "cr4", "n_agentes",
    "mes_sin", "mes_cos",
    "preco_produtor", "ptax",
    "vol_uf_log", "tendencia",
    "programa_social",
]

ALVO = "share"


def montar_features(nac: pd.DataFrame, conc: pd.DataFrame,
                    macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Junta participacao, concentracao e macro; cria defasagens e sazonalidade."""
    d = nac.merge(conc, on=["data", "uf_destino"], how="left")

    d = d.sort_values(["uf_destino", "data"]).reset_index(drop=True)
    for lag in (1, 3, 12):
        d[f"share_lag{lag}"] = d.groupby("uf_destino")["share"].shift(lag)

    # sazonalidade ciclica: dez e jan sao vizinhos, o que 1..12 nao expressa
    d["mes_sin"] = np.sin(2 * np.pi * d["mes"] / 12)
    d["mes_cos"] = np.cos(2 * np.pi * d["mes"] / 12)

    d["vol_uf_log"] = np.log1p(d["botijoes_uf"])
    d["tendencia"] = (d["data"].dt.year - d["data"].dt.year.min()) * 12 + d["data"].dt.month

    # Auxilio Gas dos Brasileiros: instituido em dez/2021 (Lei 14.237/2021),
    # pagamentos a partir de 2022. Antes disso nao havia programa nacional
    # de subsidio ao botijao em vigor continuo.
    d["programa_social"] = (d["data"] >= pd.Timestamp("2022-01-01")).astype(int)

    if macro is not None:
        m = macro.rename(columns={"mes": "data"})[["data", "preco_produtor", "ptax"]]
        d = d.merge(m, on="data", how="left")
    else:
        d["preco_produtor"] = np.nan
        d["ptax"] = np.nan

    return d


def treinar_avaliar(d: pd.DataFrame, data_corte="2022-01-01", n_estimators=400):
    """Treina LightGBM com corte temporal e compara contra baseline ingenuo."""
    import lightgbm as lgb

    d = d.dropna(subset=[ALVO]).copy()
    disp = [f for f in FEATURES if f in d.columns and d[f].notna().any()]

    treino = d[d["data"] < pd.Timestamp(data_corte)].dropna(subset=disp)
    teste = d[d["data"] >= pd.Timestamp(data_corte)].dropna(subset=disp)
    if len(treino) < 200 or len(teste) < 50:
        raise ValueError(f"amostra insuficiente (treino={len(treino)}, teste={len(teste)})")

    X_tr, y_tr = treino[disp], treino[ALVO]
    X_te, y_te = teste[disp], teste[ALVO]

    mod = lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    mod.fit(X_tr, y_tr)
    pred = mod.predict(X_te)

    # baseline ingenuo: a participacao do mesmo mes do ano anterior
    base = teste["share_lag12"]
    ok = base.notna()

    def _mae(a, b):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

    def _rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    mae_m, rmse_m = _mae(y_te, pred), _rmse(y_te, pred)
    mae_b = _mae(y_te[ok], base[ok]) if ok.any() else np.nan
    rmse_b = _rmse(y_te[ok], base[ok]) if ok.any() else np.nan

    return dict(
        modelo=mod, features=disp,
        X_te=X_te, y_te=y_te, pred=pred, teste=teste,
        n_treino=len(treino), n_teste=len(teste),
        data_corte=pd.Timestamp(data_corte),
        mae_modelo=mae_m, rmse_modelo=rmse_m,
        mae_baseline=mae_b, rmse_baseline=rmse_b,
        ganho_mae=100 * (1 - mae_m / mae_b) if mae_b and np.isfinite(mae_b) else np.nan,
    )


def importancia_shap(res: dict, n_amostra=800):
    """Valores SHAP no conjunto de TESTE (nao no treino).

    Explicar o modelo onde ele foi avaliado, e nao onde foi ajustado, evita
    apresentar como 'o que o modelo aprendeu' um padrao que so vale dentro da
    amostra de treino.
    """
    import shap

    X = res["X_te"]
    if len(X) > n_amostra:
        X = X.sample(n_amostra, random_state=42)
    expl = shap.TreeExplainer(res["modelo"])
    valores = expl.shap_values(X)

    imp = (pd.DataFrame({"feature": X.columns,
                         "shap_abs_medio": np.abs(valores).mean(axis=0)})
           .sort_values("shap_abs_medio", ascending=False)
           .reset_index(drop=True))
    return dict(valores=valores, X=X, importancia=imp)


ROTULOS = {
    "share_lag1": "Participação no mês anterior",
    "share_lag3": "Participação 3 meses antes",
    "share_lag12": "Participação 12 meses antes",
    "hhi": "HHI da UF (concentração)",
    "cr4": "CR4 da UF",
    "n_agentes": "Nº de distribuidoras na UF",
    "mes_sin": "Sazonalidade (seno)",
    "mes_cos": "Sazonalidade (cosseno)",
    "preco_produtor": "Preço do produtor (Petrobras)",
    "ptax": "Câmbio BRL/USD",
    "vol_uf_log": "Tamanho do mercado da UF",
    "tendencia": "Tendência temporal",
    "programa_social": "Programa social ativo",
}
