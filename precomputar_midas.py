"""
Pre-computa a validacao MIDAS e grava em cache/midas_resultado.json.

Roda uma vez (leva alguns minutos: sao ~47 re-estimacoes por especificacao,
em janela expansiva). O resultado e uma constatacao metodologica estatica —
nao precisa ser recalculado a cada sessao do painel.

Uso:  python precomputar_midas.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import ingestao, limpeza, midas

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)


def main():
    print("Carregando series…")
    decomp, _ = ingestao.decomposicao_anp()
    decomp, _ = limpeza.preparar_decomposicao(decomp)
    prop, _ = ingestao.propano()
    camb, _ = ingestao.cambio()
    painel = limpeza.painel_mensal(limpeza.custo_internacional(prop, camb), decomp)

    print("Estimando MIDAS dentro da amostra…")
    dentro = midas.estimar_midas(painel, camb, prop)

    print("Validando fora da amostra (pesos disciplinados)…")
    fora = midas.validacao_fora_amostra(painel, camb, prop, n_treino=60)

    print("Validando fora da amostra (pesos livres)…")
    fora_livre = midas.validacao_fora_amostra(painel, camb, prop, n_treino=60,
                                               theta1_livre=True)

    out = dict(
        n=dentro["n"], K_fx=dentro["K_fx"], K_pr=dentro["K_pr"],
        r2adj_midas=dentro["midas"]["r2adj"],
        r2adj_benchmark=dentro["benchmark"]["r2adj"],
        aic_midas=dentro["midas"]["aic"],
        aic_benchmark=dentro["benchmark"]["aic"],
        ganho_dentro=dentro["ganho_rmse_pct"],
        peso_fx_max=float(dentro["pesos_fx"].max()),
        peso_fx_min=float(dentro["pesos_fx"].min()),
        theta=[[float(x) for x in v] for v in dentro["theta"].values()],
        oos_n=fora["n"],
        oos_ganho=fora["ganho_pct"],
        oos_ganho_livre=fora_livre["ganho_pct"],
    )
    (CACHE / "midas_resultado.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nGravado em cache/midas_resultado.json:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
