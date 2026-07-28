"""
Resultados econometricos JA COMPUTADOS.

Nao recomputa nada ao vivo: cada bootstrap (1.000-3.000 replicacoes) leva
minutos e travaria a demo. Os numeros abaixo saem dos scripts em R na pasta
pai (09, 13, 15, 16, 17, 18) e dos .rds que eles gravam.

Historico resumido do que foi testado, em ordem:
  1) 2024-2026 semanal, so proxy .............. SEM cointegracao (amostra curta)
  2) 2001-2026 mensal, produtor REAL .......... "confirma" na amostra cheia,
     mas isso mistura dois regimes de precificacao
  3) Gregory-Hansen com quebra endogena ....... quebra REAL em jul/2017
     (p=0,018); regime pre-quebra NAO cointegra sozinho (p=0,42); regime
     pos-quebra cointegra sozinho (p=0,026)
  4) M-TAR so no regime confirmado ............ Phi=12,6, p=0,003
  5) Elo produtor->consumidor ................. nao cointegra em janela alguma
"""

QUEBRA = "jul/2017"

ELO1 = {
    "cheia": {
        "n": 297,
        "janela": "nov/2001 – jun/2026",
        "linhas": [
            ("Engle-Granger (β)", "1,04", "~100% de repasse — mas ver alerta"),
            ("Johansen (traço, r=0)", "25,3 vs 19,96 (5%)", "Rejeita, mas a amostra mistura regimes"),
            ("M-TAR (Φ, bootstrap)", "11,12 vs 6,62 (5%) · p=0,006", "Rejeita, mas ver Gregory-Hansen"),
        ],
    },
    "confirmado": {
        "n": 107,
        "janela": "ago/2017 – jun/2026",
        "linhas": [
            ("Gregory-Hansen (quebra endógena)", f"quebra em {QUEBRA} · p=0,018", "Quebra estrutural real confirmada"),
            ("Cointegração — regime pré-quebra", "n=190 · p=0,42", "NÃO cointegra sozinho"),
            ("Cointegração — regime pós-quebra", "n=107 · p=0,026", "CONFIRMADO"),
            ("M-TAR no regime confirmado (Φ)", "12,61 vs 6,87 (5%) · p=0,003", "Assimetria confirmada"),
            ("Meia-vida — ajuste normal", "2,0 meses", "ρ significativo (t=−5,0)"),
            ("Meia-vida — choque de alta", "~31 meses", "ρ não significativo isolado — ordem de grandeza"),
        ],
    },
}

ELO2 = {
    "linhas": [
        ("Completa (297m)", "Engle-Granger / ADF bootstrap", "β=0,83 · p=0,154", "não rejeita"),
        ("Completa (297m)", "Johansen (traço, r=0)", "12,2 vs 19,96 (5%)", "não rejeita"),
        ("Completa (297m)", "M-TAR (Φ, bootstrap)", "3,20 vs 7,25 (5%) · p=0,351", "não rejeita"),
        ("Pós-2016 (117m)", "Engle-Granger / ADF bootstrap", "β=0,58 · p=0,898", "não rejeita"),
        ("Pós-2016 (117m)", "Johansen (traço, r=0)", "21,7 vs 19,96 (5%)", "rejeita (isolado)"),
        ("Pós-2016 (117m)", "M-TAR (Φ, bootstrap)", "0,07 vs 6,76 (5%) · p=0,982", "não rejeita"),
    ],
    "conclusao": (
        "Nenhuma janela testada confirma cointegração neste elo. O único teste que rejeita "
        "(Johansen pós-2016) não é acompanhado pelo Engle-Granger nem pelo M-TAR na mesma janela — "
        "provável falta de poder estatístico com 117 meses, não evidência de relação. "
        "Isso é consistente com a literatura de concentração: o elo distribuição→revenda não se "
        "comporta como mercado competitivo, então não há por que esperar uma relação de "
        "equilíbrio estável com o preço do produtor."
    ),
}

CONTEXTO = {
    "epe": {
        "fonte": "EPE · Nota Técnica NT-EPE-DPG-SDB-2024-04 (out/2024)",
        "periodo": "2019–2023",
        "igpm": 48,
        "margem_liquida": 188,
    },
    "unila": {
        "fonte": "UNILA · Grupo de Pesquisa em Mobilidade e Matriz Energética (2025)",
        "cr4": 60,
        "margem_glp": 52.1,
        "margem_gasolina": 19.5,
    },
}

EVENTOS = [
    ("nov/2024", "Petrobras inicia leilões de GLP — parte do volume passa a ser precificada por leilão, em vez de tabela.", False),
    ("07/ago/2025", "Conselho de administração da Petrobras aprova a retomada da distribuição de GLP, segmento que a empresa havia deixado em 2020 com a venda da Liquigás.", False),
    ("31/mar/2026", "Leilão de GLP com forte alta, em meio ao conflito Irã–EUA e ao risco de fechamento do Estreito de Ormuz. O Brent quase dobra no período (US$ 70 → US$ 138/bbl).", True),
    ("08/abr/2026", "Petrobras neutraliza o efeito do leilão de 31/mar e devolve a diferença às distribuidoras — decisão discricionária, não resposta automática de mercado.", True),
    ("2024–2026", "MME, CADE e ANP acompanham práticas concorrenciais na distribuição e revenda, com estudos (EPE, UNILA) apontando concentração e margem crescendo muito acima da inflação.", False),
]

FONTES = [
    ("Propano Mont Belvieu (semanal)", "FRED · WPROPANEMBTX (EIA)", "proxy",
     "Base do custo internacional. Propano puro — o GLP brasileiro é mistura propano/butano."),
    ("Câmbio BRL/USD (diário)", "BCB · SGS série 1 (PTAX venda)", "real", "Convertido para média mensal."),
    ("Preço do produtor (mensal)", "ANP/SDC · tabelas históricas 2001-2026", "real",
     "Linha “Preço de Realização do Produtor”, bloco nacional."),
    ("Preço ao consumidor (mensal)", "ANP/SDC · tabelas históricas 2001-2026", "real",
     "Linha “Preço Final ao Consumidor”, bloco nacional."),
    ("Tributos e margens (mensal)", "ANP/SDC · tabelas históricas 2001-2026", "real",
     "CIDE, PIS/COFINS, ICMS, margem de distribuição e de revenda."),
    ("Volume por distribuidora (mensal)", "ANP · relatório de vendas por recipiente", "real",
     "Base do HHI e do CR4. Inclui a Nacional Gás nominalmente."),
    ("Conversão galão → kg", "premissa: 1 gal ≈ 1,864 kg", "proxy",
     "Densidade padrão do propano líquido."),
]
