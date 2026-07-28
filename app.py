"""
Painel de monitoramento da cadeia de precos do GLP no Brasil
Nacional Gas · Grupo Edson Queiroz — Oficina de Ideias, Trainee GEQ 2026

Rodar:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import graficos as G
from src import cenarios as CEN, elasticidade as EL
from src import ingestao, irf, limpeza, lp as LP, midas as MD, ml as ML
from src import modelagem as M, vendas as V
from src import tema as T

st.set_page_config(
    page_title="GLP · Painel Nacional Gás",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(T.CSS_GLOBAL, unsafe_allow_html=True)

PT = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun",
      7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"}
rot = lambda ts: f"{PT[ts.month]}/{ts:%Y}"


def brl(v: float, casas=2) -> str:
    """Formata no padrao brasileiro: milhar com ponto, decimal com virgula."""
    s = f"{v:,.{casas}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


# ---------------------------------------------------------------------------
# Carga (cacheada em disco pelo modulo de ingestao + em memoria pelo Streamlit)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando séries…")
def carregar():
    decomp_bruto, ts_decomp = ingestao.decomposicao_anp()
    decomp, interpolados = limpeza.preparar_decomposicao(decomp_bruto)
    prop, ts_prop = ingestao.propano()
    camb, ts_camb = ingestao.cambio()
    dist, ts_dist = ingestao.distribuidoras()

    custo = limpeza.custo_internacional(prop, camb) if prop is not None and camb is not None else None
    painel = limpeza.painel_mensal(custo, decomp) if custo is not None else None
    return dict(decomp=decomp, ts_decomp=ts_decomp, painel=painel,
                interpolados=interpolados, cambio=camb, propano=prop,
                ts_dados=max(x for x in [ts_prop, ts_camb, ts_decomp] if x),
                dist=dist, ts_dist=ts_dist)


@st.cache_resource(show_spinner=False)
def irf_cache(_painel):
    """Estima o modelo de limiar uma vez por sessao (nao por interacao)."""
    return irf.estimar_mtar(irf.preparar(_painel))


@st.cache_data(show_spinner="Simulando trajetórias…")
def irf_girf_cache(_painel, choque, horizonte):
    par = irf_cache(_painel)
    return irf.girf(par, irf.preparar(_painel), choque, horizonte=horizonte,
                    n_sim=250, n_hist=40)


@st.cache_data(show_spinner=False)
def irf_linear_cache(_painel, choque, horizonte):
    return irf.irf_linear(irf.preparar(_painel), choque, horizonte=horizonte, n_boot=150)


@st.cache_data(show_spinner=False)
def lp_cache(_painel, choque, horizonte):
    d = irf.preparar(_painel)
    return LP.projecao_local(d, horizonte=horizonte, choque_pct=choque)


@st.cache_data(show_spinner="Estimando projeções locais…")
def lp_estado_cache(_painel, choque, horizonte):
    d = irf.preparar(_painel)
    return LP.projecao_local_estado(d, horizonte=horizonte, choque_pct=choque)


@st.cache_data(show_spinner=False)
def midas_cache():
    """Le o resultado pre-computado por `precomputar_midas.py`.

    A validacao completa re-estima o modelo ~47 vezes por especificacao em
    janela expansiva — minutos de CPU. E uma constatacao metodologica
    estatica, entao roda offline e o painel so exibe.
    """
    caminho = Path(__file__).resolve().parent / "cache" / "midas_resultado.json"
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Carregando vendas por UF…")
def vendas_cache():
    g, ts = V.carregar()
    return g, V.painel_nacional(g), V.concentracao_uf(g), V.cobertura(g), ts


@st.cache_data(show_spinner="Treinando modelo…")
def modelo_cache(data_corte):
    g, nac, conc, _cob, _ts = vendas_cache()
    cm = D["cambio"].copy()
    cm["mes"] = cm["data"].dt.to_period("M").dt.to_timestamp()
    cm = cm.groupby("mes", as_index=False)["ptax"].mean()
    macro = painel[["mes", "preco_produtor"]].merge(cm, on="mes", how="left")
    feats = ML.montar_features(nac, conc, macro)
    res = ML.treinar_avaliar(feats, data_corte=data_corte)
    return feats, res, ML.importancia_shap(res)


@st.cache_data(show_spinner="Estimando elasticidade da demanda…")
def elasticidade_cache():
    """Roda uma vez: a estimacao econometrica nao depende dos sliders."""
    g, _nac, _conc, _cob, _ts = vendas_cache()
    d = EL.painel_demanda(g, decomp)
    diag = EL.diagnostico_raiz_unitaria(d)
    r = EL.estimar(d)
    # especificacoes para a mistura do Monte Carlo: nivel e diferenca na
    # amostra principal, mais os cortes de robustez — a incerteza real inclui
    # a escolha de especificacao, nao so o IC de uma delas
    espec = [(r["nivel"]["eps"], r["nivel"]["se"]),
             (r["diferenca"]["eps"], r["diferenca"]["se"])]
    for ini in ("2013-01-01", "2015-01-01"):
        try:
            rr = EL.estimar(EL.painel_demanda(g, decomp, inicio=ini))
            espec.append((rr["nivel"]["eps"], rr["nivel"]["se"]))
        except Exception:
            pass
    return d, diag, r, espec


@st.cache_data(show_spinner=False)
def base_cenario():
    _g, nac_, _c, _cob, _ts = vendas_cache()
    u = nac_[nac_["ano"] == nac_["ano"].max()]
    vol = float(u.groupby("data")["botijoes_uf"].sum().mean())
    sh = float((u["share"] * u["botijoes_uf"]).sum() / u["botijoes_uf"].sum())
    return CEN.base_atual(decomp, vol, sh), int(u["ano"].iloc[0])


try:
    D = carregar()
except Exception as e:
    st.error(f"Falha ao carregar os dados: {e}")
    st.stop()

decomp, painel, dist = D["decomp"], D["painel"], D["dist"]

# ---------------------------------------------------------------------------
# Cabecalho
# ---------------------------------------------------------------------------
st.markdown("# Cadeia de preços do GLP no Brasil")
st.markdown(
    '<p style="font-size:15px;color:#52514e;margin-top:0;max-width:76ch;">'
    "Monitoramento do repasse de custo do produtor ao consumidor, com "
    "25 anos de dado público real — e o que a econometria diz sobre onde "
    "esse repasse trava.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="fonte">Nacional Gás · Grupo Edson Queiroz &nbsp;·&nbsp; '
    f'dados atualizados em {D["ts_dados"]:%d/%m/%Y %H:%M}</p>',
    unsafe_allow_html=True,
)

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Decomposição do preço",
    "Posição competitiva",
    "Radar regulatório",
    "Modelo de repasse",
    "Simulação de choque",
    "Demanda & Market Share",
    "Simulador de cenários",
])

# ===========================================================================
# ABA 1 — DECOMPOSICAO
# ===========================================================================
with t1:
    ult = decomp.dropna(subset=["preco_consumidor"]).iloc[-1]
    total = float(ult["preco_consumidor"])
    p_prod = 100 * float(ult["preco_produtor"]) / total
    p_trib = 100 * float(ult["tributos"]) / total
    p_marg = 100 * (float(ult["margem_distribuicao"]) + float(ult["margem_revenda"])) / total

    st.markdown(T.tiles([
        (f"R$ {brl(total)}", f"preço final ao consumidor · {rot(ult['mes'])}", ""),
        (f"{p_prod:.0f}%", "é o preço do produtor (Petrobras)", "accent"),
        (f"{p_trib:.0f}%", "são tributos", ""),
        (f"{p_marg:.0f}%", "são margens de distribuição e revenda", "warn"),
    ]), unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("##### Composição ao longo de 25 anos")
        st.markdown(
            '<p class="fonte"><span class="selo selo-real">DADO REAL</span> '
            "ANP/SDC · série mensal nacional, nov/2001 – jun/2026</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(G.area_decomposicao(decomp), width='stretch',
                        config={"displayModeBar": False})
    with c2:
        st.markdown(f"##### Onde vai cada real hoje · {rot(ult['mes'])}")
        st.markdown('<p class="fonte">&nbsp;</p>', unsafe_allow_html=True)
        st.plotly_chart(G.barra_composicao_atual(ult), width='stretch',
                        config={"displayModeBar": False})
        st.markdown(
            f'<div class="nota nota-alerta">A parcela que fica com '
            f"<b>distribuição e revenda</b> ({p_marg:.0f}%) é hoje maior que a parcela do "
            f"<b>produtor</b> ({p_prod:.0f}%). Em 2002, essa relação era inversa.</div>",
            unsafe_allow_html=True,
        )

    meses_interp = ", ".join(rot(m) for m in D["interpolados"]) or "nenhum"
    st.markdown(
        '<div class="nota">A reconciliação contábil (produtor + tributos + margem de '
        "distribuição = preço de distribuição; + margem de revenda = preço final) fecha em "
        "<b>290 dos 296 meses</b>. As exceções são nov/2001–mar/2002 (CIDE lançada em notação "
        "contábil negativa, não capturada na extração da tabela).<br>"
        f'<span class="selo selo-interp">INTERPOLADO</span> &nbsp;{meses_interp} — '
        "componente ausente na tabela original, preenchido por interpolação linear para não "
        "desenhar uma queda que não existiu.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Ver série mensal completa"):
        tab = decomp[["mes", "preco_produtor", "tributos", "margem_distribuicao",
                      "margem_revenda", "preco_consumidor"]].copy()
        tab["mes"] = tab["mes"].apply(rot)
        tab.columns = ["Mês", "Produtor", "Tributos", "Margem distrib.",
                       "Margem revenda", "Preço final"]
        st.dataframe(tab.iloc[::-1], width='stretch', hide_index=True, height=380)

# ===========================================================================
# ABA 2 — POSICAO COMPETITIVA
# ===========================================================================
with t2:
    if dist is None or dist.empty:
        st.warning("Dado de distribuidoras indisponível nesta sessão.")
    else:
        meses = sorted(dist["mes"].unique())
        col_f, _ = st.columns([1, 3])
        with col_f:
            mes_sel = st.selectbox("Mês de referência", meses, index=len(meses) - 1,
                                    format_func=lambda m: rot(pd.Timestamp(m)))

        hhi, cr4, v = limpeza.hhi_cr4(dist[dist["mes"] == mes_sel])
        classe, estilo = limpeza.classificar_hhi(hhi) if hhi else ("", "")

        ng = v[v["distribuidora"].str.upper().str.contains("NACIONAL")]
        ng_share = float(ng["share"].iloc[0]) * 100 if len(ng) else None
        ng_pos = int(ng.index[0]) + 1 if len(ng) else None

        st.markdown(T.tiles([
            (brl(hhi, 0), f"HHI — {classe}", estilo),
            (f"{cr4:.0f}%", "das vendas P13 nas 4 maiores", "warn"),
            (f"{brl(ng_share,1)}%" if ng_share else "—", "participação da Nacional Gás", "accent"),
            (f"{ng_pos}º" if ng_pos else "—", "posição no ranking nacional", ""),
        ]), unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.markdown("##### Participação por distribuidora (volume P13)")
            st.markdown(
                '<p class="fonte"><span class="selo selo-real">DADO REAL</span> '
                f"ANP · relatório de vendas por recipiente · {rot(pd.Timestamp(mes_sel))}</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(G.barras_market_share(v), width='stretch',
                            config={"displayModeBar": False})
        with c2:
            st.markdown("##### Concentração ao longo do tempo")
            st.markdown('<p class="fonte">CR4 — soma das 4 maiores</p>', unsafe_allow_html=True)
            hist = []
            for m in meses:
                _h, _c, _ = limpeza.hhi_cr4(dist[dist["mes"] == m])
                if _c:
                    hist.append({"mes": pd.Timestamp(m), "cr4": _c})
            if hist:
                st.plotly_chart(G.linha_cr4(pd.DataFrame(hist)), width='stretch',
                                config={"displayModeBar": False})

            ctx = M.CONTEXTO
            st.markdown(
                f'<div class="nota nota-alerta">Entre {ctx["epe"]["periodo"]}, a inflação '
                f'(IGP-M) subiu <b>{ctx["epe"]["igpm"]}%</b> — a margem líquida das '
                f'distribuidoras de GLP subiu <b>{ctx["epe"]["margem_liquida"]}%</b>, quase '
                f'4× mais.<br><span style="font-size:11px;">{ctx["epe"]["fonte"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="nota">Distribuição + revenda ficam com '
                f'<b>{ctx["unila"]["margem_glp"]}%</b> do preço do GLP, contra '
                f'<b>{ctx["unila"]["margem_gasolina"]}%</b> no caso da gasolina — apesar de '
                f'margens de produção quase idênticas.'
                f'<br><span style="font-size:11px;">{ctx["unila"]["fonte"]}</span></div>',
                unsafe_allow_html=True,
            )

# ===========================================================================
# ABA 3 — RADAR REGULATORIO
# ===========================================================================
with t3:
    st.markdown("##### Eventos que mudaram (ou podem mudar) a formação de preço")
    st.markdown(
        '<p class="fonte">Curadoria manual — atualizar à mão a cada novo evento.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        for data, texto, critico in M.EVENTOS:
            cls = "ev crit" if critico else "ev"
            st.markdown(
                f'<div class="{cls}"><div class="d">{data}</div>'
                f'<div class="t">{texto}</div></div>',
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown(
            '<div class="nota nota-alerta">O episódio de <b>mar–abr/2026</b> é o melhor '
            "exemplo de por que o repasse não é automático: diante de um choque que quase "
            "dobrou o Brent, a Petrobras <b>escolheu</b> não repassar e devolveu a diferença. "
            "Um modelo que assume ajuste contínuo de mercado erraria esse episódio inteiro."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="nota">Para a Nacional Gás, a leitura prática: o risco relevante não é '
            "só o preço internacional — é a <b>janela de decisão da Petrobras</b>, que hoje "
            "combina reajuste administrado, leilões e intervenções discricionárias.</div>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# ABA 4 — MODELO DE REPASSE
# ===========================================================================
with t4:
    col_f, col_a = st.columns([1.1, 2.4], gap="large")
    with col_f:
        janela = st.radio(
            "Janela amostral",
            ["Regime atual (ago/2017 – jun/2026)", "Série completa (2001 – 2026)"],
            index=0,
        )
    confirmado = janela.startswith("Regime atual")
    bloco = M.ELO1["confirmado"] if confirmado else M.ELO1["cheia"]

    with col_a:
        if confirmado:
            st.markdown(
                '<div class="nota nota-ok" style="margin-top:8px;">'
                "<b>Regime validado.</b> A quebra estrutural de jul/2017 foi confirmada por "
                "Gregory-Hansen com quebra endógena (p=0,018) e este regime cointegra "
                "sozinho (p=0,026). É a janela que descreve o mercado de hoje.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="nota nota-alerta" style="margin-top:8px;">'
                "<b>Cuidado ao citar esta janela.</b> Ela mistura o regime de preço "
                "administrado (até 2016) com o de paridade de importação. O regime "
                "pré-quebra <b>não</b> cointegra sozinho (p=0,42) — a “confirmação” aqui é "
                "em parte artefato do contraste entre os dois regimes.</div>",
                unsafe_allow_html=True,
            )

    d = painel.copy()
    if confirmado:
        d = d[d["mes"] >= pd.Timestamp("2017-08-01")]
    d = d.dropna(subset=["custo_intl_brl_p13", "preco_produtor", "preco_consumidor"])
    d = limpeza.indexar(d, ["custo_intl_brl_p13", "preco_produtor", "preco_consumidor"])

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("##### Custo internacional → produtor → consumidor")
        st.markdown(
            '<p class="fonte">'
            '<span class="selo selo-proxy">PROXY</span> custo internacional &nbsp; '
            '<span class="selo selo-real">DADO REAL</span> produtor e consumidor &nbsp;·&nbsp; '
            "índice base 100, escala log</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(G.linhas_indexadas(d, marcar_quebra=not confirmado),
                        width='stretch', config={"displayModeBar": False})
    with c2:
        st.markdown("##### Quanto tempo o repasse leva")
        st.markdown('<p class="fonte">Modelo M-TAR no regime validado</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(G.barras_meia_vida(), width='stretch',
                        config={"displayModeBar": False})
        st.markdown(
            '<div class="nota nota-alerta">Assimetria confirmada (Φ=12,6 · p=0,003): o preço '
            "do produtor volta ao equilíbrio em <b>2 meses</b> em condição normal, mas leva "
            "<b>ordem de um ano e meio ou mais</b> depois de um choque de alta. O coeficiente "
            "do regime de choque não é significativo isoladamente — trate os ~31 meses como "
            "ordem de grandeza, não como número exato.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.markdown(f"##### Elo 1 · custo internacional → produtor")
        st.markdown(
            f'<p class="fonte">{bloco["janela"]} · n={bloco["n"]} meses &nbsp; '
            f'<span class="selo selo-real">COINTEGRA</span></p>'
            if confirmado else
            f'<p class="fonte">{bloco["janela"]} · n={bloco["n"]} meses</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(bloco["linhas"], columns=["Teste", "Resultado", "Leitura"]),
            width='stretch', hide_index=True,
        )
    with e2:
        st.markdown("##### Elo 2 · produtor → consumidor")
        st.markdown(
            '<p class="fonte">Todas as janelas testadas &nbsp; '
            '<span class="selo selo-nao">NÃO COINTEGRA</span></p>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(M.ELO2["linhas"],
                         columns=["Janela", "Teste", "Resultado", "Veredito"]),
            width='stretch', hide_index=True,
        )
    st.markdown(f'<div class="nota nota-alerta">{M.ELO2["conclusao"]}</div>',
                unsafe_allow_html=True)

    with st.expander("Procedência de cada série (o que é dado real e o que é premissa)"):
        f = pd.DataFrame(M.FONTES, columns=["Série", "Fonte", "tipo", "Observação"])
        f["Tipo"] = f["tipo"].map({"real": "dado real", "proxy": "proxy / premissa"})
        st.dataframe(f[["Série", "Fonte", "Tipo", "Observação"]],
                     width='stretch', hide_index=True)
        st.markdown(
            '<div class="nota">Interpolação: gaps de até 2 meses são preenchidos por '
            "interpolação linear e ficam marcados internamente; gaps maiores permanecem "
            "vazios, nunca inventados. Na prática, só set/2020 foi interpolado.</div>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# ABA 5 — SIMULAÇÃO DE CHOQUE (resposta a impulso)
# ===========================================================================
with t5:
    st.markdown("##### Se o custo internacional mudar, o que acontece com o preço da Petrobras?")
    st.markdown(
        '<p class="fonte">Modelo de correção de erro com limiar estimado sobre o regime '
        "validado (ago/2017 – jun/2026, n=107). A resposta é obtida por simulação "
        "Monte Carlo — não é extrapolação de tendência.</p>",
        unsafe_allow_html=True,
    )

    cfg1, cfg2, cfg3 = st.columns([2, 1, 1], gap="large")
    with cfg1:
        choque = st.slider("Choque no custo internacional (%)", -50, 100, 30, step=5)
    with cfg2:
        horizonte = st.slider("Horizonte (meses)", 6, 36, 24, step=6)
    with cfg3:
        comparar = st.toggle("Comparar com IRF linear", value=True)

    try:
        par_irf = irf_cache(painel)
        g = irf_girf_cache(painel, choque, horizonte)
        lin = irf_linear_cache(painel, choque, horizonte) if comparar else None
    except Exception as e:
        st.error(f"Falha ao simular: {e}")
        st.stop()

    eq = g["equilibrio_pct"]
    med = g["mediana"]

    def _mes_para(frac):
        alvo = frac * eq
        for h, v in enumerate(med):
            if (eq > 0 and v >= alvo) or (eq < 0 and v <= alvo):
                return h
        return None

    h50, h90 = _mes_para(0.5), _mes_para(0.9)
    preco_hoje = float(decomp.dropna(subset=["preco_produtor"]).iloc[-1]["preco_produtor"])

    st.markdown(T.tiles([
        (f"{brl(eq, 1)}%", "repasse integral no equilíbrio de longo prazo", "accent"),
        (f"{h50} meses" if h50 is not None else "> horizonte", "para metade do repasse acontecer", ""),
        (f"{h90} meses" if h90 is not None else "> horizonte", "para 90% do repasse acontecer",
         "warn" if (h90 is None or h90 > 12) else ""),
        (f"R$ {brl(preco_hoje * eq / 100)}", "impacto final no preço por botijão", ""),
    ]), unsafe_allow_html=True)

    metodo = st.radio(
        "Método de estimação da resposta",
        ["GIRF — modelo de limiar (simulação)",
         "Local Projections — Jordà (2005), com assimetria por estado"],
        horizontal=True, label_visibility="collapsed",
    )
    usa_lp = metodo.startswith("Local")

    cg1, cg2 = st.columns([3, 2], gap="large")
    with cg1:
        if usa_lp:
            r_lp = lp_estado_cache(painel, abs(choque), horizonte)
            st.plotly_chart(G.grafico_lp_estado(r_lp, choque), width='stretch',
                            config={"displayModeBar": False})
            sig = [int(h) for h in r_lp["h"] if r_lp["p_simetria"][h] < 0.10]
            faixa = f"h={min(sig)} a h={max(sig)}" if sig else "nenhum horizonte"
            st.markdown(
                '<div class="nota nota-alerta"><b>Assimetria confirmada com inferência '
                f'formal.</b> O teste de igualdade entre os dois regimes rejeita simetria em '
                f"{faixa} (menor p = {r_lp['p_simetria'].min():.3f}). A direção é a clássica de "
                "<i>rockets and feathers</i>: <b>alta de custo é repassada; queda não é</b>. "
                "Isso <u>inverte</u> a leitura sugerida pelo M-TAR, que particiona a amostra "
                "por momento do desvio, não pelo sinal do choque — a partição por sinal é a "
                "que responde à pergunta de negócio.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(G.grafico_irf(g, lin, choque), width='stretch',
                            config={"displayModeBar": False})
    with cg2:
        st.plotly_chart(G.grafico_repasse_acumulado(g), width='stretch',
                        config={"displayModeBar": False})

        if choque > 0:
            st.markdown(
                '<div class="nota nota-ok">Choque de <b>alta</b>: o modelo diz que a Petrobras '
                "demora a repassar. Para a Nacional Gás, isso é uma <b>almofada de margem</b> "
                "temporária — o custo de aquisição sobe mais devagar que o custo internacional."
                "</div>",
                unsafe_allow_html=True,
            )
        elif choque < 0:
            st.markdown(
                '<div class="nota nota-alerta">Choque de <b>baixa</b>: a queda é repassada mais '
                "rápido do que a alta. A janela para capturar margem com estoque comprado caro "
                "é curta.</div>",
                unsafe_allow_html=True,
            )

    # --- tradução para volume real da Nacional Gás -------------------------
    if dist is not None and not dist.empty:
        ng_hist = dist[dist["distribuidora"].str.upper().str.contains("NACIONAL")]
        if len(ng_hist):
            kg_mes = float(ng_hist.sort_values("mes").iloc[-1]["p13_kg"])
            botijoes_mes = kg_mes / 13.0
            # diferença entre repasse imediato e repasse observado, mês a mês
            defasagem_pct = (eq - med[:min(13, len(med))]) / 100.0
            efeito_rs = float((defasagem_pct * preco_hoje * botijoes_mes).sum())
            sinal = "a menos" if choque > 0 else "a mais"
            st.markdown(
                f'<div class="nota"><b>Traduzindo para o volume real da Nacional Gás.</b> '
                f"Com {brl(botijoes_mes/1e6, 1)} milhões de botijões P13/mês "
                f"(volume declarado à ANP), a defasagem do repasse nos primeiros 12 meses "
                f"equivale a <b>R$ {brl(abs(efeito_rs)/1e6, 1)} milhões</b> {sinal} em custo "
                f"de aquisição, comparado a um repasse instantâneo. "
                f'<span class="selo selo-proxy">ESTIMATIVA</span> supõe volume constante e '
                f"repasse integral do preço do produtor ao custo de aquisição.</div>",
                unsafe_allow_html=True,
            )

    with st.expander("Como este número é calculado (e onde ele pode errar)"):
        st.markdown(
            f"""
**Modelo.** Relação de longo prazo estimada por Engle-Granger no regime validado:
`log(preço produtor) = a + {par_irf['beta']:.3f} · log(custo internacional)`.
O desvio desse equilíbrio corrige a uma velocidade que **depende do regime**:
ρ₁ = {par_irf['rho1']:.3f} quando o desvio está se fechando, ρ₂ = {par_irf['rho2']:.3f}
quando está se abrindo (limiar τ = {par_irf['tau']:.4f}).

**Por que GIRF e não IRF linear.** A IRF dos livros-texto é simétrica por construção:
devolve a mesma resposta para alta e baixa, só trocando o sinal. Como os testes
rejeitaram simetria (M-TAR, p = 0,003), ela **subestima a lentidão** do ajuste depois
de uma alta. A GIRF (Koop, Pesaran & Potter, 1996) resolve isso simulando as duas
trajetórias — com e sem choque — a partir das mesmas histórias observadas e dos
mesmos choques futuros, e tomando a diferença.

**Checagem cruzada.** Os parâmetros acima foram estimados de forma independente em
Python e em R (script `18_mtar_regime2_confirmado.R`). Batem: β 0,621 nos dois,
τ 0,0181 nos dois, meias-vidas 1,9 vs 2,0 e 30,4 vs 30,8 meses.

**Onde pode errar.**
- O custo internacional ainda é *proxy* (propano puro, sem butano/frete/tancagem).
- ρ₂ não é estatisticamente significativo isolado — a cauda longa da resposta é
  ordem de grandeza, não previsão precisa. A faixa de 80% no gráfico mostra isso.
- Os primeiros 1–2 meses da resposta não são distinguíveis de zero (a faixa cruza
  o eixo): o modelo tem um termo de momento que gera ruído no impacto imediato.
- Vale para o elo custo→**produtor**. O elo produtor→**consumidor** não cointegra,
  então não dá para encadear a simulação até a bomba.
"""
        )

    # --- validação metodológica: MIDAS -------------------------------------
    st.markdown("---")
    st.markdown("##### Vale a pena usar câmbio diário e propano semanal? (teste MIDAS)")
    mid = midas_cache()
    if mid is None:
        st.info("Resultado não pré-computado. Rode `python precomputar_midas.py` uma vez.")
    else:
        st.markdown(T.tiles([
            (f"{brl(mid['ganho_dentro'], 1)}%", "ganho de ajuste dentro da amostra", ""),
            (f"{brl(mid['oos_ganho'], 2)}%", "ganho fora da amostra · pesos disciplinados", ""),
            (f"{brl(mid['oos_ganho_livre'], 1)}%", "ganho fora da amostra · pesos livres", "warn"),
            (f"{mid['oos_n']}", "previsões fora da amostra avaliadas", ""),
        ]), unsafe_allow_html=True)

        st.markdown(
            '<div class="nota nota-alerta"><b>Resultado negativo — e informativo.</b> '
            f"Deixando o MIDAS ponderar livremente os últimos {mid['K_fx']} pregões de câmbio "
            f"e {mid['K_pr']} semanas de propano, os pesos ótimos saem <b>praticamente "
            f"uniformes</b> (entre {brl(100*mid['peso_fx_min'],1)}% e "
            f"{brl(100*mid['peso_fx_max'],1)}% por dia) — o modelo redescobre a média mensal "
            "simples. Com pesos flexíveis o ajuste dentro da amostra melhora "
            f"(R²aj {brl(mid['r2adj_midas'],2)}), mas a previsão fora da amostra <b>piora "
            f"{brl(abs(mid['oos_ganho_livre']), 1)}%</b>: era sobreajuste.<br><br>"
            "A leitura econômica é o ponto: o preço do produtor é <b>administrado</b>, revisto "
            "em decisões discretas. Não existe mecanismo pelo qual a trajetória do dólar "
            "<i>dentro</i> do mês carregue informação extra sobre essa decisão. MIDAS brilha "
            "onde a série de baixa frequência agrega um processo contínuo — volatilidade "
            "mensal a partir de retornos diários, por exemplo. Aqui, não é o caso.</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# ABA 6 — DEMANDA & MARKET SHARE
# ===========================================================================
with t6:
    try:
        g_v, nac_v, conc_v, cob_v, ts_v = vendas_cache()
    except Exception as e:
        st.error(f"Não foi possível carregar as vendas por UF: {e}")
        st.stop()

    # ---- Seção 4 (aviso) vem PRIMEIRO: não é rodapé -----------------------
    st.markdown(
        f'<div class="nota nota-alerta" style="margin-top:4px;">'
        f'<span class="selo selo-interp">DEFASAGEM DE DADO</span> &nbsp;'
        f'O detalhamento <b>por distribuidora</b> só existe até '
        f'<b>{rot(cob_v["fim"])}</b> — são <b>{cob_v["meses_atraso"]} meses</b> de '
        f'atraso, restrição de defesa da concorrência da própria ANP. Os meses '
        f'posteriores existem apenas agregados, sem separar quem vendeu. '
        f'Portanto: toda participação de mercado nesta aba, inclusive a que o modelo '
        f'usa como último valor observado, é de {rot(cob_v["fim"])}. '
        f'Não há como saber a participação de hoje com dado público.</div>',
        unsafe_allow_html=True,
    )

    ult_data = cob_v["fim"]
    ult = nac_v[nac_v["data"] == ult_data]
    share_nac = float((ult["share"] * ult["botijoes_uf"]).sum() / ult["botijoes_uf"].sum())
    n_lider = int((nac_v[nac_v["data"] == ult_data]["share"] > 0.30).sum())
    vol_ano = nac_v[nac_v["ano"] == ult_data.year]["botijoes_nacional"].sum()

    st.markdown(T.tiles([
        (f"{brl(100 * share_nac, 1)}%", f"participação nacional · {rot(ult_data)}", "accent"),
        (f"{brl(vol_ano / 1e6, 1)} mi", f"botijões P13 vendidos em {ult_data.year}", ""),
        (f"{cob_v['n_ufs']}", "UFs atendidas", ""),
        (f"{n_lider}", "UFs com participação acima de 30%", ""),
    ]), unsafe_allow_html=True)

    # ---- Seção 1 — visão geral -------------------------------------------
    st.markdown("##### Evolução da participação por UF")
    st.markdown(
        '<p class="fonte"><span class="selo selo-real">DADO REAL</span> '
        f"ANP/SIMP · P13 para revenda, {rot(cob_v['inicio'])} – {rot(cob_v['fim'])} · "
        f"{cob_v['n_celulas']:,} células UF×mês</p>".replace(",", "."),
        unsafe_allow_html=True,
    )

    cf1, cf2 = st.columns([2, 1])
    with cf1:
        ranking = (nac_v[nac_v["data"] == ult_data]
                   .sort_values("share", ascending=False)["uf_destino"].tolist())
        ufs_sel = st.multiselect("UFs", ranking, default=ranking[:4], max_selections=8)
    with cf2:
        metrica = st.radio("Métrica", ["Participação (%)", "Volume (botijões)"],
                           horizontal=True, label_visibility="collapsed")
    met = "share" if metrica.startswith("Part") else "volume"

    cv1, cv2 = st.columns([3, 2], gap="large")
    with cv1:
        if ufs_sel:
            st.plotly_chart(G.serie_share_uf(nac_v, ufs_sel, met), width="stretch",
                            config={"displayModeBar": False})
        else:
            st.info("Selecione ao menos uma UF.")
    with cv2:
        st.markdown(f"###### Participação por UF · {rot(ult_data)}")
        st.plotly_chart(G.barras_share_uf(nac_v, ult_data), width="stretch",
                        config={"displayModeBar": False})

    st.markdown(
        '<div class="nota">Checagem de sanidade executada na agregação: a soma das '
        "participações de <b>todos</b> os distribuidores fecha em 100,0000% nas "
        f"{cob_v['n_celulas']:,} células UF×mês, sem exceção.<br>".replace(",", ".") +
        "A <b>NGC Distribuidora</b> foi somada à Nacional Gás em 2021–2023: a própria ANP "
        "instrui a consolidá-las no período de transição da aquisição da Liquigás (CADE, "
        "nov/2020), e a incorporação formal ocorreu em mai/2023. Sem somar, a série mostraria "
        "uma queda artificial em 2021 seguida de recuperação — degrau contábil, não de "
        "mercado.<br>"
        "<b>Degraus nas séries por UF são reais, não falha de dado.</b> Em AP, por exemplo, "
        "o mercado existe desde 2007 com 5 distribuidoras e a Nacional Gás só passa a vender "
        "em mar/2018 — o 0% anterior é ausência verdadeira, não lacuna. Em AL, o salto no fim "
        "de 2011 vem junto com o mercado da UF triplicando e duas distribuidoras a mais: "
        "evento estrutural do mercado, que vale investigar antes de citar no pitch.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- Seção 2 — modelo -------------------------------------------------
    st.markdown("##### Modelo preditivo de participação (LightGBM)")
    corte = st.select_slider(
        "Corte temporal treino/teste",
        options=["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
        value="2022-01-01",
        format_func=lambda x: f"treina até {pd.Timestamp(x).year}",
    )

    try:
        feats, res, shp = modelo_cache(corte)

        st.markdown(T.tiles([
            (f"{brl(100 * res['mae_modelo'], 2)} pp", "erro médio do modelo (MAE, fora da amostra)", "accent"),
            (f"{brl(100 * res['mae_baseline'], 2)} pp", "erro do baseline ingênuo", ""),
            (f"{brl(res['ganho_mae'], 0)}%", "ganho sobre o baseline", ""),
            (f"{res['n_treino']} / {res['n_teste']}", "observações treino / teste", ""),
        ]), unsafe_allow_html=True)

        cm1, cm2 = st.columns([3, 2], gap="large")
        with cm1:
            st.plotly_chart(G.previsto_vs_real(res), width="stretch",
                            config={"displayModeBar": False})
            st.markdown(
                '<div class="nota">Validação por <b>corte temporal</b>, nunca k-fold '
                "aleatório: em painel de série temporal, embaralhar linhas coloca o futuro "
                "no treino e produz uma métrica otimista que não significa nada. O baseline "
                "de comparação é ingênuo de propósito — “a participação do mesmo mês do ano "
                "anterior”. Um modelo que não bate isso não deveria ir para o slide.</div>",
                unsafe_allow_html=True,
            )
        with cm2:
            st.markdown("###### O que move a previsão (SHAP)")
            st.plotly_chart(G.shap_importancia(shp["importancia"], ML.ROTULOS),
                            width="stretch", config={"displayModeBar": False})

        top = shp["importancia"].iloc[0]["feature"]
        st.markdown(
            '<div class="nota nota-alerta"><b>Leitura honesta do SHAP.</b> O que domina a '
            f"previsão é <b>{ML.ROTULOS.get(top, top).lower()}</b> — ou seja, participação de "
            "mercado é <b>persistente</b>: o melhor previsor do mês que vem é o mês passado. "
            "Preço do produtor e concentração da UF aparecem em seguida, mas com impacto uma "
            "ordem de grandeza menor. Isso é informativo, não decepcionante: significa que "
            "ganho de participação em GLP se constrói por movimento estrutural (rede de "
            "revenda, aquisição, logística), não por oscilação de preço mês a mês.</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Modelo indisponível para este corte: {e}")


# ===========================================================================
# ABA 7 — SIMULADOR DE CENÁRIOS
# ===========================================================================
with t7:
    try:
        d_el, diag_el, r_el, espec_el = elasticidade_cache()
        base_cen, ano_base = base_cenario()
    except Exception as e:
        st.error(f"Não foi possível preparar o simulador: {e}")
        st.stop()

    niv, dif = r_el["nivel"], r_el["diferenca"]

    # ---- Seção 1 — elasticidade ------------------------------------------
    st.markdown("##### 1 · Elasticidade-preço da demanda agregada")
    st.markdown(
        '<p class="fonte"><span class="selo selo-real">DADO REAL</span> volume ANP/SIMP '
        '(todos os distribuidores) &nbsp; <span class="selo selo-proxy">ESTIMATIVA</span> '
        f'regressão log-log, {len(d_el)} meses desde {rot(d_el["mes"].min())} · '
        'preço deflacionado pelo IPCA</p>',
        unsafe_allow_html=True,
    )

    st.markdown(T.tiles([
        (f"{brl(niv['eps'], 3)}", "elasticidade (nível) — estimativa central", "accent"),
        (f"[{brl(niv['ic'][0], 2)}; {brl(niv['ic'][1], 2)}]", "intervalo de confiança 95%", ""),
        (f"{brl(dif['eps'], 3)}", "elasticidade (1ª diferença) — confirma a ordem", ""),
        (f"{brl(100 * (1 - abs(niv['eps'])), 0)}%", "do aumento de preço fica em receita (demanda inelástica)", ""),
    ]), unsafe_allow_html=True)

    ce1, ce2 = st.columns([3, 2], gap="large")
    with ce1:
        st.plotly_chart(
            G.dispersao_elasticidade(d_el, niv["eps"],
                                     float(r_el["modelo_nivel"].params["const"])),
            width="stretch", config={"displayModeBar": False})
    with ce2:
        st.markdown(
            '<div class="nota nota-alerta"><span class="selo selo-nao">LIMITAÇÃO</span> '
            "<b>Viés de simultaneidade não resolvido.</b> Preço e quantidade se determinam "
            "juntos no equilíbrio de mercado: um choque de demanda move os dois. O "
            "coeficiente de MQO mistura elasticidade de demanda e de oferta. Não usamos "
            "variável instrumental aqui — o candidato natural (custo internacional) é o "
            "mesmo choque que alimenta o simulador, e instrumento mal validado é pior que "
            "nenhum. <b>Trate como ordem de grandeza, não como elasticidade estrutural "
            "causal.</b> Instrumentar é o próximo passo.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="nota nota-alerta"><span class="selo selo-nao">LIMITAÇÃO</span> '
            f"<b>A versão em 1ª diferença não é significativa a 5%</b> "
            f"(p = {brl(dif['p'], 3)}, IC [{brl(dif['ic'][0], 2)}; {brl(dif['ic'][1], 2)}] "
            "cruza o zero). Ou seja: a resposta <i>mês a mês</i> não está precisamente "
            "identificada — o que o dado sustenta é a relação de médio prazo, não uma "
            "reação imediata a cada variação de preço.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="nota">Amostra iniciada em '
        f'{rot(d_el["mes"].min())}: antes disso o volume agregado da ANP cresce +33% ao ano '
        "(2008–2010), o que é ampliação da base de declarantes, não do mercado. Incluir esses "
        "anos levava a elasticidade a −1,34 — implausível para bem de necessidade. Na amostra "
        "limpa, as especificações em nível e em diferença convergem "
        f"({brl(niv['eps'], 3)} e {brl(dif['eps'], 3)}), o que não acontecia antes.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- Seção 2 — motor do simulador -------------------------------------
    st.markdown("##### 2 · Simulador")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        choque_c = st.slider("Choque no custo internacional (%)", -50, 100, 30, 5,
                             key="cen_choque")
    with sc2:
        repasse_pct = st.slider("Repasse ao preço final (%)", 0, 100, 50, 5,
                                key="cen_repasse",
                                help="Variável de escolha, não estimada: as abas 4 e 5 "
                                     "mostraram que o elo produtor→consumidor não segue "
                                     "equilíbrio competitivo.")
    with sc3:
        horiz_c = st.slider("Horizonte (meses)", 6, 24, 12, 3, key="cen_horizonte")

    choque_share = st.slider("Choque na participação de mercado (pp) — cenários de M&A",
                             -5.0, 5.0, 0.0, 0.5, key="cen_share")

    g_cen = irf_girf_cache(painel, choque_c, horiz_c)
    repasse = repasse_pct / 100.0
    cen = CEN.cenario(base_cen, g_cen["mediana"], repasse, niv["eps"], choque_share)
    fim = cen.iloc[-1]

    p_dist_base = base_cen["preco_consumidor"] - base_cen["margem_revenda"]
    receita_base = p_dist_base * base_cen["volume_mercado"] * base_cen["share"]

    st.markdown(T.tiles([
        (f"R$ {brl(receita_base * 12 / 1e9, 2)} bi", f"receita-base anual · P13 revenda ({ano_base})", ""),
        (f"R$ {brl(fim['preco_consumidor'])}", f"preço ao consumidor em {horiz_c} meses", ""),
        (f"{brl(100 * (fim['volume_mercado'] / base_cen['volume_mercado'] - 1), 2)}%", "volume de mercado", ""),
        (f"{brl(fim['margem_var_pct'], 1)}%", "margem bruta da Nacional Gás",
         "warn" if fim["margem_var_pct"] < 0 else "accent"),
    ]), unsafe_allow_html=True)

    # ---- Seção 3 — Monte Carlo -------------------------------------------
    st.markdown("##### 3 · Incerteza propagada (Monte Carlo, 5.000 réplicas)")
    mc = CEN.monte_carlo(base_cen, g_cen["mediana"], g_cen["p10"], g_cen["p90"],
                         espec_el, repasse, choque_share, n_sim=5000)

    cmc1, cmc2 = st.columns(2, gap="large")
    with cmc1:
        st.plotly_chart(G.leque_monte_carlo(mc, "receita"), width="stretch",
                        config={"displayModeBar": False})
    with cmc2:
        st.plotly_chart(G.leque_monte_carlo(mc, "margem"), width="stretch",
                        config={"displayModeBar": False})

    st.markdown(
        f'<div class="nota">A incerteza propagada combina três fontes: a <b>elasticidade</b> '
        f"(sorteando entre {len(espec_el)} especificações — nível, 1ª diferença e cortes de "
        "robustez — e não só o IC de uma delas, porque já vimos o número se mover com a "
        "amostra); a <b>banda da GIRF</b> do repasse ao produtor; e a <b>participação de "
        "mercado</b> (desvio de 1 pp, da ordem do erro fora da amostra do modelo da aba 6)."
        f"<br>Em {brl(100 * mc['frac_eps_positiva'], 1)}% dos sorteios a elasticidade sai "
        "positiva — herança do IC da especificação em diferença, que cruza o zero. Não são "
        "descartados: descartá-los seria impor a conclusão que queremos.</div>",
        unsafe_allow_html=True,
    )

    # ---- Seção 4 — comparação de cenários ---------------------------------
    st.markdown("---")
    st.markdown("##### 4 · Repassar ou absorver?")

    linhas = []
    for frac, rotulo in [(1.0, "Repasse integral (100%)"),
                         (repasse, f"Repasse escolhido ({repasse_pct}%)"),
                         (0.0, "Absorver tudo (0%)")]:
        c = CEN.cenario(base_cen, g_cen["mediana"], frac, niv["eps"], choque_share).iloc[-1]
        linhas.append(dict(
            rotulo=rotulo,
            volume=100 * (c["volume_mercado"] / base_cen["volume_mercado"] - 1),
            receita=c["receita_var_pct"],
            margem=c["margem_var_pct"],
        ))

    cc1, cc2 = st.columns([3, 2], gap="large")
    with cc1:
        st.plotly_chart(G.comparacao_repasse(linhas), width="stretch",
                        config={"displayModeBar": False})
    with cc2:
        integral, absorver = linhas[0], linhas[-1]
        ganho_vol = absorver["volume"] - integral["volume"]
        custo_marg = integral["margem"] - absorver["margem"]
        if choque_c > 0:
            st.markdown(
                f'<div class="nota nota-alerta"><b>O trade-off é assimétrico.</b> Absorver '
                f"todo o choque em vez de repassá-lo compra apenas "
                f"<b>{brl(ganho_vol, 2)} pp</b> de volume — porque a demanda é inelástica "
                f"(ε = {brl(niv['eps'], 2)}: quem cozinha a gás não para de cozinhar por "
                f"causa do preço) — e custa <b>{brl(custo_marg, 1)} pp</b> de margem bruta."
                f"<br><br>Ou seja: <b>segurar preço não compra volume relevante, só queima "
                f"margem.</b> A decisão de repasse é de posicionamento e de exposição "
                f"regulatória, não de defesa de participação de mercado.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="nota">Com choque de baixa, o sentido se inverte: repassar a '
                "queda reduz preço e receita, mas a demanda inelástica devolve pouco volume "
                "em troca. Segurar o preço na queda é o que preserva margem.</div>",
                unsafe_allow_html=True,
            )
