"""
Construtores de grafico. Regras seguidas aqui:

  - marcas finas (linha 2px), grade recessiva, sem moldura
  - paleta categorica na ordem documentada (slots 1..4), nunca ciclada
  - eixo unico sempre; duas medidas de escala diferente => indexadas a 100
  - camada de hover por padrao (crosshair unificado em linha/area)
  - legenda presente para >=2 series + rotulo direto seletivo (nunca em
    todo ponto)
  - texto em tinta neutra; a cor fica na marca, nao no numero
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import tema as T

MESES_PT = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun",
            7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"}


def _rot(ts) -> str:
    return f"{MESES_PT[ts.month]}/{ts:%Y}"


# ---------------------------------------------------------------------------
# Aba 1 — decomposicao do preco (area empilhada)
# ---------------------------------------------------------------------------
def area_decomposicao(d: pd.DataFrame) -> go.Figure:
    """Ordem do stack (base -> topo): produtor, tributos, margem dist., margem revenda.

    A ordem dos slots e o mecanismo de seguranca para daltonismo: agua fica
    entre laranja e amarelo, entao esse par (o unico problematico da paleta)
    nunca encosta.
    """
    camadas = [
        ("preco_produtor", "Preço do produtor", T.SERIE_1),
        ("tributos", "Tributos (CIDE+PIS/COFINS+ICMS)", T.SERIE_2),
        ("margem_distribuicao", "Margem de distribuição", T.SERIE_3),
        ("margem_revenda", "Margem de revenda", T.SERIE_4),
    ]
    fig = go.Figure()
    for col, nome, cor in camadas:
        fig.add_trace(go.Scatter(
            x=d["mes"], y=d[col], name=nome,
            mode="lines", stackgroup="preco",
            fillcolor=cor,
            # fio de 2px na cor da superficie entre segmentos empilhados
            line=dict(width=2, color=T.SUPERFICIE),
            hovertemplate="%{fullData.name}: <b>R$ %{y:.2f}</b><extra></extra>",
        ))

    T.aplicar_layout(fig, altura=430, titulo_y="R$ por botijão P13")
    fig.update_yaxes(ticksuffix="", tickprefix="R$ ", rangemode="tozero")
    fig.update_xaxes(dtick="M48", tickformat="%Y")
    return fig


# ---------------------------------------------------------------------------
# Aba 1 — composicao relativa hoje (barra empilhada horizontal unica)
# ---------------------------------------------------------------------------
def barra_composicao_atual(linha: pd.Series) -> go.Figure:
    camadas = [
        ("preco_produtor", "Produtor", T.SERIE_1),
        ("tributos", "Tributos", T.SERIE_2),
        ("margem_distribuicao", "Margem distrib.", T.SERIE_3),
        ("margem_revenda", "Margem revenda", T.SERIE_4),
    ]
    total = sum(float(linha[c]) for c, _, _ in camadas)
    fig = go.Figure()
    for col, nome, cor in camadas:
        v = float(linha[col])
        pct = 100 * v / total
        fig.add_trace(go.Bar(
            x=[v], y=[""], name=nome, orientation="h",
            marker=dict(color=cor, line=dict(color=T.SUPERFICIE, width=2)),
            # rotulo direto so onde cabe (fatias >= 12%)
            text=[f"{nome}<br><b>{pct:.0f}%</b>" if pct >= 12 else ""],
            textposition="inside", insidetextanchor="middle",
            # tinta escura em todos os segmentos: contra amarelo e agua o
            # branco fica em ~2:1 (ilegivel). Escuro passa >4.5:1 nos quatro.
            textfont=dict(color=T.TINTA_1, size=12, family=T.FONTE),
            hovertemplate=f"{nome}: <b>R$ %{{x:.2f}}</b> ({pct:.1f}%)<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=120, showlegend=False,
        paper_bgcolor=T.SUPERFICIE, plot_bgcolor=T.SUPERFICIE,
        margin=dict(l=0, r=0, t=6, b=6),
        font=dict(family=T.FONTE, color=T.TINTA_2),
        hoverlabel=dict(bgcolor=T.SUPERFICIE, bordercolor=T.BORDA,
                        font=dict(family=T.FONTE, size=12, color=T.TINTA_1)),
        bargap=0,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ---------------------------------------------------------------------------
# Aba 2 — participacao de mercado (barra horizontal, entidade em foco)
# ---------------------------------------------------------------------------
def barras_market_share(v: pd.DataFrame, foco="NACIONAL", top=12) -> go.Figure:
    d = v.head(top).copy()
    d["is_foco"] = d["distribuidora"].str.upper().str.contains(foco)
    def _curto(nome: str, limite=24) -> str:
        s = nome.title()
        for padrao in (r"\s+Distribuidora.*$", r"\s+Distrib.*$", r"\s+Armazenadora.*$",
                       r"\s+Comercio.*$", r"\s+(Ltda|S\.?\s?A\.?|S/A|Epp)\.?$"):
            s = pd.Series([s]).str.replace(padrao, "", regex=True).iloc[0]
        s = s.strip()
        if len(s) <= limite:
            return s
        # corta em fronteira de palavra, nunca no meio
        corte = s[:limite].rsplit(" ", 1)[0]
        return (corte or s[:limite]) + "…"

    d["curto"] = d["distribuidora"].map(_curto)
    d = d.iloc[::-1]  # maior no topo

    cores = [T.SERIE_1 if f else T.CINZA_RECESSIVO for f in d["is_foco"]]
    fig = go.Figure(go.Bar(
        x=d["share"] * 100, y=d["curto"], orientation="h",
        marker=dict(color=cores, cornerradius=3),
        text=[f"{s*100:.1f}%" for s in d["share"]],
        textposition="outside",
        textfont=dict(size=11.5, color=T.TINTA_2, family=T.FONTE),
        cliponaxis=False,
        hovertemplate="%{customdata}<br><b>%{x:.2f}%</b> do volume P13<extra></extra>",
        customdata=d["distribuidora"],
    ))
    T.aplicar_layout(fig, altura=max(340, 26 * len(d) + 60), legenda=False)
    fig.update_xaxes(showgrid=True, gridcolor=T.GRADE, ticksuffix="%",
                     showline=False, ticks="", zeroline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=T.TINTA_2))
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=52, t=10, b=8))
    return fig


# ---------------------------------------------------------------------------
# Aba 2 — evolucao do CR4 (linha unica, sem legenda)
# ---------------------------------------------------------------------------
def linha_cr4(serie: pd.DataFrame) -> go.Figure:
    # sem preenchimento: o eixo comeca em 50%, e area sob eixo truncado
    # sugere magnitude a partir do zero, que aqui seria falso
    fig = go.Figure(go.Scatter(
        x=serie["mes"], y=serie["cr4"], mode="lines",
        line=dict(color=T.SERIE_1, width=2),
        hovertemplate="<b>%{y:.1f}%</b> nas 4 maiores<extra></extra>",
    ))
    # ponto final enfatizado + rotulo direto (so no ultimo ponto)
    ult = serie.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[ult["mes"]], y=[ult["cr4"]], mode="markers+text",
        marker=dict(color=T.SERIE_1, size=9,
                    line=dict(color=T.SUPERFICIE, width=2)),
        text=[f"  {ult['cr4']:.1f}%"], textposition="middle right",
        textfont=dict(size=12, color=T.TINTA_1, family=T.FONTE),
        hoverinfo="skip", showlegend=False, cliponaxis=False,
    ))
    T.aplicar_layout(fig, altura=260, legenda=False, titulo_y="CR4 (% do volume P13)")
    fig.update_yaxes(range=[50, 100], ticksuffix="%")
    fig.update_layout(margin=dict(l=8, r=56, t=12, b=8))
    return fig


# ---------------------------------------------------------------------------
# Aba 4 — tres series indexadas (eixo unico, escala log)
# ---------------------------------------------------------------------------
def linhas_indexadas(d: pd.DataFrame, marcar_quebra=False) -> go.Figure:
    series = [
        ("idx_custo_intl_brl_p13", "Custo internacional (proxy)", T.SERIE_1, "dot"),
        ("idx_preco_produtor", "Preço do produtor (real)", T.SERIE_2, "solid"),
        ("idx_preco_consumidor", "Preço ao consumidor (real)", T.SERIE_3, "solid"),
    ]
    fig = go.Figure()
    for col, nome, cor, dash in series:
        if col not in d:
            continue
        fig.add_trace(go.Scatter(
            x=d["mes"], y=d[col], name=nome, mode="lines",
            line=dict(color=cor, width=2, dash=dash),
            hovertemplate="%{fullData.name}: <b>%{y:.0f}</b><extra></extra>",
        ))

    if marcar_quebra:
        q = pd.Timestamp("2017-07-01")
        eixo = pd.to_datetime(d["mes"])   # aceita Timestamp ou date
        if eixo.min() <= q <= eixo.max():
            # datetime nativo (nao pd.Timestamp): o serializador do Plotly
            # aceita os dois, mas o exportador estatico (kaleido/orjson) so
            # aceita o nativo
            qx = q.to_pydatetime()
            fig.add_vline(x=qx, line=dict(color=T.LINHA_BASE, width=1, dash="dash"))
            fig.add_annotation(
                x=qx, y=1, yref="paper", yanchor="bottom",
                text="quebra estrutural (jul/2017)", showarrow=False,
                font=dict(size=10.5, color=T.TINTA_MUDA, family=T.FONTE),
                xanchor="left", xshift=6, yshift=-14,
            )

    T.aplicar_layout(fig, altura=430, titulo_y="Índice (base 100 no início da janela)")
    # ticks explicitos no log: o automatico deixa metade do grafico sem
    # nenhuma linha de grade abaixo de 100
    fig.update_yaxes(type="log", tickmode="array",
                     tickvals=[50, 60, 80, 100, 150, 200, 250, 300, 350, 400],
                     ticktext=["50", "60", "80", "100", "150", "200", "250", "300", "350", "400"])
    fig.update_xaxes(tickformat="%Y")
    return fig


# ---------------------------------------------------------------------------
# Aba 5 — resposta a impulso
# ---------------------------------------------------------------------------
def grafico_irf(g: dict, linear: dict | None, choque_pct: float) -> go.Figure:
    """GIRF assimetrica com banda de 80% + IRF linear como referencia."""
    fig = go.Figure()

    # banda de incerteza (mesma cor da linha, bem diluida)
    fig.add_trace(go.Scatter(
        x=np.r_[g["h"], g["h"][::-1]],
        y=np.r_[g["p90"], g["p10"][::-1]],
        fill="toself", fillcolor="rgba(42,120,214,0.13)",
        line=dict(width=0), hoverinfo="skip",
        name="faixa de 80%", showlegend=True,
    ))

    # equilibrio de longo prazo
    fig.add_hline(y=g["equilibrio_pct"], line=dict(color=T.LINHA_BASE, width=1, dash="dash"))
    fig.add_annotation(
        x=g["h"][-1], y=g["equilibrio_pct"], xanchor="right", yanchor="bottom",
        text=f"repasse integral: {g['equilibrio_pct']:+.1f}%".replace(".", ","),
        showarrow=False, font=dict(size=10.5, color=T.TINTA_MUDA, family=T.FONTE),
    )

    if linear is not None:
        fig.add_trace(go.Scatter(
            x=linear["h"], y=linear["central"], name="IRF linear (simétrica)",
            mode="lines", line=dict(color=T.TINTA_MUDA, width=2, dash="dot"),
            hovertemplate="linear: <b>%{y:+.1f}%</b><extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=g["h"], y=g["mediana"], name="GIRF assimétrica (modelo estimado)",
        mode="lines", line=dict(color=T.SERIE_1, width=2.5),
        hovertemplate="mês %{x}: <b>%{y:+.1f}%</b><extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color=T.LINHA_BASE, width=1))
    T.aplicar_layout(fig, altura=400,
                     titulo_y="Resposta do preço do produtor (%)")
    fig.update_xaxes(title=dict(text="meses após o choque",
                                font=dict(size=11, color=T.TINTA_MUDA)),
                     dtick=3, showgrid=False)
    return fig


def grafico_repasse_acumulado(g: dict) -> go.Figure:
    """Quanto do repasse ja aconteceu, mes a mes (% do total de longo prazo)."""
    eq = g["equilibrio_pct"]
    pct = 100 * g["mediana"] / eq if eq != 0 else g["mediana"] * 0
    pct = np.clip(pct, 0, 130)

    fig = go.Figure(go.Bar(
        x=g["h"], y=pct, marker=dict(color=T.SERIE_1, cornerradius=2),
        hovertemplate="mês %{x}: <b>%{y:.0f}%</b> do repasse<extra></extra>",
    ))
    fig.add_hline(y=100, line=dict(color=T.LINHA_BASE, width=1, dash="dash"))
    T.aplicar_layout(fig, altura=240, legenda=False,
                     titulo_y="% do repasse já ocorrido")
    fig.update_xaxes(title=dict(text="meses", font=dict(size=11, color=T.TINTA_MUDA)),
                     dtick=3, showgrid=False)
    fig.update_yaxes(range=[0, 125], ticksuffix="%")
    fig.update_layout(hovermode="closest")
    return fig


def grafico_lp_estado(r: dict, choque_pct: float) -> go.Figure:
    """LP dependente de estado: resposta a alta vs baixa do custo, com faixas
    HAC e marcacao dos horizontes onde a assimetria e significativa."""
    fig = go.Figure()
    h = r["h"]

    # faixa onde a assimetria e estatisticamente significativa (10%)
    sig = np.where(r["p_simetria"] < 0.10)[0]
    if len(sig):
        fig.add_vrect(x0=float(h[sig.min()]) - 0.5, x1=float(h[sig.max()]) + 0.5,
                      fillcolor="rgba(137,135,129,0.10)", line_width=0, layer="below")
        fig.add_annotation(
            x=(h[sig.min()] + h[sig.max()]) / 2, y=1, yref="paper", yanchor="bottom",
            text="assimetria significativa (p&lt;0,10)", showarrow=False, yshift=-14,
            font=dict(size=10, color=T.TINTA_MUDA, family=T.FONTE),
        )

    for chave, nome, cor, rgba in [
        ("alta", f"Custo sobe {abs(choque_pct):.0f}%", T.STATUS_CRITICO, "rgba(208,59,59,0.13)"),
        ("baixa", f"Custo cai {abs(choque_pct):.0f}%", T.SERIE_1, "rgba(42,120,214,0.13)"),
    ]:
        s = r[chave]
        fig.add_trace(go.Scatter(
            x=np.r_[h, h[::-1]], y=np.r_[s["p90"], s["p10"][::-1]],
            fill="toself", fillcolor=rgba, line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=h, y=s["central"], name=nome, mode="lines",
            line=dict(color=cor, width=2.5),
            hovertemplate=nome + ": <b>%{y:+.1f}%</b><extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color=T.LINHA_BASE, width=1))
    T.aplicar_layout(fig, altura=400, titulo_y="Resposta do preço do produtor (%)")
    fig.update_xaxes(title=dict(text="meses após o choque",
                                font=dict(size=11, color=T.TINTA_MUDA)),
                     dtick=3, showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Aba 4 — barras de meia-vida (o numero-chave do pitch)
# ---------------------------------------------------------------------------
def barras_meia_vida() -> go.Figure:
    rotulos = ["Custo <b>cai</b><br>(ajuste normal)", "Custo <b>dispara</b><br>(choque de alta)"]
    valores = [2.0, 31.0]
    # cores de status (nao slots categoricos): aqui a cor carrega estado
    # (rapido = bom / lento = critico), e cada barra ja vem rotulada, entao
    # a identidade nunca depende so da cor
    cores = [T.STATUS_BOM, T.STATUS_CRITICO]

    fig = go.Figure(go.Bar(
        x=rotulos, y=valores, marker=dict(color=cores, cornerradius=4),
        width=0.42,
        text=["2 meses", "~31 meses"], textposition="outside",
        textfont=dict(size=15, color=T.TINTA_1, family=T.FONTE),
        cliponaxis=False,
        hovertemplate="%{x}<br>meia-vida: <b>%{y:.0f} meses</b><extra></extra>",
    ))
    T.aplicar_layout(fig, altura=300, legenda=False,
                     titulo_y="Meia-vida do ajuste (meses)")
    fig.update_yaxes(range=[0, 38])
    fig.update_xaxes(tickfont=dict(size=12.5, color=T.TINTA_2))
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=8, t=26, b=8))
    return fig


# ---------------------------------------------------------------------------
# Aba 6 — Demanda & Market Share
# ---------------------------------------------------------------------------
def serie_share_uf(d: pd.DataFrame, ufs: list[str], metrica="share") -> go.Figure:
    """Participacao (ou volume) da Nacional Gas por UF ao longo do tempo.

    Ate 4 UFs recebem cor propria (slots categoricos na ordem documentada);
    acima disso o grafico vira 'uma em foco + as demais recessivas', porque
    a paleta nao garante separacao para daltonismo alem do 4o slot.
    """
    cores = [T.SERIE_1, T.SERIE_2, T.SERIE_3, T.SERIE_4]
    fig = go.Figure()
    muitos = len(ufs) > 4

    for i, uf in enumerate(ufs):
        s = d[d["uf_destino"] == uf].sort_values("data")
        if metrica == "share":
            y, tmpl = s["share"] * 100, "%{fullData.name}: <b>%{y:.1f}%</b><extra></extra>"
        else:
            y, tmpl = s["botijoes_nacional"] / 1e6, "%{fullData.name}: <b>%{y:.2f} mi</b><extra></extra>"
        fig.add_trace(go.Scatter(
            x=s["data"], y=y, name=uf, mode="lines",
            line=dict(color=cores[i] if not muitos else T.CINZA_RECESSIVO,
                      width=2 if not muitos else 1.4),
            hovertemplate=tmpl,
        ))

    T.aplicar_layout(
        fig, altura=400,
        titulo_y="Participação da Nacional Gás (%)" if metrica == "share"
                 else "Volume Nacional Gás (milhões de botijões)")
    if metrica == "share":
        fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(tickformat="%Y", dtick="M24")
    return fig


def barras_share_uf(d: pd.DataFrame, data_ref) -> go.Figure:
    """Participacao por UF num mes de referencia, ordenada."""
    s = d[d["data"] == data_ref].sort_values("share")
    s = s[s["botijoes_uf"] > 0]
    fig = go.Figure(go.Bar(
        x=s["share"] * 100, y=s["uf_destino"], orientation="h",
        marker=dict(color=T.SERIE_1, cornerradius=2),
        hovertemplate="%{y}: <b>%{x:.1f}%</b> de participação<extra></extra>",
    ))
    T.aplicar_layout(fig, altura=max(360, 17 * len(s) + 60), legenda=False)
    fig.update_xaxes(showgrid=True, gridcolor=T.GRADE, ticksuffix="%",
                     showline=False, ticks="", zeroline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=T.TINTA_2))
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=20, t=10, b=8))
    return fig


def previsto_vs_real(res: dict) -> go.Figure:
    """Previsto x observado no periodo de TESTE, agregado por mes."""
    t = res["teste"].copy()
    t["previsto"] = res["pred"]
    ag = (t.groupby("data")
            .apply(lambda x: pd.Series({
                "real": np.average(x["share"], weights=x["botijoes_uf"]),
                "prev": np.average(x["previsto"], weights=x["botijoes_uf"]),
            }), include_groups=False)
            .reset_index())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ag["data"], y=ag["real"] * 100, name="Observado", mode="lines+markers",
        line=dict(color=T.SERIE_1, width=2.5), marker=dict(size=6),
        hovertemplate="observado: <b>%{y:.2f}%</b><extra></extra>"))
    fig.add_trace(go.Scatter(
        x=ag["data"], y=ag["prev"] * 100, name="Previsto (LightGBM)", mode="lines+markers",
        line=dict(color=T.SERIE_2, width=2.5, dash="dot"), marker=dict(size=6),
        hovertemplate="previsto: <b>%{y:.2f}%</b><extra></extra>"))

    T.aplicar_layout(fig, altura=320, titulo_y="Participação nacional (%)")
    fig.update_yaxes(ticksuffix="%")
    return fig


def shap_importancia(imp: pd.DataFrame, rotulos: dict, top=10) -> go.Figure:
    """Importancia media |SHAP| — magnitude, entao rampa de um hue so."""
    s = imp.head(top).iloc[::-1].copy()
    s["rotulo"] = s["feature"].map(lambda f: rotulos.get(f, f))
    vmax = s["shap_abs_medio"].max()
    # rampa sequencial azul: mais escuro = mais importante
    cores = [f"rgba(42,120,214,{0.30 + 0.70 * v / vmax:.2f})" for v in s["shap_abs_medio"]]

    fig = go.Figure(go.Bar(
        x=s["shap_abs_medio"] * 100, y=s["rotulo"], orientation="h",
        marker=dict(color=cores, cornerradius=2),
        hovertemplate="%{y}<br><b>%{x:.3f} pp</b> de impacto médio<extra></extra>",
    ))
    T.aplicar_layout(fig, altura=max(300, 30 * len(s) + 50), legenda=False)
    fig.update_xaxes(showgrid=True, gridcolor=T.GRADE, showline=False, ticks="",
                     zeroline=False, ticksuffix=" pp",
                     title=dict(text="impacto médio na previsão (|SHAP|)",
                                font=dict(size=11, color=T.TINTA_MUDA)))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11.5, color=T.TINTA_2))
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=20, t=10, b=8))
    return fig


# ---------------------------------------------------------------------------
# Aba 7 — Simulador de cenarios
# ---------------------------------------------------------------------------
def dispersao_elasticidade(d: pd.DataFrame, eps: float, intercepto: float) -> go.Figure:
    """log(preco real) x log(volume), com a reta ajustada."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["log_preco"], y=d["log_vol"], mode="markers",
        marker=dict(color=T.SERIE_1, size=7, opacity=0.55,
                    line=dict(color=T.SUPERFICIE, width=1)),
        name="meses observados",
        customdata=d["mes"].dt.strftime("%b/%Y"),
        hovertemplate="%{customdata}<br>preço log %{x:.3f} · volume log %{y:.3f}<extra></extra>",
    ))
    xs = np.linspace(d["log_preco"].min(), d["log_preco"].max(), 50)
    fig.add_trace(go.Scatter(
        x=xs, y=intercepto + eps * xs, mode="lines",
        line=dict(color=T.STATUS_CRITICO, width=2.5),
        name=f"ajuste · ε = {eps:.3f}".replace(".", ","),
        hoverinfo="skip",
    ))
    T.aplicar_layout(fig, altura=340, titulo_y="log(volume mensal)")
    fig.update_xaxes(title=dict(text="log(preço real ao consumidor)",
                                font=dict(size=11, color=T.TINTA_MUDA)), showgrid=True,
                     gridcolor=T.GRADE)
    fig.update_layout(hovermode="closest")
    return fig


def leque_monte_carlo(mc: dict, chave="receita") -> go.Figure:
    """Fan chart P10/P50/P90 da variação de receita ou margem."""
    if chave == "receita":
        p10, p50, p90 = mc["var_p10"], mc["var_p50"], mc["var_p90"]
        rot, cor, rgba = "Receita", T.SERIE_1, "rgba(42,120,214,0.16)"
    else:
        p10, p50, p90 = mc["mvar_p10"], mc["mvar_p50"], mc["mvar_p90"]
        rot, cor, rgba = "Margem bruta", T.STATUS_CRITICO, "rgba(208,59,59,0.16)"
    h = mc["h"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.r_[h, h[::-1]], y=np.r_[p90, p10[::-1]],
        fill="toself", fillcolor=rgba, line=dict(width=0),
        name="faixa P10–P90", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=h, y=p50, mode="lines", line=dict(color=cor, width=2.5),
        name=f"{rot} (mediana)",
        hovertemplate="mês %{x}: <b>%{y:+.1f}%</b><extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=T.LINHA_BASE, width=1))
    T.aplicar_layout(fig, altura=360, titulo_y=f"{rot} vs. cenário-base (%)")
    fig.update_xaxes(title=dict(text="meses após o choque",
                                font=dict(size=11, color=T.TINTA_MUDA)),
                     dtick=3, showgrid=False)
    return fig


def comparacao_repasse(linhas: list[dict]) -> go.Figure:
    """Barras agrupadas: volume, receita e margem sob cada política de repasse."""
    metricas = [("volume", "Volume", T.SERIE_3),
                ("receita", "Receita", T.SERIE_1),
                ("margem", "Margem bruta", T.STATUS_CRITICO)]
    fig = go.Figure()
    rotulos = [l["rotulo"] for l in linhas]
    for chave, nome, cor in metricas:
        fig.add_trace(go.Bar(
            x=rotulos, y=[l[chave] for l in linhas], name=nome,
            marker=dict(color=cor, cornerradius=3),
            text=[f"{l[chave]:+.1f}%" for l in linhas],
            textposition="outside",
            textfont=dict(size=11, color=T.TINTA_2, family=T.FONTE),
            cliponaxis=False,
            hovertemplate=nome + ": <b>%{y:+.2f}%</b><extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=T.LINHA_BASE, width=1))
    T.aplicar_layout(fig, altura=340, titulo_y="variação vs. cenário-base (%)")
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08,
                      hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    return fig
