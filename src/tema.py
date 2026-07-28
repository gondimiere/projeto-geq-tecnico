"""
Tema visual do painel — paleta e template Plotly.

A paleta e a de referencia do metodo de dataviz, usada na ORDEM documentada:
slots 1..4 = azul, laranja, agua, amarelo. Essa ordem e o mecanismo de
seguranca para daltonismo (nao e escolha estetica) — ela passa todos os
gates na pairlist adjacente (stacks, barras, linhas). Em particular, amarelo
nunca encosta em laranja num stack de 4 series porque agua fica entre eles.

O painel se compromete com o modo CLARO: e feito para projecao e impressao
num pitch presencial. E uma escolha, nao um esquecimento.
"""

# --- Slots categoricos (modo claro) ---------------------------------------
SERIE_1 = "#2a78d6"   # azul
SERIE_2 = "#eb6834"   # laranja
SERIE_3 = "#1baf7a"   # agua
SERIE_4 = "#eda100"   # amarelo

# --- Chrome e tinta --------------------------------------------------------
SUPERFICIE = "#fcfcfb"   # superficie do grafico
PLANO = "#f9f9f7"        # plano da pagina
TINTA_1 = "#0b0b0b"      # tinta primaria
TINTA_2 = "#52514e"      # tinta secundaria
TINTA_MUDA = "#898781"   # eixos e rotulos
GRADE = "#e1e0d9"        # linha de grade (fio de cabelo)
LINHA_BASE = "#c3c2b7"   # linha de base / eixo
BORDA = "rgba(11,11,11,0.10)"

# --- Status (reservado - nunca vira "serie 5") -----------------------------
STATUS_BOM = "#0ca30c"
STATUS_ATENCAO = "#fab219"
STATUS_GRAVE = "#ec835a"
STATUS_CRITICO = "#d03b3b"

FONTE = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Cinza recessivo para as barras nao destacadas na aba de concentracao
# (nao e um slot categorico: e "todo mundo menos a entidade em foco")
CINZA_RECESSIVO = "#c9c8c2"


def aplicar_layout(fig, altura=420, titulo=None, titulo_y=None, legenda=True):
    """Aplica o chrome padrao a uma figura Plotly.

    Marcas finas, grade recessiva, sem moldura, hover unificado.
    """
    # title=None faz o Plotly renderizar a string "undefined" no SVG;
    # a chave precisa ser omitida, nao anulada
    if titulo:
        fig.update_layout(title=dict(
            text=titulo, font=dict(size=15, color=TINTA_1),
            x=0, xanchor="left", y=0.97,
        ))

    fig.update_layout(
        height=altura,
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        font=dict(family=FONTE, size=13, color=TINTA_2),
        margin=dict(l=8, r=8, t=48 if titulo else 16, b=8),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=SUPERFICIE,
            bordercolor=BORDA,
            font=dict(family=FONTE, size=12, color=TINTA_1),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=12, color=TINTA_2),
            bgcolor="rgba(0,0,0,0)",
        ) if legenda else dict(visible=False),
        showlegend=legenda,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=LINHA_BASE,
        linewidth=1,
        ticks="outside",
        tickcolor=LINHA_BASE,
        ticklen=4,
        tickfont=dict(size=11, color=TINTA_MUDA),
        title=None,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRADE,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(size=11, color=TINTA_MUDA),
    )
    if titulo_y:
        fig.update_yaxes(title=dict(text=titulo_y, font=dict(size=11, color=TINTA_MUDA)))
    return fig


CSS_GLOBAL = f"""
<style>
  .stApp {{ background: {PLANO}; }}
  html, body, [class*="css"] {{ font-family: {FONTE}; }}

  /* tipografia */
  h1 {{ font-size: 26px !important; font-weight: 680 !important; color: {TINTA_1};
        letter-spacing: -0.015em; margin-bottom: 2px !important; }}
  h2 {{ font-size: 19px !important; font-weight: 640 !important; color: {TINTA_1};
        letter-spacing: -0.01em; margin: 4px 0 2px 0 !important; }}
  h3 {{ font-size: 15px !important; font-weight: 620 !important; color: {TINTA_1}; }}
  p, li {{ color: {TINTA_2}; font-size: 14px; line-height: 1.55; }}

  /* abas */
  .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {GRADE}; }}
  .stTabs [data-baseweb="tab"] {{
      height: 42px; padding: 0 18px; background: transparent;
      font-size: 14px; font-weight: 560; color: {TINTA_MUDA};
  }}
  .stTabs [aria-selected="true"] {{ color: {TINTA_1} !important; box-shadow: inset 0 -2px 0 {SERIE_1}; }}

  /* cartoes de estatistica */
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1px; background: {GRADE}; border: 1px solid {GRADE};
            border-radius: 8px; overflow: hidden; margin: 6px 0 18px 0; }}
  .tile {{ background: {SUPERFICIE}; padding: 14px 16px; }}
  .tile .v {{ font-size: 25px; font-weight: 660; color: {TINTA_1}; line-height: 1.15;
              letter-spacing: -0.02em; }}
  .tile .v.accent {{ color: {SERIE_1}; }}
  .tile .v.warn {{ color: {STATUS_CRITICO}; }}
  .tile .k {{ font-size: 11.5px; color: {TINTA_MUDA}; margin-top: 5px; line-height: 1.35; }}

  /* selos de procedencia do dado */
  .selo {{ display: inline-block; font-size: 10.5px; font-weight: 660;
           letter-spacing: 0.03em; padding: 2px 8px; border-radius: 10px;
           vertical-align: 2px; }}
  .selo-real {{ background: #e3f3e3; color: #0a5c0a; }}
  .selo-proxy {{ background: #fdeae1; color: #a33d14; }}
  .selo-interp {{ background: #fdf2d9; color: #8a6100; }}
  .selo-nao {{ background: #fae3e3; color: #93201f; }}

  /* caixas de nota */
  .nota {{ border-left: 2px solid {LINHA_BASE}; padding: 9px 0 9px 13px;
           margin: 12px 0; font-size: 12.5px; color: {TINTA_MUDA}; line-height: 1.55; }}
  .nota-alerta {{ border-left-color: {STATUS_CRITICO}; color: {TINTA_2}; }}
  .nota-ok {{ border-left-color: {STATUS_BOM}; color: {TINTA_2}; }}
  .nota b {{ color: {TINTA_1}; }}

  /* fonte / timestamp */
  .fonte {{ font-size: 11.5px; color: {TINTA_MUDA}; margin: -2px 0 10px 0; }}

  /* linha do tempo */
  .ev {{ border-left: 2px solid {GRADE}; padding: 0 0 20px 18px; position: relative; }}
  .ev::before {{ content:""; position:absolute; left:-5px; top:4px; width:8px; height:8px;
                 border-radius:50%; background:{SERIE_1}; }}
  .ev.crit::before {{ background: {STATUS_CRITICO}; }}
  .ev .d {{ font-size: 11.5px; font-weight: 660; color: {SERIE_1}; letter-spacing: 0.02em; }}
  .ev.crit .d {{ color: {STATUS_CRITICO}; }}
  .ev .t {{ font-size: 13.5px; color: {TINTA_2}; margin-top: 3px; line-height: 1.5; }}

  /* tabela */
  .stDataFrame {{ border: 1px solid {GRADE}; border-radius: 8px; }}

  [data-testid="stMetricValue"] {{ font-size: 24px; color: {TINTA_1}; }}
  footer, #MainMenu {{ visibility: hidden; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
</style>
"""


def tile(valor, rotulo, estilo=""):
    cls = f"v {estilo}".strip()
    return f'<div class="tile"><div class="{cls}">{valor}</div><div class="k">{rotulo}</div></div>'


def tiles(itens):
    """itens: lista de (valor, rotulo, estilo)"""
    corpo = "".join(tile(v, r, e) for v, r, e in itens)
    return f'<div class="tiles">{corpo}</div>'
