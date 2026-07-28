# Painel técnico — cadeia de preços do GLP no Brasil

Nacional Gás · Grupo Edson Queiroz. Sete abas interativas: decomposição do
preço, posição competitiva, radar regulatório, modelo de repasse (M-TAR),
simulação de choque (GIRF), demanda & market share (LightGBM + SHAP) e
simulador de cenários (elasticidade + Monte Carlo).

A versão resumida, não-interativa, está em
**[gondimiere.github.io/projeto-geq](https://gondimiere.github.io/projeto-geq/)**.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Este repositório é autônomo

O projeto completo (`PROJETO_GEQ/`) tem pastas de dado bruto fora daqui —
`dados_brutos/` (tabelas históricas da ANP extraídas de PDF) e `GLP/`
(microdados de vendas, ~32 MB). Este repositório **não inclui** essas pastas:
ao invés disso, o código lê primeiro uma cópia leve versionada aqui dentro.

| Arquivo grande do projeto-mãe | O que roda aqui |
|---|---|
| `dados_brutos/anp_decomposicao_nacional_2001_2026.csv` (15 KB) | Cópia em `dados_estaticos/` |
| `GLP/GLP_Vendas_Historico.csv` (32 MB) | **Não incluído.** O app usa só `cache/vendas_uf_agente.parquet` (852 KB), já processado — o CSV bruto nunca é lido em produção. |

`src/ingestao.py` procura primeiro em `dados_estaticos/` (este repo) e só cai
para o caminho do projeto-mãe se estiver rodando localmente dentro da árvore
completa. `src/vendas.py` já teria essa mesma proteção naturalmente: seu
`carregar()` só abre o CSV de 32 MB se o parquet em cache não existir.

**Consequência prática:** se algum dia for preciso reprocessar os dados do
zero (novo mês de vendas, nova extração da ANP), isso precisa rodar na
máquina com acesso ao projeto completo — não neste repositório publicado.
Depois, copiar os artefatos atualizados para `dados_estaticos/` e `cache/`
e dar commit.

## Cache versionado de propósito

A pasta `cache/` (parquet + json, ~1 MB) está no git, não no `.gitignore`.
Isso faz o app:

- não precisar retreinar o modelo de participação (LightGBM) nem
  reprocessar vendas a cada reinício
- não depender de FRED/BCB/ANP estarem no ar no exato momento do cold start
  do Streamlit Cloud — câmbio e propano se atualizam sozinhos quando o cache
  vence (1 dia), mas partem de um estado válido

`cache/midas_resultado.json` é a validação fora-da-amostra do MIDAS
(`precomputar_midas.py`) — leva minutos para calcular, então roda offline;
o app só lê.

## Publicar no Streamlit Community Cloud

1. Repositório já deve estar no GitHub (ver passos de deploy no chat)
2. Em [share.streamlit.io](https://share.streamlit.io) → **New app**
   - **Repository:** `gondimiere/projeto-geq-tecnico`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Depois de publicado, copiar a URL gerada (`algo.streamlit.app`) e:
   - atualizar `URL_TECNICO` em `../painel_executivo/gerar_site.py` e
     `../painel_executivo/app.py`, e regerar o site estático

## Estrutura

| Caminho | Papel |
|---|---|
| `app.py` | As 7 abas. |
| `src/` | Ingestão, limpeza, modelagem (M-TAR, GIRF, LP, MIDAS, elasticidade, cenários, ML). |
| `dados_estaticos/` | Cópia leve de um dado que só existe via extração de PDF. |
| `cache/` | Séries e resultados pré-computados, versionados. |
| `precomputar_midas.py` | Roda a validação fora-da-amostra do MIDAS; grava em `cache/`. |
