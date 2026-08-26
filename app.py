import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
import glob
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dotenv import load_dotenv
from google import genai

# Carregar variáveis de ambiente do arquivo chaves.env
load_dotenv("chaves.env")

st.set_page_config(page_title="Dashboard Análise NBA", page_icon="🏀", layout="wide")

# Paleta de Cores NBA
NBA_BLUE = '#1d428a'
NBA_RED = '#c8102e'
NBA_LIGHT = '#f9f9f9'

# Estilos Customizados
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_dados_particionados():
    """Lê e junta os arquivos divididos (agora referem-se aos jogadores)."""
    arquivos = glob.glob('dados_parte_*.csv')
    lista_dfs = []
    
    for arquivo in arquivos:
        df = pd.read_csv(arquivo)
        lista_dfs.append(df)
        
    return pd.concat(lista_dfs, ignore_index=True)

@st.cache_data
def load_data():
    """Carrega todas as bases e aplica os filtros iniciais."""
    team_df = pd.read_csv("team_traditional.csv")
    
    # player_df agora recebe as partes concatenadas
    player_df = carregar_dados_particionados()

 # Filtra para incluir apenas temporada regular e playoffs
    team_df = team_df[team_df['type'].isin(['regular', 'playoff'])]
    player_df = player_df[player_df['type'].isin(['regular', 'playoff'])]

    return team_df, player_df
# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🏀 NBA StatWeek")
    st.markdown("""
    **Hackathon StatWeek III**
    
    Este dashboard foi projetado para extrair inteligência tática e tendências da NBA, unindo **Estatística** e **Inteligência Artificial**.
    """)
    st.divider()
    st.info("Navegue pelas abas na tela principal para explorar os insights de dados.")
    st.caption("Desenvolvido para análise competitiva.")

# ----------------- HERO HEADER -----------------
st.title("Estatísticas e Histórico da NBA (1996-2026)")

try:
    team_df, player_df = load_data()
except FileNotFoundError:
    st.error("Arquivos de dados não encontrados no diretório especificado.")
    st.stop()

with st.container(border=True):
    st.write("**Dimensões do Banco de Dados Analítico:**")
    col_hero1, col_hero2, col_hero3 = st.columns(3)
    col_hero1.metric("Temporadas Analisadas", f"{team_df['season'].nunique()}")
    col_hero2.metric("Partidas Analisadas", f"{len(team_df):,}".replace(",", "."))
    col_hero3.metric("Jogadores Registrados", f"{player_df['playerid'].nunique():,}".replace(",", "."))


tabs = st.tabs([
    "📈 Evolução do Jogo", 
    "🧪 Teste de Hipótese", 
    "⭐ Destaques Individuais", 
    "🏠 Fator Casa vs Visitante",
    "🧭 A Bússola da Vitória"
])

# ----------------- TAB 1: Séries Temporais -----------------
with tabs[0]:
    st.header("Cenário Atual vs Cenários Anteriores")
    st.write("Analise como o estilo de jogo e as métricas coletivas mudaram ao longo das temporadas.")
    
    season_stats = team_df.groupby('season')[['PTS', '3PM', '3P%', 'FT%', 'PF']].mean().reset_index()
    
    metrics = {
        'PTS': 'Média de Pontos por Jogo',
        '3PM': 'Cestas de 3 Pontos Convertidas',
        '3P%': 'Aproveitamento de 3 Pontos (%)',
        'FT%': 'Aproveitamento de Lances Livres (%)',
        'PF': 'Faltas Cometidas'
    }
    
    with st.container(border=True):
        selected_metric = st.selectbox(
            "Selecione o indicador que deseja analisar:", 
            list(metrics.keys()), 
            format_func=lambda x: metrics[x],
            help="Escolha uma estatística para ver como a liga se comportou ao longo dos anos."
        )
    
    with st.container(border=True):
        fig1 = px.line(season_stats, x='season', y=selected_metric, markers=True, 
                       color_discrete_sequence=[NBA_RED])
        fig1.update_layout(
            title=f"Evolução: {metrics[selected_metric]}",
            xaxis_title="Temporada",
            yaxis_title=metrics[selected_metric],
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig1, width="stretch")

# ----------------- TAB 2: Teste de Hipótese -----------------
with tabs[1]:
    st.header("Análise de Agressividade e Mudanças de Perfil")
    st.write("Teste estatisticamente se houve mudança significativa no perfil das equipes entre diferentes épocas.")
    
    decades = {
        "Anos 90 (1993-1997)": [1993, 1994, 1995, 1996, 1997],
        "Anos 2000 (2003-2007)": [2003, 2004, 2005, 2006, 2007],
        "Anos 2010 (2013-2017)": [2013, 2014, 2015, 2016, 2017],
        "Anos 2020 (2020-2026)": [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    }
    
    with st.container(border=True):
        st.markdown("### ⚙️ Configuração do Teste")
        col1, col2, col3 = st.columns(3)
        with col1:
            grupo1_name = st.selectbox("Período Base (Grupo 1):", list(decades.keys()), index=0, help="Década mais antiga de referência.")
        with col2:
            grupo2_name = st.selectbox("Período de Comparação (Grupo 2):", list(decades.keys()), index=3, help="Década mais recente para comparar.")
        with col3:
            var_teste = st.selectbox("Variável de Análise:", ['PTS', 'PF', '3PM', '3PA', 'FGA'], format_func=lambda x: {
                'PTS': 'Pontos (Ofensividade)',
                'PF': 'Faltas (Fisicalidade)',
                '3PM': 'Cestas de 3 (Eficiência Externa)',
                '3PA': 'Tentativas de 3 (Volume Externo)',
                'FGA': 'Arremessos de Quadra (Ritmo/Pace)'
            }[x], help="Qual variável você deseja testar estatisticamente?")
            
        st.write("") # Espaçamento leve
        executar_teste = st.button("Executar Análise Estatística", type="primary", use_container_width=True)
        
    if executar_teste:
        with st.container(border=True):
            g1_seasons = decades[grupo1_name]
            g2_seasons = decades[grupo2_name]
            
            df_g1 = team_df[team_df['season'].isin(g1_seasons)].groupby(['season', 'teamid'])[var_teste].mean().reset_index()
            df_g2 = team_df[team_df['season'].isin(g2_seasons)].groupby(['season', 'teamid'])[var_teste].mean().reset_index()
            
            data1 = df_g1[var_teste].dropna()
            data2 = df_g2[var_teste].dropna()
            
            if len(data1) == 0 or len(data2) == 0:
                st.warning("Dados insuficientes para um dos grupos.")
            else:
                c1, c2 = st.columns(2)
                c1.metric(f"Média - {grupo1_name}", f"{data1.mean():.2f}", f"Desvio Padrão: {data1.std():.2f}", delta_color="off")
                c2.metric(f"Média - {grupo2_name}", f"{data2.mean():.2f}", f"Desvio Padrão: {data2.std():.2f}", delta_color="off")
                
                stat_l, p_levene = stats.levene(data1, data2)
                equal_var = p_levene > 0.05
                stat_t, p_valor = stats.ttest_ind(data1, data2, equal_var=equal_var)
                
                st.markdown("### Conclusão do Teste")
                if p_valor < 0.05:
                    st.success(f"**Diferença Significativa Detectada (p-valor = {p_valor:.4e}).** Os períodos apresentam perfis estatisticamente distintos para esta métrica.")
                else:
                    st.info(f"**Sem Diferença Significativa (p-valor = {p_valor:.4f}).** Não há evidências suficientes para afirmar que os períodos são diferentes.")
                    
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=data1, name=grupo1_name, marker_color=NBA_BLUE))
                fig_box.add_trace(go.Box(y=data2, name=grupo2_name, marker_color=NBA_RED))
                fig_box.update_layout(title=f"Distribuição de {var_teste}", yaxis_title=var_teste, template="plotly_dark")
                st.plotly_chart(fig_box, width="stretch")

# ----------------- TAB 3: Melhores Jogadores -----------------
with tabs[2]:
    st.header("Recordistas e Destaques por Temporada")
    st.write("Descubra os líderes em estatísticas. O valor destacado compara o topo atual com o topo da temporada anterior.")
    
    anos_disponiveis = sorted(player_df['season'].unique(), reverse=True)
    
    with st.container(border=True):
        ano_selecionado = st.selectbox("Selecione a Temporada:", anos_disponiveis, index=0, help="Escolha o ano para ver os líderes.")
    
    def get_top_players(df, year):
        df_ano = df[df['season'] == year]
        if df_ano.empty: return None
        games_played = df_ano.groupby('playerid')['gameid'].count()
        valid_players = games_played[games_played >= 10].index
        df_valid = df_ano[df_ano['playerid'].isin(valid_players)]
        if df_valid.empty: return None
        return df_valid.groupby(['playerid', 'player'])[['PTS', 'FG%', '3PM', '3P%', 'FT%', 'REB', 'AST', 'STL', 'BLK']].mean().reset_index()
    
    player_stats = get_top_players(player_df, ano_selecionado)
    ano_anterior = ano_selecionado - 1
    player_stats_prev = get_top_players(player_df, ano_anterior)
    
    stats_config = {
        'PTS': 'Pontos por Jogo',
        'FG%': 'Aproveitamento de Quadra (%)',
        '3PM': 'Cestas de 3 por Jogo',
        '3P%': 'Aproveitamento de 3 (%)',
        'FT%': 'Lances Livres (%)',
        'REB': 'Rebotes por Jogo',
        'AST': 'Assistências por Jogo',
        'STL': 'Roubos por Jogo',
        'BLK': 'Tocos por Jogo'
    }
    
    if player_stats is not None:
        with st.container(border=True):
            st.markdown(f"### Líderes da Temporada {ano_selecionado}")
            cols = st.columns(3)
            for idx, (col_stat, title) in enumerate(stats_config.items()):
                top_player = player_stats.sort_values(by=col_stat, ascending=False).iloc[0]
                val_atual = top_player[col_stat]
                nome_atual = top_player['player']
                
                delta_val = None
                if player_stats_prev is not None:
                    top_player_prev = player_stats_prev.sort_values(by=col_stat, ascending=False).iloc[0]
                    val_anterior = top_player_prev[col_stat]
                    delta_val = val_atual - val_anterior
                
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.metric(
                            label=title, 
                            value=f"{val_atual:.1f} ({nome_atual})", 
                            delta=f"{delta_val:.1f} vs Ano Anterior" if delta_val is not None else "Sem dados"
                        )

# ----------------- TAB 4: Casa vs Visitante -----------------
with tabs[3]:
    st.header("O Impacto do Mando de Quadra")
    st.write("Analise se jogar em casa realmente oferece uma vantagem estatística ao longo das décadas.")
    
    home_games = team_df[team_df['team'] == team_df['home']].copy()
    home_agg = home_games.groupby(['season', 'team'])['win'].agg(vit_casa='sum', jogos_casa='count').reset_index()
    home_agg['der_casa'] = home_agg['jogos_casa'] - home_agg['vit_casa']
    
    away_games = team_df[team_df['team'] == team_df['away']].copy()
    away_agg = away_games.groupby(['season', 'team'])['win'].agg(vit_fora='sum', jogos_fora='count').reset_index()
    away_agg['der_fora'] = away_agg['jogos_fora'] - away_agg['vit_fora']
    
    team_season_perf = pd.merge(home_agg, away_agg, on=['season', 'team'])
    team_season_perf['der_casa'] = team_season_perf['der_casa'].replace(0, 1)
    team_season_perf['der_fora'] = team_season_perf['der_fora'].replace(0, 1)
    
    team_season_perf['razao_casa'] = ((team_season_perf['vit_casa'] / team_season_perf['der_casa']) - 1) * 100
    team_season_perf['razao_fora'] = ((team_season_perf['vit_fora'] / team_season_perf['der_fora']) - 1) * 100
    
    yearly_perf = team_season_perf.groupby('season')[['vit_casa', 'vit_fora', 'razao_casa', 'razao_fora']].mean().reset_index()
    
    with st.container(border=True):
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=yearly_perf['season'], y=yearly_perf['vit_casa'], mode='lines', name='Em Casa', line=dict(color=NBA_BLUE, width=3)))
            fig2.add_trace(go.Scatter(x=yearly_perf['season'], y=yearly_perf['vit_fora'], mode='lines', name='Visitante', line=dict(color=NBA_RED, width=3)))
            fig2.update_layout(title="Volume Médio de Vitórias", xaxis_title="Temporada", yaxis_title="Vitórias", template="plotly_dark", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig2, width="stretch")
            
        with col_chart2:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=yearly_perf['season'], y=yearly_perf['razao_casa'], mode='lines', name='Razão Casa', line=dict(color=NBA_BLUE, width=3)))
            fig3.add_trace(go.Scatter(x=yearly_perf['season'], y=yearly_perf['razao_fora'], mode='lines', name='Razão Visitante', line=dict(color=NBA_RED, width=3)))
            fig3.update_layout(title="Proporção Vitória/Derrota (%)", xaxis_title="Temporada", yaxis_title="Razão (%)", template="plotly_dark", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig3, width="stretch")

    with st.expander("❓ Como interpretar a Proporção Vitória/Derrota?"):
        st.write("A razão é calculada pela fórmula `((Vitórias / Derrotas) - 1) * 100`.")
        st.write("Valores positivos indicam superioridade de vitórias. Por exemplo, 20% significa que o time vence 20% a mais do que perde. Valores negativos indicam o oposto.")

# ----------------- TAB 5: Regressão Logística -----------------
with tabs[4]:
    st.header("A Bússola da Vitória: O que faz um time ganhar?")
    st.write("Descubra, através de uma Regressão Logística Dinâmica, quais estatísticas mais aumentam a chance de vitória do time.")
    
    with st.expander("🔍 Visualize a metodologia estatística usada neste modelo"):
        st.markdown("""
        **Pipeline de Modelagem Rigoroso:**
        1. **Padronização:** Todas as métricas são transformadas em *Z-scores*. Isso nos permite comparar de forma justa o peso de diferentes estatísticas no mesmo gráfico (ex: comparar 1 rebote com 1% de aproveitamento).
        2. **Filtro de Multicolinearidade:** O modelo calcula dinamicamente o Fator de Inflação da Variância (VIF). Variáveis muito correlacionadas entre si (VIF > 5) são removidas para não distorcer os coeficientes.
        3. **Seleção de Variáveis (Backward Elimination):** O algoritmo descarta iterativamente qualquer estatística cujo **p-valor seja superior a 0.05 (5%)**. O gráfico final exibe estritamente as estatísticas com comprovação estatística forte para prever vitórias.
        4. **Razão de Chances (Odds Ratio):** O impacto gerado é convertido em *Odds Ratios*. Valores acima de 1 indicam que a variável aumenta a chance de vitória, e abaixo de 1 indicam que diminui.
        """)
    
    anos_disp = sorted(team_df['season'].unique())
    min_ano, max_ano = int(min(anos_disp)), int(max(anos_disp))
    
    
    with st.form("form_regressao"):
        st.write("### ⚙️ Configuração do Modelo")
        sel_anos = st.slider("Selecione o período de análise (Temporadas):", min_value=min_ano, max_value=max_ano, value=(max_ano-5, max_ano), help="O modelo aprenderá apenas com os jogos dentro dessa janela de tempo.")
        gerar_ia = st.checkbox("Gerar Relatório Tático com IA (Gemini 3.5 Flash Lite)", value=True, help="Envia os resultados matemáticos para um LLM gerar uma narrativa tática esportiva.")
        
        submit_btn = st.form_submit_button("Calcular Modelo", type="primary")
        
    if submit_btn:
        with st.spinner("Treinando modelo, padronizando dados e selecionando variáveis..."):
            df_model = team_df[(team_df['season'] >= sel_anos[0]) & (team_df['season'] <= sel_anos[1])].copy()
            
            cols_to_use = ['3PA', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'PF']
            df_model = df_model.dropna(subset=cols_to_use + ['win']) 
            
            if len(df_model) < 50:
                st.error("Dados insuficientes para rodar o modelo no período selecionado.")
            else:
                X = df_model[cols_to_use]
                y = df_model['win']
                
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                
                def calculate_vif(data):
                    vif = pd.DataFrame()
                    vif["Feature"] = data.columns
                    vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
                    return vif
                    
                features_atuais = list(X_scaled.columns)
                while len(features_atuais) > 0:
                    vif_data = calculate_vif(X_scaled[features_atuais])
                    max_vif = vif_data['VIF'].max()
                    if max_vif > 5.0:
                        feature_to_remove = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
                        features_atuais.remove(feature_to_remove)
                    else:
                        break
                        
                X_scaled = X_scaled[features_atuais]
                
                modelo_valido = True
                while len(X_scaled.columns) > 0:
                    X_with_const = sm.add_constant(X_scaled)
                    try:
                        model = sm.Logit(y, X_with_const).fit(disp=0)
                    except Exception as e:
                        st.error(f"Erro na convergência do modelo: {e}")
                        modelo_valido = False
                        break
                        
                    p_values = model.pvalues.drop('const')
                    max_p = p_values.max()
                    if max_p > 0.05:
                        feature_to_remove = p_values.idxmax()
                        X_scaled = X_scaled.drop(columns=[feature_to_remove])
                    else:
                        break
                
                if not modelo_valido:
                    pass
                elif len(X_scaled.columns) == 0:
                    st.warning("Nenhuma variável foi considerada estatisticamente significativa (p < 0.05) isoladamente para prever vitórias neste período.")
                else:
                    X_final = sm.add_constant(X_scaled)
                    model_final = sm.Logit(y, X_final).fit(disp=0)
                    
                    params = model_final.params.drop('const')
                    odds_ratios = np.exp(params)
                    
                    df_pareto = pd.DataFrame({
                        'Variável': params.index,
                        'Beta': params.values,
                        'Abs_Beta': np.abs(params.values),
                        'Odds_Ratio': odds_ratios.values
                    })
                    df_pareto = df_pareto.sort_values(by='Abs_Beta', ascending=False)
                    
                    df_pareto['Cor'] = df_pareto['Odds_Ratio'].apply(lambda x: NBA_BLUE if x > 1 else NBA_RED)
                    df_pareto['Status'] = df_pareto['Odds_Ratio'].apply(lambda x: 'Aumenta Chance' if x > 1 else 'Diminui Chance')
                    
                    df_pareto['Cumulative'] = df_pareto['Abs_Beta'].cumsum() / df_pareto['Abs_Beta'].sum() * 100
                    
                    fig4 = go.Figure()
                    fig4.add_trace(go.Bar(
                        x=df_pareto['Variável'],
                        y=df_pareto['Odds_Ratio'],
                        marker_color=df_pareto['Cor'],
                        name='Odds Ratio',
                        text=df_pareto['Odds_Ratio'].round(2),
                        textposition='auto',
                        hovertemplate='Variável: %{x}<br>Odds Ratio: %{y:.2f}<br>%{customdata}<extra></extra>',
                        customdata=df_pareto['Status']
                    ))
                    
                    fig4.add_trace(go.Scatter(
                        x=df_pareto['Variável'],
                        y=df_pareto['Cumulative'],
                        mode='lines+markers',
                        name='Importância Acumulada (%)',
                        yaxis='y2',
                        line=dict(color='white', width=2)
                    ))
                    
                    fig4.update_layout(
                        title="O DNA da Vitória (Odds Ratio por Variável)",
                        template="plotly_dark",
                        yaxis=dict(
                            title='Odds Ratio (Razão de Chances)',
                            type='log'
                        ),
                        yaxis2=dict(
                            title='Impacto Acumulado (%)',
                            overlaying='y',
                            side='right',
                            range=[0, 105]
                        ),
                        showlegend=False,
                        hovermode="x unified"
                    )
                    
                    y_pred = (model_final.predict(X_final) >= 0.5).astype(int)
                    acc = accuracy_score(y, y_pred)
                    pseudo_r2 = model_final.prsquared
                    llr_pvalue = model_final.llr_pvalue
                    
                    st.success(f"Modelo finalizado! {len(X_scaled.columns)} variáveis passaram no funil de significância (p < 0.05).")
                    
                    with st.container(border=True):
                        st.plotly_chart(fig4, width="stretch")
                    
                    with st.container(border=True):
                        st.markdown("### Métricas de Ajuste do Modelo Estatístico")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Acurácia das Previsões", f"{acc*100:.1f}%", help="Percentual de partidas previstas corretamente pelo modelo.")
                        m2.metric("Pseudo R² (McFadden)", f"{pseudo_r2:.3f}", help="Qualidade estatística do ajuste (varia de 0 a 1). Valores acima de 0.2 são considerados ótimos para regressão logística.")
                        m3.metric("p-valor do Modelo (LLR)", f"{llr_pvalue:.4e}", help="Se próximo de 0, significa que o modelo possui alto poder preditivo global comparado ao acaso.")

                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y, y_pred)
                        
                        st.markdown("---")
                        st.markdown("### Matriz de Confusão")
                        
                        col_cm1, col_cm2 = st.columns([1.5, 1])
                        with col_cm1:
                            fig_cm = px.imshow(cm, 
                                               text_auto=True, 
                                               color_continuous_scale='Blues',
                                               labels=dict(x="Previsão do Modelo", y="Resultado Real", color="Jogos"),
                                               x=['Derrota (0)', 'Vitória (1)'],
                                               y=['Derrota (0)', 'Vitória (1)'])
                            fig_cm.update_layout(template='plotly_dark', margin=dict(t=30, b=0, l=0, r=0))
                            st.plotly_chart(fig_cm, width="stretch")
                            
                        with col_cm2:
                            st.write(" ")
                            st.write(" ")
                            st.info(f"✅ **Acertos (Verdadeiros):**\n"
                                    f"- Previu Vitória e Ganhou: **{cm[1,1]}** jogos\n"
                                    f"- Previu Derrota e Perdeu: **{cm[0,0]}** jogos\n\n"
                                    f"❌ **Erros (Falsos):**\n"
                                    f"- Previu Vitória, mas Perdeu: **{cm[0,1]}** jogos\n"
                                    f"- Previu Derrota, mas Ganhou: **{cm[1,0]}** jogos")

                    # Chamada ao Gemini
                    if gerar_ia:
                        with st.container(border=True):
                            st.markdown("### 🧠 O Analista de IA")
                            
                            api_key = os.environ.get("GEMINI_API_KEY")
                            
                            if not api_key:
                                try:
                                    api_key = st.secrets["GEMINI_API_KEY"]
                                except:
                                    api_key = None
                            
                            if not api_key:
                                st.error("Chave da API do Gemini não encontrada no arquivo chaves.env nem no secrets do Streamlit.")
                            else:
                                with st.spinner("O Analista está revisando a fita do jogo (interpretando os dados)..."):
                                    try:
                                        client = genai.Client(api_key=api_key)
                                        
                                        contexto_pareto = df_pareto[['Variável', 'Odds_Ratio']].to_string(index=False)
                                        
                                        contexto_cm = (f"Verdadeiros Positivos (Acertou Vitória): {cm[1,1]}\n"
                                                       f"Falsos Positivos (Previu Vitória, deu Derrota): {cm[0,1]}\n"
                                                       f"Verdadeiros Negativos (Acertou Derrota): {cm[0,0]}\n"
                                                       f"Falsos Negativos (Previu Derrota, deu Vitória): {cm[1,0]}")
                                        
                                        contexto_metricas = f"Acurácia: {acc*100:.1f}%\nPseudo R2: {pseudo_r2:.3f}"
                                        
                                        prompt = f"""Você é um renomado analista tático e estatístico da NBA. 
                                        Acabamos de treinar um modelo de Regressão Logística para prever vitórias na liga durante as temporadas de {sel_anos[0]} a {sel_anos[1]}.
                                        
                                        **DADOS OBTIDOS:**
                                        
                                        1. Métricas de Ajuste do Modelo:
                                        {contexto_metricas}
                                        
                                        2. Tabela Completa de Odds Ratios (Impacto de TODAS as variáveis selecionadas):
                                        (Lembre-se: Odds Ratios maiores que 1 aumentam a chance de vitória. Menores que 1 diminuem a chance.)
                                        {contexto_pareto}
                                        
                                        3. Matriz de Confusão Completa:
                                        {contexto_cm}
                                        
                                        Sua tarefa:
                                        Responda à pergunta 'Qual era a receita da vitória nessa época?' de forma breve traduzindo rigorosamente esses dados estatísticos brutos para uma linguagem de basquete acessível.
                                        - Use o formato Markdown nativo (com negritos, bullet points e emojis esportivos).
                                        - Cite explicitamente os números da tabela de Odds Ratios, mas explique o que eles significam na quadra (ex: se 3PA ou OREB forem altos, contextualize com a evolução do jogo).
                                        - Faça uma breve análise sobre a eficácia do modelo (matriz de confusão e acurácia).
                                        - Regra Crítica: NÃO invente informações estatísticas que não estão no contexto fornecido."""
                                        
                                        response = client.models.generate_content(
                                            model='gemini-3.5-flash-lite',
                                            contents=prompt
                                        )
                                        
                                        st.success("Análise tática gerada com sucesso!")
                                        st.markdown(response.text)
                                        
                                    except Exception as e:
                                        st.error(f"Erro ao conectar com a IA do Google: {e}")
