import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
import glob

st.set_page_config(page_title="Dashboard Análise NBA", page_icon="🏀", layout="wide")

# Estilos Customizados
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
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

st.title("🏀 Estatísticas e Histórico da NBA (1996-2026)")
st.write("Explore tendências temporais, analise perfis de equipes e descubra os maiores destaques individuais de cada temporada.")
st.divider()

try:
    team_df, player_df = load_data()
except FileNotFoundError:
    st.error("Arquivos de dados não encontrados no diretório especificado.")
    st.stop()

tabs = st.tabs([
    "📈 Evolução do Jogo", 
    "🧪 Teste de Hipótese", 
    "⭐ Destaques Individuais", 
    "🏠 Fator Casa vs Visitante"
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
    
    selected_metric = st.selectbox("Selecione o indicador que deseja analisar:", list(metrics.keys()), format_func=lambda x: metrics[x])
    
    fig1 = px.line(season_stats, x='season', y=selected_metric, markers=True, 
                   color_discrete_sequence=['#ff4b4b'])
    fig1.update_layout(
        title=f"Evolução: {metrics[selected_metric]}",
        xaxis_title="Temporada",
        yaxis_title=metrics[selected_metric],
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)

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
    
    st.markdown("### Configuração do Teste")
    col1, col2, col3 = st.columns(3)
    with col1:
        grupo1_name = st.selectbox("Período Base (Grupo 1):", list(decades.keys()), index=0)
    with col2:
        grupo2_name = st.selectbox("Período de Comparação (Grupo 2):", list(decades.keys()), index=3)
    with col3:
        var_teste = st.selectbox("Variável de Análise:", ['PTS', 'PF', '3PM', '3PA', 'FGA'], format_func=lambda x: {
            'PTS': 'Pontos (Ofensividade)',
            'PF': 'Faltas (Fisicalidade)',
            '3PM': 'Cestas de 3 (Eficiência Externa)',
            '3PA': 'Tentativas de 3 (Volume Externo)',
            'FGA': 'Arremessos de Quadra (Ritmo/Pace)'
        }[x])
        
    if st.button("Executar Análise Estatística", type="primary"):
        g1_seasons = decades[grupo1_name]
        g2_seasons = decades[grupo2_name]
        
        df_g1 = team_df[team_df['season'].isin(g1_seasons)].groupby(['season', 'teamid'])[var_teste].mean().reset_index()
        df_g2 = team_df[team_df['season'].isin(g2_seasons)].groupby(['season', 'teamid'])[var_teste].mean().reset_index()
        
        data1 = df_g1[var_teste].dropna()
        data2 = df_g2[var_teste].dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            st.warning("Dados insuficientes para um dos grupos.")
        else:
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric(f"Média - {grupo1_name}", f"{data1.mean():.2f}", f"Desvio Padrão: {data1.std():.2f}", delta_color="off")
            c2.metric(f"Média - {grupo2_name}", f"{data2.mean():.2f}", f"Desvio Padrão: {data2.std():.2f}", delta_color="off")
            
            stat_l, p_levene = stats.levene(data1, data2)
            equal_var = p_levene > 0.05
            stat_t, p_valor = stats.ttest_ind(data1, data2, equal_var=equal_var)
            
            st.markdown("### Conclusão do Teste")
            if p_valor < 0.05:
                st.success(f"**Diferença Significativa Detectada (p-valor = {p_valor:.4f}).** Os períodos apresentam perfis estatisticamente distintos para esta métrica.")
            else:
                st.info(f"**Sem Diferença Significativa (p-valor = {p_valor:.4f}).** Não há evidências suficientes para afirmar que os períodos são diferentes.")
                
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=data1, name=grupo1_name, marker_color='#ff4b4b'))
            fig_box.add_trace(go.Box(y=data2, name=grupo2_name, marker_color='#4b4bff'))
            fig_box.update_layout(title=f"Distribuição de {var_teste}", yaxis_title=var_teste, template="plotly_dark")
            st.plotly_chart(fig_box, use_container_width=True)

# ----------------- TAB 3: Melhores Jogadores -----------------
with tabs[2]:
    st.header("Recordistas e Destaques por Temporada")
    st.write("Descubra os líderes em estatísticas. O valor destacado compara o topo atual com o topo da temporada anterior.")
    
    anos_disponiveis = sorted(player_df['season'].unique(), reverse=True)
    
    # Criar um container para o seletor
    with st.container():
        ano_selecionado = st.selectbox("Selecione a Temporada:", anos_disponiveis, index=0)
    
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
                with st.container():
                    st.metric(
                        label=title, 
                        value=f"{val_atual:.1f} ({nome_atual})", 
                        delta=f"{delta_val:.1f} vs Ano Anterior" if delta_val is not None else "Sem dados"
                    )
                st.write("")
                st.write("")

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
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=yearly_perf['season'], y=yearly_perf['vit_casa'], mode='lines', name='Em Casa', line=dict(color='#00d4ff', width=3)))
        fig2.add_trace(go.Scatter(x=yearly_perf['season'], y=yearly_perf['vit_fora'], mode='lines', name='Visitante', line=dict(color='#ff4b4b', width=3)))
        fig2.update_layout(title="Volume Médio de Vitórias", xaxis_title="Temporada", yaxis_title="Vitórias", template="plotly_dark", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_chart2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=yearly_perf['season'], y=yearly_perf['razao_casa'], mode='lines', name='Razão Casa', line=dict(color='#00d4ff', width=3)))
        fig3.add_trace(go.Scatter(x=yearly_perf['season'], y=yearly_perf['razao_fora'], mode='lines', name='Razão Visitante', line=dict(color='#ff4b4b', width=3)))
        fig3.update_layout(title="Proporção Vitória/Derrota (%)", xaxis_title="Temporada", yaxis_title="Razão (%)", template="plotly_dark", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Como interpretar a Proporção Vitória/Derrota?"):
        st.write("A razão é calculada pela fórmula `((Vitórias / Derrotas) - 1) * 100`.")
        st.write("Valores positivos indicam superioridade de vitórias. Por exemplo, 20% significa que o time vence 20% a mais do que perde. Valores negativos indicam o oposto.")
