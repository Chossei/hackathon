# Análise Estatística da NBA (1996-2026)

Este projeto contém um dashboard interativo desenvolvido em Python utilizando a biblioteca Streamlit. O objetivo principal é fornecer uma análise profunda do cenário da NBA nas últimas três décadas, explorando tendências de jogo, perfis de equipes, recordes individuais e a influência de jogar em casa.

## Funcionalidades do Dashboard

O dashboard está dividido em quatro seções principais:

1. **Evolução do Jogo:** Gráficos de série temporal interativos mostrando como métricas cruciais (pontos, aproveitamento de arremessos, faltas) evoluíram ano após ano.
2. **Teste de Hipótese:** Uma ferramenta estatística que permite selecionar duas décadas distintas e rodar um Teste T de Student, avaliando matematicamente se o estilo de jogo (ofensividade, volume de 3 pontos, fisicalidade) realmente mudou.
3. **Destaques Individuais:** Classificação automática dos líderes estatísticos de cada temporada. Compara dinamicamente o líder atual com a marca estabelecida no ano anterior, ajudando a visualizar a grandiosidade de cada recorde.
4. **Fator Casa vs Visitante:** Análise histórica sobre o impacto do mando de quadra, calculando médias de vitórias e a proporção de vitórias sobre derrotas para equipes mandantes e visitantes ao longo dos anos.

## Como Executar Localmente

Certifique-se de ter o Python instalado em sua máquina.

1. Instale as bibliotecas necessárias:
   ```bash
   pip install streamlit pandas plotly scipy
   ```

2. Execute o aplicativo:
   ```bash
   streamlit run app.py
   ```

## Fonte de Dados

Os dados são provenientes de registros de "boxscores" oficiais da NBA e divididos em duas bases principais:
* `team_traditional.csv`: Dados agregados por equipe.
* `traditional.csv`: Dados granulares por jogador.
