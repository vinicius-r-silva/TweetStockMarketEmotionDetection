# TweetStockMarketEmotionDetection
Membros:    Cesar Antonio Madera Garcés
            Gabriela Barion Vidal
            Vinícius Ribeiro da Silva

Trabalho com o objetivo de identificar emoções em Tweets sobre o mercado financeiro.
Língua da base de dados: Português-BR
Base de dados utilizada: [Dataset](https://www.kaggle.com/datasets/fernandojvdasilva/stock-tweets-ptbr-emotions)

## Dificuldades encontradas no Dataset
O dataset possui algumas singularidades que precisam ser resolvidas. Abaixo estão os problemas encontrados e as soluções possíveis desenvolvidas.
| Problema | Solução |
| --- | --- |
| `tweets_stocks-full` possui os dados do `tweets_stocks` | Gerar um `.csv` novo com dados de teste e treino separados |
| Strings desnecessárias | Criação de um filtro para retirá-las e normalizá-las |
| Multioutputs | Duplicar tweets com mais de 1 saída e treiná-los |
| Valores que não possuem emoções | Retiramos esses dados (1398) |

## Abordagens escolhidas
As abordagens escolhidas foram:

### Probabilidade
Para esse método, utilizamos a frequência de n tuplas aparecerem para cada emoção

### Aprendizado de máquina
