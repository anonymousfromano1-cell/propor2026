Em relação aos datasets, o dataset em seu formato original está intitulado como ssa_holder_original. O mesmo contém
a estrutura original do dataset sobre o qual foi anotado o holder em substituição da categoria. 

O dataset ssa_holder_as_dict foi resultado de uma segunda anotação para facilitar a iteração sobre os dados e suas rotulações, 
pois o holder não possuia localização, detalhe que poderia causar a identificação errônea do holder pois a mesma palavra poderia
aparecer em outro contexto dentro do texto.

Por fim, o arquivo dataset foi utilizado para o treinamento do modelo pois o mesmo possui o mapeamento para BIO tag. Este dataset
inclui o texto original e a BIO tag correspondente ao texto pré-processado.

O código de pré-processamento é responsável pela criação de listas BIO tag de acordo com as anotações da palavra
no dataset. Cada texto possui anotações, e cada anotação possui uma chave, sendo cada chave um elemento da tupla. Então este 
valor é utilizado para mapeamento da BIO tag que será inserida em uma lista de mesmo tamanho do texto original, mas em lugar 
dos termos originais, serão utilizadas as tags. O intervalo da localização é utilizado principalmente para que saibamos o número
de palavras correspondentes ao elemento da tupla, como "Café-da-manhã" ou "Meu filho e eu", que quando "splitados", serão utilizados
para o mapeamento inicial e incluso da BIO tag. Exemplo: [café, da, manhã] -> [B-TARGET, I-TARGET, I-TARGET].

Por fim, o arquivo model.py contém o código que realiza o treinamento do BERTimbau. Os dados são separados para treino e teste.
Para o treinamento, a função compute_metrics realiza a comparação entre dados preditos e verdadeiros. A biblioteca seqeval fornece o relatório de métricas para cada entidade presente nos dados estruturados em BIO TAG. Ou seja, se tivermos [B-HOLDER, B-TARGET, B-NOVODADO], ela irá fornecer métricas a nível de span para holder, target e novodado. Sendo assim, foi utilizado para avaliação a nível de span.

Para medirmos o acerto da polaridade, separamos a polaridade da expressão. Mantendo apenas a polaridade junto ao B e ao I, foi possível realizar a análise F1 a nível de span das polaridades também.
