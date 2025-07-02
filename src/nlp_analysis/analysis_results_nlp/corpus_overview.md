# Corpus Overview
| Métrica                     | Valor                                      |
|-----------------------------|--------------------------------------------|
| `total_commits`             | 7240                         |
| `total_repos`               | 54                           |
| `date_range`                | 2014-05-24 – 2025-05-30                            |
| `msg_len_mean`              | 49.65                          |
| `msg_len_median`            | 49.0                        |
| `msg_len_p10/p90`           | 19.0/78.0                       |
| `generic_merges`            | 0                        | 

> _Nota: Extraímos somente modificações em arquivos `.tf`, pois o foco são operadores de mutação em Terraform._
## Recomendações de Embedding e Clustering
* **Comparação de Embedding:**
    * **TF-IDF (n-gram 1–3):**
        * *Tempo de Execução Estimado:* [Preencher após teste com script real - TF-IDF]
        * *Esparsidade da Matriz:* [Preencher após teste com script real - TF-IDF, e.g., 98%]
        * *Prós:* Rápido, bom para palavras-chave literais.
        * *Contras:* Não captura semântica profunda, sensível ao tamanho do vocabulário.
    * **SBERT (e.g., `sentence-transformers/all-roberta-large-v1`):**
        * *Tempo de Execução Estimado:* [Preencher após teste com script real - SBERT]
        * *Qualidade Semântica:* Alta, captura nuances de significado.
        * *Custo de Transformação:* Computacionalmente mais intensivo que TF-IDF para gerar embeddings.
        * *Prós:* Melhores representações semânticas, embeddings densos.
        * *Contras:* Mais lento para embeddar, pode exigir mais recursos.

* **Valores Iniciais Sugeridos (ajustar com base na exploração):**
    * **UMAP:**
      | Parâmetro        | Valor Sugerido                                                                 |
      |------------------|--------------------------------------------------------------------------------|
      | `n_neighbors`    | ~15-50 (considerar `total_commits`; < local, > global)                         |
      | `min_dist`       | ~0.0-0.1 (controla densidade do cluster; < para maior densidade)               |
      | `n_components`   | (Depende do objetivo, e.g., 2 para visualização, 5-10 para HDBSCAN)            |
    * **HDBSCAN:**
      | Parâmetro           | Valor Sugerido                                                                    |
      |---------------------|-----------------------------------------------------------------------------------|
      | `min_cluster_size`  | ~max(10, int(np.sqrt(7240))) |
      | `min_samples`       | ~max(5, int(20 / 2)) (ou valor explícito como 10) |


## Diagnóstico de Qualidade de Texto
| Métrica                                   | Percentual   |
|-------------------------------------------|--------------|
| Mensagens com URLs                        | 4.79%   |
| Mensagens com IDs de Issue (`#\d+`)       | 24.17% |
| Mensagens com Emojis                      | 0.28% |
| Mensagens não-inglesas (via langdetect)   | 25.28% |
| Commits "fix:" (Conventional Commits)     | 14.64%    |
| Commits "feat:" (Conventional Commits)    | 2.89%   |
