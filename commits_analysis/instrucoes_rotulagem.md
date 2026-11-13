# Instruções de rotulagem manual (bug fix em commits Terraform)

Esta rotulagem substitui a anterior (`sample_<nome>.csv`). Vamos rotular os commits nas mesmas três
condições em que a LLM os classifica, para comparar humano e LLM em cada uma.

## Suas planilhas

Na pasta `rotulagem/<seu_nome>/` há três planilhas com os mesmos ~100 commits, cada uma numa ordem
aleatória diferente:

| Rodada | Arquivo | O que você vê |
|---|---|---|
| 1 | `<nome>_rodada1_diff.xlsx` | só o diff (arquivos alterados e mudanças) |
| 2 | `<nome>_rodada2_mensagem.xlsx` | só a mensagem do commit |
| 3 | `<nome>_rodada3_mensagem_diff.xlsx` | mensagem + diff |

## Visualizador (recomendado)

Abra `rotulagem/visualizador.html` no navegador e arraste a planilha da rodada para a página. Ele mostra a
mensagem e o diff colorido por arquivo, e você rotula com os botões ou com as teclas `B` (BUG_FIX) e `N`
(NON_BUG_FIX). No Chrome ou Edge, ele grava os rótulos na própria planilha: clique em **Salvar na planilha**
(ou `Ctrl+S`) uma vez para dar permissão e, a partir daí, ele salva sozinho a cada rótulo; mantenha a
planilha fechada no Excel enquanto isso. Nos outros navegadores (Firefox, Safari), **Baixar planilha** baixa
uma cópia com o mesmo nome, que é a que você devolve. O progresso também fica guardado no navegador.
Precisa de internet só para abrir; a planilha não sai do seu computador.

**Dica para o Firefox:** em Configurações → Geral → Arquivos e aplicativos, marque "Sempre perguntar onde
salvar arquivos". Assim o **Baixar planilha** abre a janela de salvar com o nome original, e você pode
escolher a pasta da planilha e substituir o arquivo, em vez de juntar cópias em Downloads.

Se preferir rotular direto no Excel: alguns itens muito longos não cabem numa célula e continuam nas
colunas `evidencia_2`, `evidencia_3`, ... à direita.

## Regras

1. **Faça as rodadas em ordem (1, 2, 3) e termine uma antes de abrir a próxima.** Não volte a uma
   rodada já entregue para mudar respostas.
2. **Use só o que está na planilha.** Não abra o commit no GitHub, não pesquise o repositório, a issue
   ou o PR. A LLM não vê nada disso, e a comparação só vale se você também não vir.
3. Se reconhecer um commit de uma rodada anterior, rotule mesmo assim com base apenas no que a rodada
   atual mostra.
4. Quando o diff é muito grande, ele aparece cortado (`[...]`), exatamente como a LLM recebe. Decida
   com o que estiver visível.
5. Em cada linha, escolha na coluna `rotulo` **BUG_FIX** ou **NON_BUG_FIX** (lista suspensa). Não deixe
   em branco: na dúvida, escolha o mais provável e explique em `obs`.
6. Use `obs` para dúvidas, evidências conflitantes ou falta de informação (ex.: "só diff, não dá para
   saber se o valor antigo estava errado").

## Definição (a mesma que a LLM recebe)

**BUG_FIX**: o propósito *principal* do commit é corrigir um comportamento, configuração ou
implementação existente que estava errado, inválido, inseguro ou diferente do pretendido. Ele restaura
algo que já deveria funcionar. Exemplos: corrigir valor ou atributo errado de um recurso, referência
ou dependência inválida, lógica errada de `count`/`for_each`/condicional, regressão, configuração que
faz o `plan`/`apply`/validação falhar, defeito de segurança, default incorreto ou configuração ausente
que causava comportamento indesejado.

**NON_BUG_FIX**: o propósito principal não é corrigir um defeito existente. Exemplos: nova
funcionalidade, novo recurso/módulo/variável/output, refatoração, limpeza, formatação, renomeação,
documentação ou comentários, exemplos, atualização de versão ou dependência sem evidência de corrigir
um defeito específico, só testes, arquivos gerados, resolução de merge, revert simples, migração ou
reestruturação que não repara um defeito.

**Regras de decisão:**

- Palavras como "fix", "bug", "error", "issue" na mensagem **não bastam** para ser BUG_FIX; e um commit
  pode ser BUG_FIX sem nenhuma dessas palavras.
- A mensagem indica a intenção declarada; o diff mostra o que mudou de fato. Se discordarem, priorize o
  que a mudança faz.
- Diferencie *consertar comportamento existente* (BUG_FIX) de *introduzir comportamento novo*
  (NON_BUG_FIX).
- Tamanho não decide: uma mudança pequena pode ser BUG_FIX e uma grande pode não ser.
- Não presuma que é bug fix só porque o commit foi minerado por palavras-chave.
- **Na dúvida, sem evidência de que um defeito existente foi corrigido, escolha NON_BUG_FIX** e anote em
  `obs`.

## Entrega

Devolva cada planilha preenchida assim que terminar a rodada, sem renomear o arquivo.
