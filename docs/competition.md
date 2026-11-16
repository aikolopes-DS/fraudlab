# Competição e regras verificadas

Fonte: [regras oficiais do Playground S6E11](https://www.kaggle.com/competitions/playground-series-s6e11/rules), lidas pela árvore de acessibilidade do Brave em 16/11/2026. O texto recuperado foi conservado apenas no diretório local ignorado pelo Git. Este resumo não substitui as regras oficiais.

- Problema: prever a probabilidade de `is_fraud` para cada `id` de teste.
- Métrica: ROC-AUC. Submissão: CSV com `id,is_fraud`.
- Prazo publicado: 30/11/2026 às 23h59 UTC, equivalente a 20h59 em São Paulo.
- Limite: dez submissões por dia; até duas submissões finais.
- Equipe: no máximo três pessoas; uma única conta por participante.
- Dados: licença CC BY 4.0, com os demais termos de acesso e segurança da competição.
- Dados e ferramentas externos: precisam atender aos critérios de acesso e custo previstos nas regras. O protocolo inicial usou somente os arquivos oficiais. A composição selecionada usa previsões públicas licenciadas e versionadas, com [autoria e origem explícitas](../evidence/external-gpt2.json).
- Código público: permitido sob licença aberta compatível. As regras também exigem disponibilizá-lo nos fóruns ou notebooks associados à competição. O [notebook público do projeto](https://www.kaggle.com/code/aikoog/fraudlab-reproducible-neural-fraud-detection) está associado à edição e ligado ao repositório; as versões publicadas foram salvas sem execução.
- Não há compartilhamento privado de código ou dados entre equipes separadas.
- O ranking público durante o evento é provisório. A classificação final depende do conjunto privado e das regras da competição.

Não incluímos os CSVs ou checkpoints no repositório. Cada participante deve obter os dados no Kaggle, com sua própria conta e aceitação das regras.

A sessão do Brave foi confirmada como `aiko-og` (usuário `aikoog`). Os arquivos oficiais foram baixados e submissões foram processadas pela plataforma. Identificadores, scores e posições com data estão em [evidence/public.json](../evidence/public.json); a conferência de bytes do arquivo enviado foi registrada separadamente.
