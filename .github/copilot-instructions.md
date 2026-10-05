# Instruções do projeto

- Projeto Python orientado a notebooks para previsão diária de demanda de múltiplos perecíveis, horizonte inicial de sete dias.
- Use dados sintéticos com semente fixa nos exemplos; não inclua dados de vendas reais ou segredos no repositório.
- Mantenha features disponíveis no momento da previsão. Respeite a ordem temporal em divisões, validação e backtesting.
- Trate vendas observadas durante ruptura como demanda censurada; nunca as interprete silenciosamente como demanda real.
- Compare modelos com baselines e reporte MAE, WAPE e viés por produto e horizonte quando possível. Métricas não substituem os custos operacionais de ruptura e desperdício.
- Documente limitações e não prometa acurácia garantida.
