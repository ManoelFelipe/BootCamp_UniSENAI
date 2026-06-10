# Solução 3 - Dados sintéticos e detector leve

## Ideia central

Gerar rótulos fracos a partir da solução OpenCV, aumentar artificialmente as imagens e preparar uma base em formato YOLO para treinar um detector leve.

## Técnicas utilizadas

- Pseudo-rotulagem por contornos.
- Data augmentation: brilho, contraste e espelhamento.
- Exportação de labels YOLO.
- Treinamento opcional de YOLO nano.
- Roadmap de exportação para ONNX/TFLite.

## Metodologia

1. Detectar caixas candidatas de parafusos nas 5 imagens.
2. Salvar pseudo-rótulos em formato YOLO.
3. Gerar variações controladas das imagens.
4. Dividir em treino e validação.
5. Treinar um detector leve apenas como protótipo.
6. Validar manualmente as predições, pois os rótulos iniciais são fracos.

## Como a contagem é feita

O detector retorna caixas de parafusos. A contagem é o número de detecções acima do threshold definido.

## Como lidar com poucas imagens

Esta solução reconhece a limitação: não há dados suficientes para afirmar generalização. O objetivo é criar um pipeline de evolução para quando a empresa coletar novas imagens.

## Validação

Obrigatória validação manual dos pseudo-rótulos e das predições. O relatório deve separar claramente "protótipo promissor" de "solução pronta para produção".

## Vantagens

- Alta criatividade.
- Mostra maturidade de ciclo de vida de IA.
- Pode evoluir para mobile com YOLO nano exportado.

## Limitações

- Risco alto de overfitting.
- Pseudo-rótulos podem propagar erros.
- Precisa de mais dados reais para ser solução final.

## Avaliação

- Criatividade: 5/5.
- Viabilidade: 3/5.
- Complexidade: alta.
- Potencial em smartphone: médio-alto se exportado.

## Como apresentar no relatório

Apresente como evolução futura ou experimento complementar. O diferencial é explicar por que ela não deve substituir a solução clássica sem nova coleta de dados.

## Código recomendado

Use `solucao_3_dados_sinteticos_detector_leve.py`.
