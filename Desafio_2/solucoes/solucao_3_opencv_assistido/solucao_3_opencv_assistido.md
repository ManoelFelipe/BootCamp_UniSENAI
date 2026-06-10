# Solução 3 - OpenCV assistido por métricas

## Ideia central

Criar um baseline clássico para destacar regiões suspeitas de trinca usando filtros, bordas e morfologia, validando contra as máscaras derivadas dos labels.

## Tipo de problema

Abordagem híbrida: segmentação heurística e inspeção assistida.

## Técnicas ou modelos

- CLAHE.
- Black-hat morfológico para realçar linhas escuras.
- Canny.
- Threshold Otsu/adaptativo.
- Remoção de pequenos componentes.
- Comparação com máscaras dos labels.

## Metodologia

1. Pre-processar a imagem em cinza.
2. Realçar linhas finas e escuras.
3. Gerar máscara candidata.
4. Limpar ruído com morfologia.
5. Sobrepor a máscara na imagem original.
6. Calcular IoU/Dice quando houver label.

## Como os rótulos são utilizados

Os polígonos são convertidos em máscara binária apenas para avaliação e calibração. O método em si não aprende pesos.

## Como a posição da trinca é identificada

A posição é dada pela máscara candidata e pelos contornos extraídos.

## Validação e métricas

- IoU.
- Dice.
- Precision.
- Recall.
- Análise visual dos casos com baixo desempenho.

## Robustez

A robustez depende de calibração por iluminação e textura. Pode ser combinado com revisão humana e usado como fallback quando não houver modelo treinado.

## Vantagens

- Explicável.
- Leve para dispositivo móvel.
- Bom baseline para relatório.

## Limitações

- Menor precisão esperada que modelos supervisionados.
- Sensível à textura de parede e sombras.
- Difícil generalizar com parâmetros fixos.

## Avaliação

- Criatividade: 3/5.
- Viabilidade: 4/5.
- Complexidade: média.
- Potencial em dispositivo de baixo processamento: alto.

## Como apresentar no relatório

Use como baseline analítico. A comparação com YOLO ajuda a justificar o ganho de usar dados rotulados.
