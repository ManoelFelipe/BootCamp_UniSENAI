# Solução 2 - U-Net leve com máscaras binárias

## Ideia central

Converter os polígonos YOLO em máscaras binárias e treinar uma rede de segmentação semântica leve para marcar pixels de fissura.

## Tipo de problema

Segmentação semântica binária: parede normal versus fissura.

## Técnicas ou modelos

- Conversão de polígonos para máscaras.
- U-Net pequena.
- Perda BCE + Dice.
- Métricas Dice, IoU, precision e recall.
- Possível quantização futura.

## Metodologia

1. Ler labels YOLO.
2. Converter coordenadas normalizadas para pixels.
3. Preencher polígonos em máscaras binárias.
4. Treinar U-Net em imagens redimensionadas.
5. Avaliar Dice/IoU.
6. Gerar mapas de calor e overlay.

## Como os rótulos são utilizados

Cada polígono vira uma região positiva na máscara. Isso permite treinamento pixel a pixel.

## Como a posição da trinca é identificada

A saída da rede é um mapa de probabilidade. Pixels acima do threshold formam a máscara final da fissura.

## Validação e métricas

- Dice coefficient.
- IoU.
- Recall para reduzir risco de deixar fissura passar.
- Precision para evitar excesso de alarmes.

## Robustez

Usar augmentations fotométricas e geométricas: brilho, contraste, blur, rotação pequena, crop e resize.

## Vantagens

- Muito interpretável visualmente.
- Boa para fissuras finas e alongadas.
- Permite mapas de calor.

## Limitações

- Mais trabalhosa que YOLO.
- Precisa de GPU ou treino mais demorado.
- Exportação mobile exige etapa adicional.

## Avaliação

- Criatividade: 4/5.
- Viabilidade: 4/5.
- Complexidade: alta.
- Potencial em dispositivo de baixo processamento: médio-alto com rede pequena e quantização.

## Como apresentar no relatório

Apresente como alternativa tecnicamente forte para segmentação fina. Use como comparação ou evolução caso a entrega principal seja YOLO.
