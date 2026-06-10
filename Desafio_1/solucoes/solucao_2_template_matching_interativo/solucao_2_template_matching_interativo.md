# Solução 2 - Template matching calibrado com revisão humana

## Ideia central

Usar um parafuso de referência como template e procurar padrões semelhantes nas demais imagens. Quando a confiança for baixa, a aplicação pede revisão humana.

## Técnicas utilizadas

- Seleção de ROI do parafuso de referência.
- `cv2.matchTemplate` em múltiplas escalas.
- Supressão de sobreposição por NMS.
- Threshold de confiança.
- Revisão humana para corrigir falsos positivos e falsos negativos.

## Metodologia

1. Selecionar uma imagem de referência.
2. Informar um recorte contendo um parafuso isolado.
3. Rodar template matching nas imagens.
4. Aplicar NMS para não contar o mesmo parafuso várias vezes.
5. Gerar imagem anotada e CSV.
6. Encaminhar imagens com baixa confiança para conferência.

## Como a contagem é feita

Cada match aprovado após NMS equivale a um parafuso. A média dos scores pode ser usada como indicador de confiança da contagem.

## Como lidar com poucas imagens

A solução usa o conhecimento visual de um exemplar real, sem treinamento supervisionado. A revisão humana transforma o sistema em assistente de picking, reduzindo risco de erro operacional.

## Validação

Comparar contagens com anotação manual. Registrar também:

- quantidade de matches aceitos;
- score médio;
- quantidade de correções humanas necessárias.

## Vantagens

- Mais adaptável que contornos puros.
- Boa narrativa de produto: IA assistiva, não caixa-preta.
- Pode rodar em smartphone se o template for pequeno.

## Limitações

- Sofre com rotação grande, oclusão e mudança forte de escala.
- Depende de uma boa amostra de template.
- Pode subcontar parafusos muito sobrepostos.

## Avaliação

- Criatividade: 4/5.
- Viabilidade: 4/5.
- Complexidade: média.
- Potencial em smartphone: alto.

## Como apresentar no relatório

Apresente como solução equilibrada: combina automação com controle de qualidade humano, adequada para processo industrial em que erro de contagem tem custo.

## Código recomendado

Use `solucao_2_template_matching_interativo.py`.
