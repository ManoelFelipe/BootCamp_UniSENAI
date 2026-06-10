# Especificação Técnica e de Negócio — CRISP-DM + SDD

Este documento formaliza as decisões de engenharia de dados, modelagem, avaliação e implantação aplicadas no desenvolvimento dos dois desafios práticos de Visão Computacional. Ele segue rigorosamente as diretrizes e boas práticas do manual de referência [documento_mestre_crispdm_sdd_spec_kit-1.md](file:///d:/cursos/BootCamp/Desafio/Desafio/doc/documento_mestre_crispdm_sdd_spec_kit-1.md).

---

## 8. Fase 0 — Constitution (Princípios do Projeto)

Para assegurar a confiabilidade, a reprodutibilidade e o valor das soluções entregues, o projeto orienta-se pelos seguintes princípios:
1. **Negócio antes de Modelo:** Toda escolha algorítmica deve ser justificada pela viabilidade operacional e valor de negócio (ex: contagem rápida offline para picking vs. segmentação de trincas na construção).
2. **Dados antes de Algoritmo:** Avaliar as limitações do volume e qualidade dos dados disponíveis antes de definir a sofisticação da modelagem.
3. **Baseline Obrigatório:** Nenhuma IA ou algoritmo de modelagem é adotado sem antes estabelecer uma linha de base estatística ou clássica para comparação de desempenho.
4. **Alinhamento de Métricas:** A performance técnica (precisão, IoU, recall) deve refletir de forma direta um impacto na decisão de negócios e na agilidade do operador.
5. **Rastreabilidade e Reprodutibilidade:** Versionamento estrito do ambiente de execução [env_vc_01.yml](file:///d:/cursos/BootCamp/Desafio/Desafio/env/env_vc_01.yml), sementes randômicas e documentação das parametrizações.

---

## 9. Desafio 1 — Contagem de Parafusos (OpenCV + Morfologia)

### 9.1 Business Understanding
*   **Problema de Negócio:** Erros manuais e fadiga na triagem/separação (picking) de kits de fixadores metálicos na fábrica, atrasando a expedição.
*   **Decisão Apoiada:** Identificar imediatamente se a quantidade de parafusos exibidos na esteira/caixa bate com a ordem de serviço.
*   **Ação Operacional:** Sinalizar contagens incorretas na tela do picking para correção instantânea pelo operador antes do fechamento do lote.
*   **Métrica de Negócio:** Tempo médio de validação do lote e taxa de kits despachados com faltas.
*   **Métrica Técnica:** Acurácia de contagem por imagem e sensibilidade a parafusos em contato ou sobrepostos.

### 9.2 Data Understanding
*   **Fontes de Dados:** Lote pequeno contendo 5 imagens originais (`img1.jpg` a `img5.jpg`) salvas em [Desafio_1/data/images/](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/data/images/).
*   **Limitações Críticas:** 
    *   Falta de anotações oficiais (bounding boxes) inviabiliza abordagens puramente supervisionadas no primeiro dia.
    *   Presença de reflexos metálicos intensos, parafusos encostados uns nos outros (sobreposição parcial) e variação de iluminação do fundo.

### 9.3 Data Preparation
*   **Pipeline de Pré-processamento:**
    1.  **Escala de cinza:** Simplificação de canais de cor.
    2.  **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Melhora o contraste local sem estourar reflexos em regiões planas.
    3.  **Filtro Mediana/Gaussiano:** Suavização para remoção de ruídos de textura superficial.
*   **Segmentação:** Aplicação do detector de bordas Canny (`edges`) para capturar o contorno externo independente das variações cromáticas internas do metal.

### 9.4 Modeling & Baseline
*   **Baseline Clássico (Solução Oficial):** Executado em [solucao_1_opencv_morfologia.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/solucao_1_opencv_morfologia/solucao_1_opencv_morfologia.py). Utiliza filtragem geométrica dos contornos baseada em intervalos de área, circularidade e proporção largura/altura.
*   **Modo de Máscara Edges (Padrão):** Oferece contagens perfeitamente reprodutíveis e estáveis para o lote:
    *   `img1.jpg`: 8 parafusos (Correto)
    *   `img2.jpg`: 2 parafusos (Correto)
    *   `img3.jpg`: 4 parafusos (Correto)
    *   `img4.jpg`: 2 parafusos (Correto)
    *   `img5.jpg`: 12 parafusos (Estimativa assistida, caso de alta oclusão)
*   **Roadmap de Evolução:**
    *   *Template Matching:* Implementado em [solucao_2_template_matching_interativo.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/solucao_2_template_matching_interativo/solucao_2_template_matching_interativo.py) como baseline alternativo por correlação cruzada normalizada multi-escala.
    *   *Dados Sintéticos (Evolução):* Script [solucao_3_dados_sinteticos_detector_leve.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/solucao_3_dados_sinteticos_detector_leve/solucao_3_dados_sinteticos_detector_leve.py) para gerar pseudo-labels via visão clássica e expandir o dataset com aumentação sintética (brilho, espelhamentos) para futuro treino de um YOLOv8n leve.

### 9.5 Evaluation
*   **Filtros Geométricos Contornos:**
    *   `min_area`: 80.0 e `max_area`: 20000.0 (pixels) para descartar pequenas rebarbas ou poeira.
    *   `min_circularity`: 0.02 e `max_aspect_ratio`: 8.0 para filtrar traços retos que representem sombras ou margens.
    *   `corner_margin`: 4.0 para ignorar artefatos colados nas bordas.
*   **Critério de Aceite (Negócio):** Diante da impossibilidade matemática de prever 100% de oclusão apenas com visão clássica, o operador pode editar e retificar o resultado da contagem diretamente pela tela de revisão manual em tempo de execução via [app_dash_desafio_1.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/app_dash_desafio_1.py).

### 9.6 Deployment
*   **Entrega:** Interface Web Dash escutando no host local e móvel `0.0.0.0:8061`. O processamento roda de forma instantânea em milissegundos sem necessidade de GPU, viabilizando o uso do app em dispositivos móveis conectados na rede Wi-Fi local do picking.

---

## 10. Desafio 2 — Detecção de Fissuras em Concreto (Deep Learning)

### 10.1 Business Understanding
*   **Problema de Negócio:** Necessidade de inspeção de integridade estrutural contínua e automatizada em grandes obras e vias públicas. Inspeções humanas manuais são demoradas, caras e propensas a falhas subjetivas.
*   **Decisão Apoiada:** Identificar áreas críticas com risco de trincas ativas, medindo a extensão afetada para ordenar manutenções preventivas.
*   **Ação Operacional:** Classificar o grau de severidade do trecho inspecionado e registrar as coordenadas geográficas/imagens no relatório de avaria.
*   **Métrica de Negócio:** Custo médio de inspeção por km² e tempo de resposta para reparo estrutural crítico.
*   **Métrica Técnica:** IoU (Intersection over Union) e coeficiente Dice no mapeamento pixel-a-pixel da fissura.

### 10.2 Data Understanding
*   **Fontes de Dados:** Dataset estruturado de fissuras contendo 1551 pares de imagens e arquivos de rótulo em [Desafio_2/data/](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/data/). *(Nota: Devido ao limite de tamanho de 10 MB para upload de anexos na plataforma, os arquivos contidos em `Desafio_2/data/images` e `Desafio_2/data/labels` foram esvaziados no arquivo `.zip` final enviado, podendo ser reinseridos livremente para re-executar os scripts).*
*   **Formato de Rótulos:** Arquivos `.txt` no padrão YOLO segmentação: cada linha inicia com ID da classe `0` (fissura) seguido pelas coordenadas normalizadas `x1 y1 x2 y2 ...` que desenham o polígono do contorno da rachadura.

### 10.3 Data Preparation
*   **Processamento do Split:** Implementado em [solucao_1_yolo_segmentacao.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.py) via semente reprodutível `42`. Separa as amostras em:
    *   **Treino:** 1164 imagens (75%)
    *   **Validação:** 232 imagens (15%)
    *   **Teste:** 155 imagens (10%)
*   **Binarização para U-Net:** Rotina em [solucao_2_unet_leve.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/solucao_2_unet_leve/solucao_2_unet_leve.py) converte os polígonos YOLO em imagens binárias de máscara (preto e branco) no tamanho `384x384` para alimentar o treinamento supervisionado pixel-a-pixel.

### 10.4 Modeling
*   **Baseline Clássico (OpenCV Assistido):** Desenvolvido em [solucao_3_opencv_assistido.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/solucao_3_opencv_assistido/solucao_3_opencv_assistido.py). Combina CLAHE, Filtro Black-Hat morfológico e Canny para isolar linhas escuras de fissuras sem treinamento prévio. Serve como linha de base de interpretabilidade matemática.
*   **Modelo de Produção (YOLO26n-seg):** Rede profunda treinada ao longo de 120 épocas utilizando pesos pré-treinados [yolo26n-seg.pt](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/modelos/yolo26n-seg.pt) como extrator de features inicial.
*   **Roadmap de Evolução (U-Net Leve):** Arquitetura codificador-decodificador leve PyTorch configurada para segmentar rachaduras finas de forma direta.

### 10.5 Evaluation
O conjunto de testes independente foi avaliado comparando o baseline clássico e o modelo supervisionado YOLO:

| Modelo | Precision (Mask) | Recall (Mask) | mAP50 (Mask) | Foco do Cenário |
| :--- | :---: | :---: | :---: | :--- |
| **OpenCV Assistido (Baseline)** | 0.063 | 0.624 | - | Processamento puramente matemático sem IA. Alta sensibilidade a ruídos e poeira do concreto. |
| **YOLO26n-seg (Produção)** | **0,704** | **0,667** | **0,724** | Modelo final recomendado para uso industrial. Excelente isolamento de texturas irregulares. |

*   **Resultados de Pesos:** Os pesos gerados no treinamento de alta fidelidade encontram-se salvos sob o diretório [dataset_yolo_split/runs/fissuras_yolo_seg/weights/best.pt](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/dataset_yolo_split/runs/fissuras_yolo_seg/weights/best.pt).
*   **Cálculo da Área Afetada:** O modelo estima a porcentagem da estrutura afetada contando os pixels rotulados na máscara de predição sobre o total da imagem, exibido em tempo real no dashboard.

### 10.6 Deployment
*   **Entrega:** Interface interativa Dash em [app_dash_desafio_2.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/app_dash_desafio_2.py) escutando em `0.0.0.0:8062`. A inferência YOLO roda de forma fluida em tempo de execução.
*   **Exportação para Dispositivos Móveis:** A arquitetura YOLO nano é otimizada para sofrer conversão direta para ONNX ou TensorFlow Lite, permitindo o empacotamento em aplicações nativas de smartphones para engenheiros em canteiro de obras.

---

## 11. Contratos de Dados (Data Contracts)

A conformidade e o formato de entrada para execução dos pipelines de visão computacional seguem os seguintes esquemas de validação:

### 11.1 Contrato do Desafio 1 (Parafusos)
```yaml
dataset: screw_counting_dataset
description: Conjunto de imagens para contagem automatizada de fixadores.
frequency: ad-hoc
schema:
  input_directory: Desafio_1/data/images
  allowed_formats: [.jpg, .jpeg, .png, .bmp, .webp]
  image_dimensions:
    width: 640
    height: 640
    channels: 3 (RGB)
quality_rules:
  - no_corrupted_images: "Todas as imagens de entrada devem ser lidas com sucesso pelo OpenCV (cv2.imread)."
  - file_count_limit: "O lote experimental atual deve conter no mínimo 1 e no máximo 5 imagens."
```

### 11.2 Contrato do Desafio 2 (Fissuras)
```yaml
dataset: concrete_crack_segmentation
description: Imagens estruturadas de rachaduras com marcações em formato polígono YOLO.
frequency: batch_split
schema:
  images_directory: Desafio_2/data/images
  labels_directory: Desafio_2/data/labels
  allowed_formats: [.jpg, .jpeg, .png]
  polygon_annotation_format:
    line_structure: "<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>"
    class_id: 0 (fissura)
    coordinate_ranges: [0.0, 1.0] (valores normalizados pelo tamanho da imagem)
quality_rules:
  - paired_files: "Para cada imagem existente em images/, deve haver um arquivo .txt de label correspondente em labels/."
  - minimum_polygon_points: "Os polígonos de segmentação YOLO devem conter no mínimo 6 valores (3 pontos x,y) para formar uma área."
```

---

## 12. Fichas de Modelo (Model Cards)

Abaixo está documentada a especificação operacional do modelo supervisionado final de produção:

### 12.1 Ficha do YOLO26n-seg
*   **Nome do Modelo:** YOLO26n-seg (fissuras_yolo_seg)
*   **Versão:** 1.0.0
*   **Data de Treinamento:** 31/05/2026
*   **Responsável:** Manoel Furtado
*   **Objetivo Principal:** Segmentação de fissuras e trincas estruturais em superfícies de concreto por contornos de instâncias.
*   **Arquitetura:** Ultralytics YOLOv8 Nano Segmentação (peso base inicial: `yolo26n-seg.pt`).
*   **Dados de Treino:** 1164 imagens e labels correspondentes de rachaduras tabuladas de alta resolução.
*   **Dados de Validação:** 232 imagens usadas para validação durante o treinamento em lote.
*   **Hiperparâmetros de Treino:**
    *   `epochs`: 120
    *   `batch`: 4
    *   `imgsz`: 640
    *   `device`: GPU (CUDA GeForce GTX 1650)
*   **Uso Recomendado:** Inspeção visual de segurança em muros, vigas e lajes expostas.
*   **Uso Não Recomendado:** Diagnóstico interno microscópico de porosidade do cimento ou análise termográfica.
*   **Limitações Conhecidas:** Regiões com sombras de galhos ou marcas de pichações podem ocasionalmente gerar falsos positivos de baixa confiança.

---

## 13. Matriz de Rastreabilidade

A matriz de rastreabilidade mapeia de que forma cada requisito de negócio é amparado e validado por componentes e testes de código no projeto:

| ID | Objetivo de Negócio | Hipótese Técnica | Entrada de Dados | Script de Código | Métrica Técnica | Critério de Aceite | Teste de Validação |
|---|---|---|---|---|---|---|---|
| **REQ-01** | Otimização do tempo de picking de parafusos. | Processamento de bordas Canny isola formato helicoidal de fixadores. | Imagens em [Desafio_1/data/images/](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/data/images/) | [solucao_1_opencv_morfologia.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/solucao_1_opencv_morfologia/solucao_1_opencv_morfologia.py) | Acurácia de contagem absoluta. | Obter a contagem exata no modo padrão `edges` em 4 das 5 imagens teste. | Execução direta CLI salvando o relatório CSV na pasta de resultados locais. |
| **REQ-02** | Controles interativos em campo por operadores de expedição. | Host em IP `0.0.0.0` viabiliza conexões remotas via Wi-Fi comum no armazém. | Upload em lote ou tempo real. | [app_dash_desafio_1.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/app_dash_desafio_1.py) | Tempo de processamento visual (latência). | Exibição e atualização de gráficos em menos de 1 segundo por upload. | Teste de fumaça subindo o Flask do Dash e simulando concorrência de portas. |
| **REQ-03** | Mapeamento automático de fissuras em obras estruturais. | Segmentação profunda com redes YOLOv8n segmenta contornos curvos finos. | Dataset em [Desafio_2/data/](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/data/) | [solucao_1_yolo_segmentacao.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.py) | Mask Precision e mAP50. | Obter mAP50 superior a 0,70 no conjunto de validação independente. | Avaliação e logs gerados ao fim das 120 épocas de treinamento. |
| **REQ-04** | Comparabilidade analítica explicável para auditoria de trincas. | Filtros Black-Hat clássicos isolam fendas lineares escuras sem inferência de IA. | Imagens cinzentas de concreto. | [solucao_3_opencv_assistido.py](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/solucao_3_opencv_assistido/solucao_3_opencv_assistido.py) | Dice Coefficient e IoU. | Gerar máscaras segmentadas binárias para comparação direta com labels verdadeiros. | Execução de testes de métricas gerando a planilha `.csv` de scores de teste. |
