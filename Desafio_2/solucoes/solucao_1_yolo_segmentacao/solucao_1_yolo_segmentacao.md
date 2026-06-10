# Solução 1 - YOLO26n de Segmentação Leve

Esta é a solução principal recomendada para o Desafio 2. Ela consiste no treinamento supervisionado do modelo leve **YOLO26n-seg** a partir dos polígonos de fissuras fornecidos pelo cliente.

---

## 💡 Ideia Central

Treinar um modelo de segmentação de instâncias em tempo real para detectar fissuras em paredes de concreto, retornando caixas delimitadoras de localização e máscaras de segmentação com o formato exato das rachaduras.

*   **Tipo de problema:** Segmentação de Instâncias (Instance Segmentation).
*   **Classe única:** `0` (fissura).

---

## 🛠️ Técnicas e Arquitetura

*   **Modelo Base:** YOLO26n-seg da Ultralytics (variante Nano, otimizada para baixo custo computacional e execução em dispositivos móveis/borda).
*   **Aumento de Dados (Data Augmentation):** Utilização de transformações nativas (brilho, contraste, escala, rotação e blur leve) para simular diferentes condições de iluminação de campo, sombras e texturas de concreto.
*   **Split do Dataset:** Divisão reprodutível e determinística com seed fixa:
    *   **Treino:** 1164 imagens (75%)
    *   **Validação:** 232 imagens (15%)
    *   **Teste Independente:** 155 imagens (10%)

---

## 📊 Resultados e Métricas Oficiais

O treinamento oficial foi executado por **120 épocas** em GPU NVIDIA GeForce GTX 1650 de 4 GB, com tempo total de **4 horas, 23 minutos e 18 segundos**.

### Métricas de Validação (Fim do Treinamento)
*   **Imagens:** 232 | **Instâncias:** 297
*   **Box mAP50:** 0.773 | **Box mAP50-95:** 0.596
*   **Mask mAP50:** 0.668 | **Mask mAP50-95:** 0.256

### Métricas no Split de Teste Independente
Abaixo estão as métricas finais obtidas no split de teste (dados inéditos para o modelo):

| Métrica | Valor de Teste | Interpretação no Negócio |
| :--- | :---: | :--- |
| **Box Precision** | 0.894 | 89.4% das caixas detectadas como fissuras estão corretas. |
| **Box Recall** | 0.711 | O modelo localiza a região de 71.1% das fissuras reais. |
| **Box mAP50** | 0.821 | Excelente desempenho geral de localização de caixas (IoU=0.5). |
| **Mask Precision** | 0.816 | 81.6% do contorno segmentado está correto. |
| **Mask Recall** | 0.629 | Recupera 62.9% dos pixels de contorno de fissuras existentes. |
| **Mask mAP50** | 0.693 | Desempenho funcional e consistente na precisão da máscara. |
| **Mask mAP50-95** | 0.270 | Indicação de que fissuras muito finas/irregulares são o maior desafio. |
| **Inference Speed** | 14.1 ms | Inferência ultrarrápida, ideal para uso embarcado em celulares. |

---

## 📱 Portabilidade para Smartphone (Edge Computing)

O YOLO26n-seg foi selecionado devido à sua alta portabilidade para smartphones com baixa capacidade de processamento:
1.  **Acesso Local por Wi-Fi:** O app Dash escuta em `0.0.0.0`, permitindo o uso da interface web em smartphones de campo acessando o IP do computador.
2.  **Exportação Nativa:** O modelo pode ser facilmente exportado para **ONNX** ou **TensorFlow Lite (TFLite)** para inferência local offline em aplicativos móveis Android/iOS.

---

## 📓 Notebook de Estudo e Análise
*   **Arquivo:** [`analise_interativa_yolo_segmentacao.ipynb`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/analise_interativa_yolo_segmentacao.ipynb)
    *   Este notebook apresenta o passo a passo interativo de preparação, código de treinamento de referência desativado por padrão, inferência visual em amostras do teste e medição exata do percentual de área comprometida por fissuras.
