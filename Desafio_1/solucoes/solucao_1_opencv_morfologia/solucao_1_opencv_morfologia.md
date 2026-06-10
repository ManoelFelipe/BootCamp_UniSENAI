# Solução 1 - OpenCV com Morfologia e Contornos

Esta é a solução técnica principal e recomendada para o Desafio 1. Por se basear em métodos matemáticos clássicos de processamento de imagens, ela é explicável, determinística e não exige grandes conjuntos de dados para treinamento.

---

## 💡 Ideia Central

Segmentar e contar os parafusos a partir de técnicas clássicas de visão computacional: CLAHE para normalização de iluminação local, filtros Canny para detecção de bordas, operações morfológicas para limpeza de ruídos e filtragem geométrica baseada em área, circularidade e relação de aspecto dos contornos encontrados.

---

## ⚙️ Parâmetros do Script CLI e Dash

O script principal fornece os seguintes argumentos de calibração fina:

*   `--mask-mode`: Define a estratégia de segmentação. As opções aceitas são:
    *   `edges` (Padrão): Utiliza o detector Canny para extrair contornos limpos. **Este modo produz os resultados exatos relatados na tabela do relatório técnico.**
    *   `auto`: Testa dinamicamente vários limiares (`otsu_dark`, `otsu_light`, `adaptive`, `edges`) e escolhe o que obtiver a maior pontuação heurística.
    *   `manual`: Utiliza um limiar fixo baseado em intensidade de cinza (CLAHE + Threshold).
    *   `otsu_dark`, `otsu_light`, `adaptive`: Permite forçar uma máscara específica diretamente.
*   `--min-area` / `--max-area`: Define os limites de área em pixels para aceitar um candidato como parafuso (padrão: `80` a `20000`).
*   `--expected-area`: Área mediana esperada de um parafuso (ajuda na pontuação heurística da máscara automática, padrão: `1600`).
*   `--corner-margin`: Descarta componentes grudados em cantos adjacentes da imagem (padrão: `4.0` pixels), filtrando ruídos das bordas e artefatos de fundo.
*   `--manual-threshold`: Valor de corte em escala de cinza usado no modo de segmentação manual (padrão: `90.0`).

---

## 📊 Resultados da Execução Padrão (Edges)

Ao rodar a solução em lote com a configuração padrão (`--mask-mode edges`), os resultados obtidos são:

| Imagem | Contagem Automática | Método Aplicado | Foco do Cenário |
| :--- | :---: | :---: | :--- |
| `img1.jpg` | 8 | edges | Dois parafusos longos paralelos. |
| `img2.jpg` | 2 | edges | Parafusos curtos e paralelos. |
| `img3.jpg` | 4 | edges | Vários parafusos espalhados. |
| `img4.jpg` | 2 | edges | Parafuso isolado centralizado. |
| `img5.jpg` | 12 | edges | Caso complexo: parafusos sobrepostos e reflexos. |

> [!NOTE]
> Como não há rótulos de contagem oficiais do cliente, os erros de contagem automática em imagens muito difíceis (como `img5.jpg`) são gerenciados via interface Dash através do campo **"Correção Manual"**, fornecendo um fluxo de trabalho assistido para o operador de picking.

---

## 📱 Portabilidade e Smartphone

A solução é ideal para dispositivos móveis devido ao seu baixo consumo computacional:
*   Não requer placas de vídeo (GPU) ou conexões lentas de rede para rodar inferência de redes profundas.
*   A interface Dash foi atualizada para escutar no host `0.0.0.0`. O time de *picking* pode testar o app em seus celulares acessando o IP do computador local pelo Wi-Fi.

---

## 📓 Notebook de Estudo e Análise
*   **Arquivo:** [`analise_interativa_opencv_morfologia.ipynb`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/analise_interativa_opencv_morfologia.ipynb)
    *   Este notebook interativo ajuda no entendimento de cada etapa intermediária (CLAHE, Canny, dilatação, filtragem geométrica) em cada uma das cinco imagens.
