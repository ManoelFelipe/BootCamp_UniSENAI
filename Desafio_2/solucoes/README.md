# Desafio 2 - Soluções Planejadas para Segmentação de Fissuras

Este diretório contém os scripts de treinamento supervisionado, baselines clássicos, arquiteturas pixel a pixel (U-Net) e a interface demonstrativa interativa do Desafio 2.

---

## 📂 Arquivos Incluídos

| Recurso | Papel no Projeto |
| :--- | :--- |
| [`solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.py`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.py) | **Solução Principal:** Preparação, treinamento de 120 épocas e exportações do YOLO26n-seg. |
| [`solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.md`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.md) | Documentação de apoio do modelo supervisionado. |
| [`solucao_2_unet_leve/solucao_2_unet_leve.py`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/solucao_2_unet_leve/solucao_2_unet_leve.py) | **Alternativa de Pesquisa:** Binarização de polígonos e treinamento de rede U-Net leve em PyTorch. |
| [`solucao_3_opencv_assistido/solucao_3_opencv_assistido.py`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/solucao_3_opencv_assistido/solucao_3_opencv_assistido.py) | **Baseline Clássico:** Detecção clássica de fissuras via Black-Hat e Canny com cálculo comparativo de métricas. |
| [`analise_interativa_yolo_segmentacao.ipynb`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/analise_interativa_yolo_segmentacao.ipynb) | **Notebook de Estudos:** Visualizador interativo do split, visualização de polígonos originais e inferência visual. |
| [`app_dash_desafio_2.py`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_2/solucoes/app_dash_desafio_2.py) | **Aplicação Demonstrativa:** Interface Dash para comparar OpenCV, YOLO e roadmap U-Net. |

---

## 🚀 Como Executar

Ative o ambiente virtual `vc_01` e rode os comandos a partir da raiz do projeto:

### 1. Preparar Split do Dataset
Analisa a consistência dos pares imagem/rótulo e gera a divisão de Treino/Validação/Teste:
```bash
python Desafio_2/solucoes/solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.py --prepare
```

### 2. Rodar Treinamento de Teste (Sanity Check)
Roda 3 épocas para verificar o pipeline (GPU, CUDA e salvamento de pesos):
```bash
python Desafio_2/solucoes/solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.py --train --epochs 3 --batch 4 --imgsz 640
```
*   O modelo treinado oficialmente (120 épocas) encontra-se em: `Desafio_2/solucoes/dataset_yolo_split/runs/fissuras_yolo_seg/weights/best.pt`.

### 3. Executar a Aplicação Dash
```bash
python Desafio_2/solucoes/app_dash_desafio_2.py
```
*   O app Dash escuta no host `0.0.0.0` na porta `8062`.
*   Acesse localmente em: `http://127.0.0.1:8062`
*   Acesse no celular via rede Wi-Fi local em: `http://<IP_DO_COMPUTADOR>:8062`.

---

## 📈 Recomendação de Entrega
Apresente o **YOLO26n-seg** como a solução final devido à disponibilidade de um dataset supervisionado expressivo (1551 pares). Destaque a alta portabilidade do modelo de tamanho Nano para dispositivos de baixo poder de processamento (celulares e tablets de obra) através de exportação para ONNX, atendendo aos critérios extras de avaliação do desafio.
