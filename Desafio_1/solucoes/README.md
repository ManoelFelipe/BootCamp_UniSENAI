# Desafio 1 - Soluções Planejadas para Contagem de Parafusos

Este diretório contém os scripts de processamento de imagem, interface de usuário e análise para o problema de contagem de parafusos da metal-mecânica.

---

## 📂 Arquivos Incluídos

| Recurso | Papel no Projeto |
| :--- | :--- |
| [`solucao_1_opencv_morfologia/solucao_1_opencv_morfologia.py`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/solucao_1_opencv_morfologia/solucao_1_opencv_morfologia.py) | **Solução Principal:** Algoritmo clássico de segmentação de contornos e morfologia. |
| [`solucao_1_opencv_morfologia/solucao_1_opencv_morfologia.md`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/solucao_1_opencv_morfologia/solucao_1_opencv_morfologia.md) | Documentação de apoio do algoritmo clássico. |
| [`solucao_2_template_matching_interativo/solucao_2_template_matching_interativo.py`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/solucao_2_template_matching_interativo/solucao_2_template_matching_interativo.py) | **Baseline Alternativo:** Busca por similaridade geométrica usando recortes. |
| [`solucao_3_dados_sinteticos_detector_leve/solucao_3_dados_sinteticos_detector_leve.py`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/solucao_3_dados_sinteticos_detector_leve/solucao_3_dados_sinteticos_detector_leve.py) | **Roadmap de Evolução:** Pseudo-rotulação e aumentações sintéticas para YOLO leve. |
| [`analise_interativa_opencv_morfologia.ipynb`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/analise_interativa_opencv_morfologia.ipynb) | **Notebook de Estudos:** Visualizador passo a passo do pipeline e calibração de limiares. |
| [`app_dash_desafio_1.py`](file:///d:/cursos/BootCamp/Desafio/Desafio/Desafio_1/solucoes/app_dash_desafio_1.py) | **Aplicação Demonstrativa:** Interface Dash para calibrar parâmetros, rodar inferências e efetuar correções manuais. |

---

## 🚀 Como Executar

Ative o ambiente virtual `vc_01` e rode os comandos a partir da raiz do projeto:

### 1. Executar a Contagem Clássica via Terminal (CLI)
Por padrão, o script roda no modo de máscaras baseadas em Canny (`edges`), cujas contagens estão alinhadas com as tabelas oficiais do relatório:
```bash
python Desafio_1/solucoes/solucao_1_opencv_morfologia/solucao_1_opencv_morfologia.py --input-dir Desafio_1/data/images
```

### 2. Executar a Aplicação Dash
```bash
python Desafio_1/solucoes/app_dash_desafio_1.py
```
*   O app Dash escuta no host `0.0.0.0` na porta `8061`.
*   Acesse localmente em: `http://127.0.0.1:8061`
*   Para rodar no smartphone de picking, garanta que ambos os dispositivos estejam no mesmo Wi-Fi local e acesse: `http://<IP_DO_COMPUTADOR>:8061`.

---

## 📈 Recomendação de Entrega
Apresente a **Solução 1 (OpenCV clássico com morfologia)** como a solução oficial para a empresa, dado o volume restrito de dados (5 imagens sem anotações). O Dash adiciona pontuação extra, provando que o operador em campo pode revisar e corrigir a contagem automática instantaneamente pela tela de um celular.
