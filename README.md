# Desafio - Visão Computacional

Projeto desenvolvido para o Bootcamp/Residência em IA, com duas aplicações de visão computacional voltadas a cenários industriais e de infraestrutura.

## Visão Geral

- **Desafio 1:** contagem de parafusos em imagens para apoio a processos de picking e controle de qualidade.
- **Desafio 2:** detecção e segmentação de fissuras em superfícies de concreto usando aprendizado supervisionado.

## Soluções Principais

### Desafio 1 - OpenCV

A solução oficial usa OpenCV com:

- escala de cinza;
- CLAHE para realce de contraste local;
- Canny para detecção de bordas;
- operações morfológicas;
- extração e filtragem de contornos por área, circularidade e razão de aspecto.

As cinco imagens originais do Desafio 1 foram mantidas no repositório:

```text
Desafio_1/data/images/
```

Execução:

```bash
python Desafio_1/solucoes/solucao_1_opencv_morfologia/solucao_1_opencv_morfologia.py --input-dir Desafio_1/data/images
```

Aplicação Dash:

```bash
python Desafio_1/solucoes/app_dash_desafio_1.py
```

Acesso local:

```text
http://127.0.0.1:8061
```

### Desafio 2 - YOLO26n-seg

A solução oficial usa YOLO26n-seg para segmentação de instâncias de fissuras.

O dataset original do Desafio 2 e os pesos `.pt` não foram incluídos nesta versão para manter o repositório enxuto. Para reexecutar o pipeline completo, recoloque:

```text
Desafio_2/data/images/
Desafio_2/data/labels/
Desafio_2/modelos/yolo26n-seg.pt
```

Os arquivos externos podem ser baixados separadamente pelo link abaixo:

```text
https://1drv.ms/u/c/faa9e6024cd17b33/IQD5GuL3KURNQogI1Nts6v-DATqGckUzZWtbpomQRFhpCqg?e=hijPjb
```

Depois de baixar, extraia o pacote na raiz do projeto, preservando a estrutura de diretórios.

Preparar split:

```bash
python Desafio_2/solucoes/solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.py --prepare
```

Treino de teste:

```bash
python Desafio_2/solucoes/solucao_1_yolo_segmentacao/solucao_1_yolo_segmentacao.py --train --epochs 3 --batch 4 --imgsz 640
```

Aplicação Dash:

```bash
python Desafio_2/solucoes/app_dash_desafio_2.py
```

Acesso local:

```text
http://127.0.0.1:8062
```

## Resultados Registrados

No relatório técnico, o modelo YOLO26n-seg final foi treinado por 120 épocas em GPU NVIDIA GTX 1650 de 4 GB.

Principais métricas no teste independente:

| Métrica | Valor |
|---|---:|
| Box mAP50 | 0,821 |
| Mask mAP50 | 0,693 |
| Mask mAP50-95 | 0,270 |
| Tempo médio de inferência | 14,1 ms/imagem |

## Documentação

- `doc/RELATORIO_TECNICO_Manoel_Furtado.pdf`
- `doc/relatorios/RELATORIO_TECNICO_ABNT_AJUSTADO.md`
- `doc/relatorios/ESPECIFICACAO_CRISPDM_SDD.md`
- `doc/PREPARACAO_ENTREVISTA_RESIDENCIA_IA.md`

## Ambiente

Com Mamba/Conda:

```bash
cd env
mamba env create -f env_vc_01.yml --channel-priority flexible
mamba activate vc_01
python test_env_vc_01.py
cd ..
```

Com pip:

```bash
pip install -r env/requirements_solucoes.txt
```

## Observação Sobre Arquivos Grandes

Este repositório foi preparado para GitHub sem:

- ambiente virtual;
- datasets pesados do Desafio 2;
- pesos de modelos;
- logs de treino;
- runs e saídas geradas.

Esses arquivos devem ser mantidos fora do Git ou gerenciados com Git LFS/releases quando necessário.

## Arquivos Externos

Para manter o GitHub leve e respeitar boas práticas de versionamento, os dados grandes e artefatos de modelo ficam fora do repositório. O pacote externo recomendado deve preservar esta árvore:

```text
Desafio_2/
  data/
    images/
    labels/
  modelos/
    yolo26n-seg.pt
  solucoes/
    dataset_yolo_split/
      runs/
        fissuras_yolo_seg/
          weights/
            best.pt
            last.pt
```

Conteúdo recomendado do pacote:

- imagens e labels originais do Desafio 2;
- peso base `Desafio_2/modelos/yolo26n-seg.pt`;
- pesos treinados `best.pt` e `last.pt`, se disponíveis;
- logs ou resultados de treino apenas se forem úteis para auditoria.

Não inclua no pacote:

- ambiente virtual;
- caches;
- arquivos `__pycache__`;
- saídas temporárias;
- arquivos duplicados sem necessidade.

Após extrair o pacote na raiz do projeto, os comandos de preparação, treino, inferência e Dash voltam a encontrar os dados e modelos nos caminhos esperados.
