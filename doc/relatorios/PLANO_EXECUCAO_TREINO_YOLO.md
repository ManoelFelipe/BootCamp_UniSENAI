# Plano Manual de Execução - Treino YOLO de Segmentação

Este plano descreve como treinar manualmente o modelo YOLO para o Desafio 2, evitando deixar processos longos rodando sem controle.

## 1. Estado inicial limpo

Antes de iniciar, confirme que não há treino em execução:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*solucao_1_yolo_segmentacao.py*' } |
  Select-Object ProcessId, CommandLine
```

Confirme também o uso da GPU:

```powershell
nvidia-smi
```

Se aparecer algum processo Python de treino antigo, interrompa pelo `ProcessId`:

```powershell
Stop-Process -Id <PROCESS_ID> -Force
```

## 2. Ativar o ambiente

A partir da raiz do projeto:

```powershell
mamba activate vc_01
python env\test_env_vc_01.py
```

O ambiente oficial é `env/env_vc_01.yml`. Não use `requirements_solucoes.txt` como referência principal.

## 3. Escolher o modelo base

Para segmentação, o modelo precisa ter sufixo `-seg`.

Opção validada e recomendada para o projeto:

```text
Desafio_2/modelos/yolo26n-seg.pt
```

Motivo: é o modelo nano de segmentação da família YOLO26, mais recente e voltada a eficiência em dispositivos de borda. Ele foi aceito pelo `ultralytics 8.4.56` e treinou corretamente no ambiente `vc_01`.

Alternativas:

```text
yolo11n-seg.pt
yolov8n-seg.pt
```

Use `yolov8n-seg.pt` apenas como fallback se houver problema de compatibilidade em outra máquina.

Teste rápido de suporte ao modelo:

```powershell
python -c "from ultralytics import YOLO; YOLO(r'Desafio_2\modelos\yolo26n-seg.pt'); print('OK')"
```

Se falhar, tente:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt'); print('OK')"
```

Se também falhar, use:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt'); print('OK')"
```

## 4. Preparar o dataset

Este comando recria `train.txt`, `val.txt`, `test.txt` e `data.yaml`:

```powershell
python Desafio_2\solucoes\solucao_1_yolo_segmentacao.py --prepare --clean
```

Resultado esperado:

```text
Pairs: train=1164 val=232 test=155
Created YOLO dataset at ...\Desafio_2\solucoes\dataset_yolo_split
```

## 5. Fazer um treino curto de validação

Antes do treino longo, rode 3 épocas. Isso valida GPU, dataset, download do modelo e escrita dos resultados.

Com `Desafio_2/modelos/yolo26n-seg.pt`:

```powershell
python Desafio_2\solucoes\solucao_1_yolo_segmentacao.py `
  --train `
  --model Desafio_2\modelos\yolo26n-seg.pt `
  --epochs 3 `
  --batch 4 `
  --imgsz 640
```

Se aparecer erro de memória CUDA, reduza:

```powershell
python Desafio_2\solucoes\solucao_1_yolo_segmentacao.py `
  --train `
  --model Desafio_2\modelos\yolo26n-seg.pt `
  --epochs 3 `
  --batch 2 `
  --imgsz 640
```

Na GTX 1650 de 4 GB, `batch=8` chegou a funcionar, mas ficou perto do limite da VRAM. Para execução manual estável, use `batch=4`.

Resultado validado no hardware local:

| Item | Valor |
|---|---:|
| Modelo | `Desafio_2/modelos/yolo26n-seg.pt` |
| Ultralytics | `8.4.56` |
| GPU | NVIDIA GeForce GTX 1650, 4 GB |
| Épocas | 3 |
| Batch | 4 |
| Imagem | 640 |
| Tempo | 0,145 h |
| Box precision | 0,592 |
| Box recall | 0,448 |
| Box mAP50 | 0,509 |
| Box mAP50-95 | 0,318 |
| Mask precision | 0,566 |
| Mask recall | 0,404 |
| Mask mAP50 | 0,419 |
| Mask mAP50-95 | 0,137 |

Esse resultado é bom para smoke test. Ele não é métrica final, porque 3 épocas ainda são pouco para convergência.

## 6. Rodar o treino completo

Sugestão equilibrada:

```powershell
python -u Desafio_2\solucoes\solucao_1_yolo_segmentacao.py `
  --prepare `
  --train `
  --clean `
  --model Desafio_2\modelos\yolo26n-seg.pt `
  --epochs 120 `
  --batch 4 `
  --imgsz 640 `
  *> Desafio_2\solucoes\treino_yolo26n_seg.log
```

Se preferir o modelo já usado inicialmente no projeto:

```powershell
python -u Desafio_2\solucoes\solucao_1_yolo_segmentacao.py `
  --prepare `
  --train `
  --clean `
  --model yolov8n-seg.pt `
  --epochs 120 `
  --batch 4 `
  --imgsz 640 `
  *> Desafio_2\solucoes\treino_yolov8n_seg.log
```

## 7. Monitorar o treino

Em outro terminal:

```powershell
nvidia-smi -l 5
```

Para acompanhar o log:

```powershell
Get-Content Desafio_2\solucoes\treino_yolo26n_seg.log -Tail 40 -Wait
```

Para acompanhar métricas já consolidadas:

```powershell
Get-Content Desafio_2\solucoes\dataset_yolo_split\runs\fissuras_yolo_seg\results.csv -Tail 5
```

## 8. Artefatos esperados

Ao final, os principais arquivos estarão em:

```text
Desafio_2/solucoes/dataset_yolo_split/runs/fissuras_yolo_seg/
```

Pesos principais:

```text
weights/best.pt
weights/last.pt
```

Use `best.pt` para validação, inferência e entrega ao usuário final.

## 9. Validar no conjunto de teste

Depois do treino, rode validação no split de teste:

```powershell
python -c "from ultralytics import YOLO; model = YOLO(r'Desafio_2\solucoes\dataset_yolo_split\runs\fissuras_yolo_seg\weights\best.pt'); model.val(data=r'Desafio_2\solucoes\dataset_yolo_split\data.yaml', split='test', imgsz=640)"
```

Registre no relatório:

- `mAP50(M)`;
- `mAP50-95(M)`;
- precision de máscara;
- recall de máscara;
- exemplos visuais de acertos;
- exemplos de falsos positivos e falsos negativos.

## 10. Fazer predições de exemplo

```powershell
python Desafio_2\solucoes\solucao_1_yolo_segmentacao.py `
  --predict `
  --weights Desafio_2\solucoes\dataset_yolo_split\runs\fissuras_yolo_seg\weights\best.pt `
  --source Desafio_2\data\images `
  --conf 0.25
```

As imagens preditas serão salvas em:

```text
Desafio_2/solucoes/dataset_yolo_split/predicoes/
```

## 11. Exportar para uso final

Para uso web com Dash, o `best.pt` já é suficiente no servidor.

Para uso em outros ambientes, exporte:

```powershell
python Desafio_2\solucoes\solucao_1_yolo_segmentacao.py `
  --export `
  --weights Desafio_2\solucoes\dataset_yolo_split\runs\fissuras_yolo_seg\weights\best.pt `
  --export-format onnx `
  --imgsz 640
```

Para mobile, avalie depois TFLite ou outro formato suportado:

```powershell
python Desafio_2\solucoes\solucao_1_yolo_segmentacao.py `
  --export `
  --weights Desafio_2\solucoes\dataset_yolo_split\runs\fissuras_yolo_seg\weights\best.pt `
  --export-format tflite `
  --imgsz 640
```

## 12. Uso pelo usuário final

Sim: depois que o modelo for treinado de forma ampla e validado, o usuário final pode usar esse modelo treinado.

O fluxo correto é:

1. Treinar e validar o modelo uma vez.
2. Salvar `weights/best.pt`.
3. Configurar a aplicação Dash para carregar esse `best.pt`.
4. O usuário final acessa a interface web, envia uma imagem e recebe a máscara da fissura.
5. O usuário final não precisa treinar modelo nem conhecer Python.

Para uma versão web interna, o modelo pode ficar no servidor e o Dash executa apenas inferência. Para produto externo, será necessário cuidar de autenticação, limite de upload, segurança, logs, limpeza de arquivos temporários e licença de uso do Ultralytics.

## 13. Referências oficiais

- Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26/
- Ultralytics Instance Segmentation: https://docs.ultralytics.com/tasks/segment/
