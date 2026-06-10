"""Aplicação Dash do Desafio 1: contagem assistida de parafusos.

Autor: Manoel Furtado
Data: 31/05/2026

Descrição:
    Esta interface web permite testar três abordagens para o Desafio 1:
    OpenCV com contornos, template matching e pseudo-label para detector leve.
    O usuário envia uma imagem, ajusta parâmetros e compara a contagem
    automática com uma correção manual.

Execução:
    python Desafio_1/solucoes/app_dash_desafio_1.py

URL padrão:
    http://127.0.0.1:8061
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
from pathlib import Path

# Add solution subdirectories to sys.path for modular imports
script_dir = Path(__file__).resolve().parent
sys.path.extend([
    str(script_dir / "solucao_1_opencv_morfologia"),
    str(script_dir / "solucao_2_template_matching_interativo"),
    str(script_dir / "solucao_3_dados_sinteticos_detector_leve"),
])

from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from solucao_1_opencv_morfologia import parse_args, process_image, require_cv2
from solucao_2_template_matching_interativo import (
    annotate as annotate_template_matches,
    match_image,
    parse_args as parse_template_args,
)
from solucao_3_dados_sinteticos_detector_leve import (
    parse_args as parse_synthetic_args,
    weak_boxes,
)


# ---------------------------------------------------------------------------
# Conversões de imagem
# ---------------------------------------------------------------------------
# Dash recebe uploads em base64 e exibe imagens também como data URL. Estes
# helpers isolam essa conversão para manter os callbacks mais legíveis.

def data_url_from_file(path: Path, mime: str = "image/jpeg") -> str:
    """Converte um arquivo local em data URL para exibição no navegador."""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def uploaded_bytes(contents: str) -> bytes:
    """Extrai bytes de um arquivo enviado pelo componente dcc.Upload."""

    _, encoded = contents.split(",", 1)
    return base64.b64decode(encoded)


def image_array_to_data_url(cv2, image, mime: str = "image/jpeg") -> str:
    """Converte uma imagem OpenCV em data URL sem gravar arquivo intermediário."""

    extension = ".jpg" if mime == "image/jpeg" else ".png"
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise ValueError("Could not encode image.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def annotate_boxes(cv2, image, boxes, label: str):
    """Desenha caixas de candidatos usados na solução de pseudo-label."""

    output = image.copy()
    for index, box in enumerate(boxes, start=1):
        x, y, w, h = box.x, box.y, box.w, box.h
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 180, 0), 2)
        cv2.putText(
            output,
            str(index),
            (x, max(y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 120, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def auto_template_from_uploaded(cv2, np, image, min_area: int, max_area: int, expected_area: int):
    """Recorta automaticamente um possível parafuso para usar como template."""

    from solucao_1_opencv_morfologia import pick_best_mask

    args = parse_args(
        [
            "--min-area",
            str(min_area),
            "--max-area",
            str(max_area),
            "--expected-area",
            str(expected_area),
        ]
    )
    _, _, candidates = pick_best_mask(cv2, np, image, args)
    if not candidates:
        return None
    candidate = max(candidates, key=lambda item: item.area)
    margin = 8
    x1 = max(candidate.x - margin, 0)
    y1 = max(candidate.y - margin, 0)
    x2 = min(candidate.x + candidate.w + margin, image.shape[1])
    y2 = min(candidate.y + candidate.h + margin, image.shape[0])
    return image[y1:y2, x1:x2].copy()


# Metadados usados pelos cards e páginas. Manter textos aqui evita duplicação
# quando a interface mostra resumo, uso recomendado e risco principal.
SOLUTIONS = {
    "/opencv": {
        "title": "Solução 1 - OpenCV com contornos",
        "summary": "Segmentação clássica, bordas, morfologia e filtros geométricos.",
        "good_for": "Entrega principal segura com poucos dados.",
        "risk": "Pode separar cabeça/corpo ou fragmentar parafusos sobrepostos.",
    },
    "/template": {
        "title": "Solução 2 - Template matching",
        "summary": "Busca por similaridade a partir de um recorte de referência.",
        "good_for": "Validação de padrão e revisão assistida.",
        "risk": "Sensível a rotação, escala e template ruim.",
    },
    "/yolo-weak": {
        "title": "Solução 3 - YOLO leve / pseudo-label",
        "summary": "Pseudo-rotulagem e caminho de evolução para detector leve.",
        "good_for": "Mostrar criatividade e plano de escala futura.",
        "risk": "Sem mais dados reais, o detector pode decorar o cenário.",
    },
}


# `suppress_callback_exceptions=True` permite páginas dinâmicas: alguns
# componentes só existem quando a rota correspondente é renderizada.
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Contagem de parafusos"


# ---------------------------------------------------------------------------
# Componentes visuais reutilizáveis
# ---------------------------------------------------------------------------

def nav():
    """Menu de navegação simples entre as soluções."""

    return html.Nav(
        [
            dcc.Link("Início", href="/"),
            dcc.Link("OpenCV", href="/opencv"),
            dcc.Link("Template", href="/template"),
            dcc.Link("YOLO leve", href="/yolo-weak"),
        ],
        className="nav",
    )


def characteristics(path: str):
    """Bloco com resumo, recomendação e risco da solução selecionada."""

    item = SOLUTIONS[path]
    return html.Section(
        [
            html.H2(item["title"]),
            html.P(item["summary"]),
            html.Div(
                [
                    html.Div([html.Span("Uso recomendado"), html.Strong(item["good_for"])]),
                    html.Div([html.Span("Risco principal"), html.Strong(item["risk"])]),
                ],
                className="info-grid",
            ),
        ],
        className="intro",
    )


def upload_box(component_id: str, text: str = "Selecionar imagem", compact: bool = False):
    """Caixa de upload com estilo comum para imagem principal e template."""

    class_name = "upload-box compact" if compact else "upload-box"
    return dcc.Upload(
        id=component_id,
        children=html.Div([text]),
        accept="image/*",
        className=class_name,
    )


def metric_panel(prefix: str):
    """Painel com contagem automática, correção humana e resultado final."""

    return html.Section(
        [
            html.Div(
                [
                    html.Span("Automática", className="metric-label"),
                    html.Strong("-", id=f"{prefix}-automatic-count", className="metric-value"),
                ],
                className="metric",
            ),
            html.Div(
                [
                    html.Label("Correção"),
                    dcc.Input(
                        id=f"{prefix}-manual-count",
                        type="number",
                        min=0,
                        step=1,
                        value=0,
                        className="number-input",
                    ),
                ],
                className="metric",
            ),
            html.Div(
                [
                    html.Span("Final", className="metric-label"),
                    html.Strong("0", id=f"{prefix}-final-count", className="metric-value"),
                ],
                className="metric",
            ),
            html.Div(id=f"{prefix}-method-label", className="method-label"),
        ],
        className="summary",
    )


def image_stage(prefix: str):
    """Área onde a imagem anotada é exibida após o processamento."""

    return html.Section(
        [
            html.Img(id=f"{prefix}-annotated-image", className="result-image"),
            html.Div(id=f"{prefix}-empty-state", children="Aguardando imagem.", className="empty"),
        ],
        className="image-stage",
    )


def home_page():
    """Página inicial com o cardápio de abordagens."""

    return html.Div(
        [
            html.Section(
                [
                    html.H2("Escolha uma abordagem"),
                    html.P("Cada página tem parâmetros próprios, características e resultado visual."),
                ],
                className="intro",
            ),
            html.Section(
                [
                    dcc.Link(
                        html.Article(
                            [
                                html.H3("Solução 1"),
                                html.H4("OpenCV com contornos"),
                                html.P("Melhor opção para entrega principal: leve, explicável e ajustável."),
                            ],
                            className="choice-card",
                        ),
                        href="/opencv",
                    ),
                    dcc.Link(
                        html.Article(
                            [
                                html.H3("Solução 2"),
                                html.H4("Template matching"),
                                html.P("Boa para comparar com um parafuso de referência e revisar casos duvidosos."),
                            ],
                            className="choice-card",
                        ),
                        href="/template",
                    ),
                    dcc.Link(
                        html.Article(
                            [
                                html.H3("Solução 3",
                                ),
                                html.H4("YOLO leve / pseudo-label"),
                                html.P("Protótipo de evolução para detector treinável quando houver mais dados."),
                            ],
                            className="choice-card",
                        ),
                        href="/yolo-weak",
                    ),
                ],
                className="choice-grid",
            ),
        ]
    )


def opencv_page():
    """Página da solução OpenCV principal.

    Os sliders controlam filtros geométricos e modos de segmentação. A correção
    manual fica no painel de métricas para registrar a contagem final revisada.
    """

    return html.Div(
        [
            characteristics("/opencv"),
            html.Main(
                [
                    html.Section(
                        [
                            upload_box("opencv-upload-image"),
                            html.Div(
                                [
                                    html.Label("Modo de segmentação"),
                                    dcc.Dropdown(
                                        id="opencv-mask-mode",
                                        options=[
                                            {"label": "Automático", "value": "auto"},
                                            {"label": "Limiar manual", "value": "manual"},
                                        ],
                                        value="auto",
                                        clearable=False,
                                    ),
                                    html.Label("Área mínima"),
                                    dcc.Slider(
                                        id="opencv-min-area",
                                        min=20,
                                        max=1000,
                                        step=10,
                                        value=80,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("Área máxima"),
                                    dcc.Slider(
                                        id="opencv-max-area",
                                        min=1000,
                                        max=50000,
                                        step=500,
                                        value=20000,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("Área esperada"),
                                    dcc.Slider(
                                        id="opencv-expected-area",
                                        min=500,
                                        max=6000,
                                        step=100,
                                        value=1600,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("Margem dos cantos"),
                                    dcc.Slider(
                                        id="opencv-corner-margin",
                                        min=0,
                                        max=40,
                                        step=1,
                                        value=8,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("Limiar manual"),
                                    dcc.Slider(
                                        id="opencv-manual-threshold",
                                        min=40,
                                        max=220,
                                        step=5,
                                        value=90,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ],
                                className="controls",
                            ),
                        ],
                        className="panel",
                    ),
                    metric_panel("opencv"),
                    image_stage("opencv"),
                ],
                className="shell",
            ),
        ]
    )


def template_page():
    """Página de template matching, com template enviado ou extraído automaticamente."""

    return html.Div(
        [
            characteristics("/template"),
            html.Main(
                [
                    html.Section(
                        [
                            upload_box("template-upload-image"),
                            html.Div(
                                [
                                    html.Label("Template opcional"),
                                    upload_box("template-upload-template", "Selecionar recorte", compact=True),
                                    html.Label("Threshold template"),
                                    dcc.Slider(
                                        id="template-threshold",
                                        min=0.30,
                                        max=0.95,
                                        step=0.05,
                                        value=0.62,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("NMS template"),
                                    dcc.Slider(
                                        id="template-nms",
                                        min=0.05,
                                        max=0.80,
                                        step=0.05,
                                        value=0.25,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("Área mínima para template automático"),
                                    dcc.Slider(
                                        id="template-min-area",
                                        min=20,
                                        max=1000,
                                        step=10,
                                        value=80,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("Área máxima para template automático"),
                                    dcc.Slider(
                                        id="template-max-area",
                                        min=1000,
                                        max=50000,
                                        step=500,
                                        value=20000,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ],
                                className="controls",
                            ),
                        ],
                        className="panel",
                    ),
                    metric_panel("template"),
                    image_stage("template"),
                ],
                className="shell",
            ),
        ]
    )


def yolo_weak_page():
    """Página de pseudo-label/YOLO leve como caminho de evolução."""

    return html.Div(
        [
            characteristics("/yolo-weak"),
            html.Main(
                [
                    html.Section(
                        [
                            upload_box("yolo-upload-image"),
                            html.Div(
                                [
                                    html.Label("Pesos YOLO opcionais"),
                                    dcc.Input(
                                        id="yolo-weights",
                                        type="text",
                                        placeholder="runs/.../best.pt",
                                        value="",
                                        className="text-input",
                                    ),
                                    html.Label("Confianca YOLO"),
                                    dcc.Slider(
                                        id="yolo-conf",
                                        min=0.05,
                                        max=0.90,
                                        step=0.05,
                                        value=0.25,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("Área mínima pseudo-label"),
                                    dcc.Slider(
                                        id="yolo-min-area",
                                        min=20,
                                        max=1000,
                                        step=10,
                                        value=80,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("Área máxima pseudo-label"),
                                    dcc.Slider(
                                        id="yolo-max-area",
                                        min=1000,
                                        max=50000,
                                        step=500,
                                        value=20000,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ],
                                className="controls",
                            ),
                        ],
                        className="panel",
                    ),
                    metric_panel("yolo"),
                    image_stage("yolo"),
                ],
                className="shell",
            ),
        ]
    )


app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.Header(
            [
                html.Div(
                    [
                        html.H1("Contagem de parafusos"),
                        html.P("Picking assistido com tres abordagens comparaveis."),
                    ]
                ),
                nav(),
            ],
            className="hero",
        ),
        html.Div(id="page-content", className="page"),
    ]
)

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #20242a; }
            a { color: inherit; text-decoration: none; }
            .hero {
                padding: 24px 32px 16px;
                background: #ffffff;
                border-bottom: 1px solid #dfe3e8;
                display: flex;
                justify-content: space-between;
                gap: 18px;
                align-items: end;
                flex-wrap: wrap;
            }
            .hero h1 { margin: 0 0 6px; font-size: 30px; letter-spacing: 0; }
            .hero p { margin: 0; color: #5f6875; }
            .nav { display: flex; gap: 8px; flex-wrap: wrap; }
            .nav a {
                padding: 9px 12px;
                border: 1px solid #cfd6e1;
                border-radius: 8px;
                background: #f8fafc;
                font-weight: 700;
                font-size: 13px;
            }
            .page { max-width: 1180px; margin: 0 auto; padding: 24px; }
            .intro {
                background: #ffffff;
                border: 1px solid #dfe3e8;
                border-radius: 8px;
                padding: 18px;
                margin-bottom: 18px;
            }
            .intro h2 { margin: 0 0 8px; font-size: 23px; }
            .intro p { margin: 0; color: #5f6875; }
            .info-grid {
                margin-top: 14px;
                display: grid;
                grid-template-columns: repeat(2, minmax(180px, 1fr));
                gap: 10px;
            }
            .info-grid div, .choice-card, .panel, .summary, .image-stage {
                background: #ffffff;
                border: 1px solid #dfe3e8;
                border-radius: 8px;
            }
            .info-grid div { padding: 12px; display: grid; gap: 5px; }
            .info-grid span, .metric-label, .metric label, .method-label {
                font-size: 12px;
                color: #5f6875;
                font-weight: 700;
            }
            .info-grid strong { font-size: 14px; }
            .choice-grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 16px; }
            .choice-card { min-height: 190px; padding: 18px; display: grid; align-content: start; gap: 8px; }
            .choice-card h3 { margin: 0; color: #5f6875; font-size: 13px; }
            .choice-card h4 { margin: 0; font-size: 20px; }
            .choice-card p { margin: 0; color: #5f6875; line-height: 1.4; }
            .shell { display: grid; grid-template-columns: 330px 1fr; gap: 18px; }
            .panel { padding: 18px; }
            .upload-box {
                border: 2px dashed #8a96a8;
                border-radius: 8px;
                padding: 24px;
                text-align: center;
                cursor: pointer;
                background: #fbfcfd;
                font-weight: 700;
            }
            .upload-box.compact { padding: 12px; font-size: 13px; }
            .controls { margin-top: 22px; display: grid; gap: 14px; }
            .controls label { font-size: 13px; font-weight: 700; color: #47515f; }
            .text-input, .number-input {
                width: 100%;
                box-sizing: border-box;
                padding: 10px;
                border: 1px solid #c8d0da;
                border-radius: 6px;
                font-size: 14px;
            }
            .summary {
                padding: 16px;
                display: grid;
                grid-template-columns: repeat(3, minmax(110px, 1fr));
                gap: 12px;
                align-items: end;
            }
            .metric {
                min-height: 72px;
                padding: 12px;
                background: #f8fafc;
                border: 1px solid #e4e8ee;
                border-radius: 8px;
                display: grid;
                align-content: center;
                gap: 6px;
            }
            .metric-value { font-size: 30px; line-height: 1; }
            .number-input { font-size: 20px; font-weight: 700; }
            .method-label { grid-column: 1 / -1; }
            .image-stage {
                grid-column: 1 / -1;
                min-height: 520px;
                display: grid;
                place-items: center;
                padding: 18px;
            }
            .result-image { max-width: 100%; max-height: 760px; object-fit: contain; }
            .result-image[src=""] { display: none; }
            .empty { color: #6c7583; font-weight: 700; }
            @media (max-width: 900px) {
                .page { padding: 14px; }
                .shell, .choice-grid, .info-grid { grid-template-columns: 1fr; }
                .summary { grid-template-columns: 1fr; }
                .image-stage { min-height: 320px; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
"""


# ---------------------------------------------------------------------------
# Callbacks de navegação
# ---------------------------------------------------------------------------

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    """Escolhe qual página mostrar a partir da URL."""

    if pathname == "/opencv":
        return opencv_page()
    if pathname == "/template":
        return template_page()
    if pathname == "/yolo-weak":
        return yolo_weak_page()
    return home_page()


# ---------------------------------------------------------------------------
# Callbacks de processamento
# ---------------------------------------------------------------------------

@app.callback(
    Output("opencv-automatic-count", "children"),
    Output("opencv-manual-count", "value"),
    Output("opencv-method-label", "children"),
    Output("opencv-annotated-image", "src"),
    Output("opencv-empty-state", "style"),
    Input("opencv-upload-image", "contents"),
    Input("opencv-min-area", "value"),
    Input("opencv-max-area", "value"),
    Input("opencv-expected-area", "value"),
    Input("opencv-corner-margin", "value"),
    Input("opencv-mask-mode", "value"),
    Input("opencv-manual-threshold", "value"),
    State("opencv-upload-image", "filename"),
)
def run_opencv(contents, min_area, max_area, expected_area, corner_margin, mask_mode, manual_threshold, filename):
    """Processa a imagem enviada pela solução OpenCV.

    O arquivo é gravado em um diretório temporário, processado pelo script da
    solução 1 e depois convertido para data URL para aparecer no Dash.
    """

    if not contents:
        raise PreventUpdate
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        safe_name = Path(filename or "upload.jpg").name
        input_path = temp_path / safe_name
        input_path.write_bytes(uploaded_bytes(contents))
        output_dir = temp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2_mod, np_mod = require_cv2()

        # Reaproveita o parser do script de linha de comando para garantir que
        # o app e o CLI usem os mesmos parâmetros e padrões.
        selected_mask_mode = str(mask_mode or "auto")
        args = parse_args(
            [
                "--input-dir",
                str(temp_path),
                "--output-dir",
                str(output_dir),
                "--min-area",
                str(min_area),
                "--max-area",
                str(max_area),
                "--expected-area",
                str(expected_area),
                "--corner-margin",
                str(corner_margin),
                "--mask-mode",
                selected_mask_mode,
                "--manual-threshold",
                str(manual_threshold),
            ]
        )
        row = process_image(cv2_mod, np_mod, input_path, output_dir, args)
        annotated_path = output_dir / f"{input_path.stem}_annotated.jpg"
        count = int(row["count"])
        method = f"Solução 1: OpenCV | método: {row['method']} | área média: {row['mean_area']}"
        return str(count), count, method, data_url_from_file(annotated_path), {"display": "none"}


@app.callback(
    Output("template-automatic-count", "children"),
    Output("template-manual-count", "value"),
    Output("template-method-label", "children"),
    Output("template-annotated-image", "src"),
    Output("template-empty-state", "style"),
    Input("template-upload-image", "contents"),
    Input("template-upload-template", "contents"),
    Input("template-threshold", "value"),
    Input("template-nms", "value"),
    Input("template-min-area", "value"),
    Input("template-max-area", "value"),
    State("template-upload-image", "filename"),
    State("template-upload-template", "filename"),
)
def run_template(contents, template_contents, threshold, nms, min_area, max_area, filename, template_filename):
    """Executa template matching na imagem enviada."""

    if not contents:
        raise PreventUpdate
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        safe_name = Path(filename or "upload.jpg").name
        input_path = temp_path / safe_name
        input_path.write_bytes(uploaded_bytes(contents))
        cv2_mod, np_mod = require_cv2()
        image = cv2_mod.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not read uploaded image: {safe_name}")

        # O template pode vir do usuário ou ser extraído automaticamente usando
        # a solução OpenCV como apoio.
        if template_contents:
            template_name = Path(template_filename or "template.jpg").name
            template_path = temp_path / template_name
            template_path.write_bytes(uploaded_bytes(template_contents))
            template = cv2_mod.imread(str(template_path))
        else:
            template = auto_template_from_uploaded(cv2_mod, np_mod, image, int(min_area), int(max_area), 1600)
        if template is None:
            return "0", 0, "Envie um recorte/template ou ajuste os parâmetros.", "", {"display": "block"}

        args = parse_template_args(["--threshold", str(threshold), "--nms-threshold", str(nms)])
        matches = match_image(cv2_mod, np_mod, image, template, args)
        annotated = annotate_template_matches(cv2_mod, image, matches)
        avg_score = sum(score for _, score in matches) / len(matches) if matches else 0.0
        count = len(matches)
        method = f"Solução 2: template matching | score médio: {avg_score:.3f}"
        return str(count), count, method, image_array_to_data_url(cv2_mod, annotated), {"display": "none"}


@app.callback(
    Output("yolo-automatic-count", "children"),
    Output("yolo-manual-count", "value"),
    Output("yolo-method-label", "children"),
    Output("yolo-annotated-image", "src"),
    Output("yolo-empty-state", "style"),
    Input("yolo-upload-image", "contents"),
    Input("yolo-weights", "value"),
    Input("yolo-conf", "value"),
    Input("yolo-min-area", "value"),
    Input("yolo-max-area", "value"),
    State("yolo-upload-image", "filename"),
)
def run_yolo_or_weak(contents, weights, conf, min_area, max_area, filename):
    """Executa YOLO se houver pesos; caso contrário, usa pseudo-label OpenCV."""

    if not contents:
        raise PreventUpdate
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        safe_name = Path(filename or "upload.jpg").name
        input_path = temp_path / safe_name
        input_path.write_bytes(uploaded_bytes(contents))
        cv2_mod, np_mod = require_cv2()
        image = cv2_mod.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not read uploaded image: {safe_name}")

        # Caminho opcional: se pesos YOLO forem informados, o app tenta usar
        # inferência real. Sem pesos, mostra a proposta de pseudo-rotulagem.
        if weights and weights.strip():
            try:
                from ultralytics import YOLO  # type: ignore

                model = YOLO(weights.strip())
                results = model.predict(source=str(input_path), conf=float(conf))
                result = results[0]
                count = len(result.boxes)
                method = f"Solução 3: YOLO leve | confiança: {float(conf):.2f}"
                return str(count), count, method, image_array_to_data_url(cv2_mod, result.plot()), {"display": "none"}
            except Exception as exc:  # noqa: BLE001
                method = f"Falha ao executar YOLO: {exc}"
                return "0", 0, method, image_array_to_data_url(cv2_mod, image), {"display": "none"}

        args = parse_synthetic_args(["--min-area", str(min_area), "--max-area", str(max_area)])
        boxes = weak_boxes(cv2_mod, np_mod, image, args)
        annotated = annotate_boxes(cv2_mod, image, boxes, f"count={len(boxes)} method=pseudo-label")
        count = len(boxes)
        method = "Solução 3: pseudo-label para YOLO leve | sem pesos treinados"
        return str(count), count, method, image_array_to_data_url(cv2_mod, annotated), {"display": "none"}


@app.callback(Output("opencv-final-count", "children"), Input("opencv-manual-count", "value"))
def update_opencv_final(value):
    """Atualiza a contagem final revisada na página OpenCV."""

    return str(int(value or 0))


@app.callback(Output("template-final-count", "children"), Input("template-manual-count", "value"))
def update_template_final(value):
    """Atualiza a contagem final revisada na página Template."""

    return str(int(value or 0))


@app.callback(Output("yolo-final-count", "children"), Input("yolo-manual-count", "value"))
def update_yolo_final(value):
    """Atualiza a contagem final revisada na página YOLO/pseudo-label."""

    return str(int(value or 0))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "8061")))
