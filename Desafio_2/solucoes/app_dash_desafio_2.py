"""Aplicação Dash do Desafio 2: detecção e segmentação de fissuras.

Autor: Manoel Furtado
Data: 31/05/2026

Descrição:
    Esta interface web demonstra três caminhos para o Desafio 2: baseline
    OpenCV, segmentação YOLO e U-Net leve como evolução. O usuário envia uma
    imagem e o app mostra máscara/resultado visual, métricas simples e status
    do processamento.

Execução:
    python Desafio_2/solucoes/app_dash_desafio_2.py

URL padrão:
    http://127.0.0.1:8062
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
    str(script_dir / "solucao_1_yolo_segmentacao"),
    str(script_dir / "solucao_2_unet_leve"),
    str(script_dir / "solucao_3_opencv_assistido"),
])

from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from solucao_3_opencv_assistido import (
    component_summary,
    crack_mask,
    overlay,
    parse_args,
    require_cv2,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
TRAINED_YOLO_WEIGHTS = (
    PROJECT_DIR
    / "Desafio_2"
    / "solucoes"
    / "dataset_yolo_split"
    / "runs"
    / "fissuras_yolo_seg"
    / "weights"
    / "best.pt"
)
BASE_YOLO_WEIGHTS = PROJECT_DIR / "Desafio_2" / "modelos" / "yolo26n-seg.pt"
DEFAULT_YOLO_WEIGHTS = TRAINED_YOLO_WEIGHTS if TRAINED_YOLO_WEIGHTS.exists() else BASE_YOLO_WEIGHTS


# ---------------------------------------------------------------------------
# Conversões de upload e imagem
# ---------------------------------------------------------------------------

def uploaded_bytes(contents: str) -> bytes:
    """Extrai os bytes de uma imagem enviada via dcc.Upload."""

    _, encoded = contents.split(",", 1)
    return base64.b64decode(encoded)


def array_to_data_url(cv2, image, mime: str = "image/jpeg") -> str:
    """Converte uma imagem OpenCV em data URL para renderização no Dash."""

    extension = ".jpg" if mime == "image/jpeg" else ".png"
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise ValueError("Could not encode image.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# Metadados exibidos nos cards e no cabeçalho de cada página.
SOLUTIONS = {
    "/opencv": {
        "title": "Solução 3 - OpenCV baseline",
        "summary": "Black-hat, Canny, morfologia e componentes conectados.",
        "good_for": "Baseline explicável e comparação com IA supervisionada.",
        "risk": "Textura, poeira e sombras podem gerar regiões suspeitas falsas.",
    },
    "/yolo": {
        "title": "Solução 1 - YOLO segmentação",
        "summary": "Segmentação supervisionada usando os polígonos YOLO existentes.",
        "good_for": "Solução final recomendada quando houver pesos treinados.",
        "risk": "Depende de treino, validação e caminho dos pesos do modelo.",
    },
    "/unet": {
        "title": "Solução 2 - U-Net leve",
        "summary": "Conversão dos polígonos em máscaras e segmentação pixel a pixel.",
        "good_for": "Alternativa acadêmica forte para fissuras finas.",
        "risk": "Inferência no app depende de modelo treinado/exportado.",
    },
}


# Como cada rota possui componentes próprios, callbacks são registrados mesmo
# quando o componente ainda não está presente no layout inicial.
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Detecção de fissuras"


# ---------------------------------------------------------------------------
# Componentes visuais reutilizáveis
# ---------------------------------------------------------------------------

def nav():
    """Menu principal de navegação entre abordagens."""

    return html.Nav(
        [
            dcc.Link("Início", href="/"),
            dcc.Link("OpenCV", href="/opencv"),
            dcc.Link("YOLO", href="/yolo"),
            dcc.Link("U-Net", href="/unet"),
        ],
        className="nav",
    )


def characteristics(path: str):
    """Exibe resumo, uso recomendado e risco da abordagem selecionada."""

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


def upload_box(component_id: str, text: str = "Selecionar imagem"):
    """Cria o componente de upload de imagem com estilo comum."""

    return dcc.Upload(
        id=component_id,
        children=html.Div([text]),
        accept="image/*",
        className="upload-box",
    )


def metric_panel(prefix: str, status_text: str = "Aguardando"):
    """Painel de indicadores exibido em todas as páginas."""

    return html.Section(
        [
            html.Div(
                [
                    html.Span("Motor", className="metric-label"),
                    html.Strong("-", id=f"{prefix}-engine-label", className="metric-value small"),
                ],
                className="metric",
            ),
            html.Div(
                [
                    html.Span("Fissuras/regiões", className="metric-label"),
                    html.Strong("-", id=f"{prefix}-crack-count", className="metric-value"),
                ],
                className="metric",
            ),
            html.Div(
                [
                    html.Span("Área suspeita", className="metric-label"),
                    html.Strong("-", id=f"{prefix}-area-percent", className="metric-value small"),
                ],
                className="metric",
            ),
            html.Div(
                [
                    html.Span("Status", className="metric-label"),
                    html.Strong(status_text, id=f"{prefix}-status-label", className="metric-value small"),
                ],
                className="metric",
            ),
        ],
        className="summary",
    )


def image_stage(prefix: str):
    """Área central onde a imagem resultante é renderizada."""

    return html.Section(
        [
            html.Img(id=f"{prefix}-annotated-image", className="result-image"),
            html.Div(id=f"{prefix}-empty-state", children="Aguardando imagem.", className="empty"),
        ],
        className="image-stage",
    )


def home_page():
    """Página inicial com os cards de escolha da solução."""

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
                                html.H3("Solução 3"),
                                html.H4("OpenCV baseline"),
                                html.P("Linha de base explicável para comparar com modelos de IA."),
                            ],
                            className="choice-card",
                        ),
                        href="/opencv",
                    ),
                    dcc.Link(
                        html.Article(
                            [
                                html.H3("Solução 1"),
                                html.H4("YOLO segmentação"),
                                html.P("Caminho recomendado para entrega final com máscaras e posição da fissura."),
                            ],
                            className="choice-card",
                        ),
                        href="/yolo",
                    ),
                    dcc.Link(
                        html.Article(
                            [
                                html.H3("Solução 2"),
                                html.H4("U-Net leve"),
                                html.P("Alternativa pixel a pixel, forte para discutir evolução técnica."),
                            ],
                            className="choice-card",
                        ),
                        href="/unet",
                    ),
                ],
                className="choice-grid",
            ),
        ]
    )


def opencv_page():
    """Página do baseline OpenCV para fissuras."""

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
                                    html.Label("Filtro OpenCV (%)"),
                                    dcc.Slider(
                                        id="opencv-region-percent",
                                        min=0.05,
                                        max=5.00,
                                        step=0.05,
                                        value=1.00,
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


def yolo_page():
    """Página de inferência YOLO com pesos de segmentação."""

    return html.Div(
        [
            characteristics("/yolo"),
            html.Main(
                [
                    html.Section(
                        [
                            upload_box("yolo-upload-image"),
                            html.Div(
                                [
                                    html.Label("Pesos YOLO"),
                                    dcc.Input(
                                        id="yolo-weights",
                                        type="text",
                                        placeholder=str(DEFAULT_YOLO_WEIGHTS),
                                        value=str(DEFAULT_YOLO_WEIGHTS),
                                        className="text-input",
                                    ),
                                    html.Label("Confiança"),
                                    dcc.Slider(
                                        id="yolo-confidence",
                                        min=0.05,
                                        max=0.90,
                                        step=0.05,
                                        value=0.25,
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


def unet_page():
    """Página da alternativa U-Net, mantida como roadmap/inferência futura."""

    return html.Div(
        [
            characteristics("/unet"),
            html.Main(
                [
                    html.Section(
                        [
                            upload_box("unet-upload-image"),
                            html.Div(
                                [
                                    html.Label("Modelo U-Net"),
                                    dcc.Input(
                                        id="unet-weights",
                                        type="text",
                                        placeholder="dataset_unet_masks/tiny_unet_fissuras.pt",
                                        value="",
                                        className="text-input",
                                    ),
                                    html.P(
                                        "A preparação e o treino já estão nos scripts. A inferência visual no app fica como próximo passo.",
                                        className="note",
                                    ),
                                ],
                                className="controls",
                            ),
                        ],
                        className="panel",
                    ),
                    metric_panel("unet", status_text="Roadmap"),
                    image_stage("unet"),
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
                        html.H1("Detecção de fissuras"),
                        html.P("Comparação entre baseline clássico, YOLO e U-Net."),
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
            .info-grid span, .metric-label {
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
            .controls { margin-top: 22px; display: grid; gap: 14px; }
            .controls label { font-size: 13px; font-weight: 700; color: #47515f; }
            .text-input {
                width: 100%;
                box-sizing: border-box;
                padding: 10px;
                border: 1px solid #c8d0da;
                border-radius: 6px;
                font-size: 14px;
            }
            .note { margin: 0; color: #5f6875; line-height: 1.4; font-size: 13px; }
            .summary {
                padding: 16px;
                display: grid;
                grid-template-columns: 150px 150px 150px 1fr;
                gap: 12px;
                align-items: stretch;
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
            .metric-value.small { font-size: 18px; line-height: 1.25; }
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
# Callback de navegação
# ---------------------------------------------------------------------------

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    """Renderiza a página correspondente à rota atual."""

    if pathname == "/opencv":
        return opencv_page()
    if pathname == "/yolo":
        return yolo_page()
    if pathname == "/unet":
        return unet_page()
    return home_page()


# ---------------------------------------------------------------------------
# Callbacks de inferência/processamento
# ---------------------------------------------------------------------------

@app.callback(
    Output("opencv-engine-label", "children"),
    Output("opencv-crack-count", "children"),
    Output("opencv-area-percent", "children"),
    Output("opencv-status-label", "children"),
    Output("opencv-annotated-image", "src"),
    Output("opencv-empty-state", "style"),
    Input("opencv-upload-image", "contents"),
    Input("opencv-region-percent", "value"),
    State("opencv-upload-image", "filename"),
)
def run_opencv(contents, region_percent, filename):
    """Executa o baseline OpenCV na imagem enviada.

    O método gera uma máscara de regiões suspeitas e calcula uma métrica simples
    de área ocupada pela fissura/defeito.
    """

    if not contents:
        raise PreventUpdate
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        safe_name = Path(filename or "upload.jpg").name
        image_path = temp_path / safe_name
        image_path.write_bytes(uploaded_bytes(contents))
        cv2_mod, np_mod = require_cv2()
        image = cv2_mod.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read uploaded image: {safe_name}")

        # A área mínima cresce com o tamanho da imagem para evitar pequenos
        # ruídos como regiões relevantes.
        args = parse_args([])
        mask = crack_mask(cv2_mod, np_mod, image, args)
        min_region_area = max(500, int(mask.size * (float(region_percent) / 100.0)))
        summary = component_summary(cv2_mod, np_mod, mask, min_area=min_region_area)
        result = overlay(cv2_mod, np_mod, image, summary["filtered_mask"])
        status = f"Solução 3: regiões significativas >= {min_region_area} px."
        return (
            "OpenCV",
            str(summary["count"]),
            f"{summary['area_percent']:.2f}%",
            status,
            array_to_data_url(cv2_mod, result),
            {"display": "none"},
        )


@app.callback(
    Output("yolo-engine-label", "children"),
    Output("yolo-crack-count", "children"),
    Output("yolo-area-percent", "children"),
    Output("yolo-status-label", "children"),
    Output("yolo-annotated-image", "src"),
    Output("yolo-empty-state", "style"),
    Input("yolo-upload-image", "contents"),
    Input("yolo-weights", "value"),
    Input("yolo-confidence", "value"),
    State("yolo-upload-image", "filename"),
)
def run_yolo(contents, weights, confidence, filename):
    """Executa segmentação YOLO quando pesos válidos são informados."""

    if not contents:
        raise PreventUpdate
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        safe_name = Path(filename or "upload.jpg").name
        image_path = temp_path / safe_name
        image_path.write_bytes(uploaded_bytes(contents))
        cv2_mod, _ = require_cv2()
        image = cv2_mod.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read uploaded image: {safe_name}")

        # Sem pesos, o app mostra a imagem original e orienta o usuário. Isso
        # evita falha dura na demonstração.
        if not (weights and weights.strip()):
            return (
                "YOLO",
                "-",
                "-",
                "Informe o caminho dos pesos YOLO treinados para executar a segmentação.",
                array_to_data_url(cv2_mod, image),
                {"display": "none"},
            )
        try:
            from ultralytics import YOLO  # type: ignore

            weights_path = Path(weights.strip()).expanduser()
            model = YOLO(str(weights_path))
            results = model.predict(source=str(image_path), conf=float(confidence), verbose=False)
            result = results[0]

            # Em segmentação, `masks` representa as instâncias segmentadas. Se
            # não houver máscara, usa boxes como fallback de contagem.
            crack_count = len(result.masks.data) if result.masks is not None else len(result.boxes)
            area_percent = "-"
            if result.masks is not None:
                mask_pixels = float(result.masks.data.sum().item())
                total_pixels = float(result.masks.data.shape[-1] * result.masks.data.shape[-2])
                area_percent = f"{100.0 * mask_pixels / max(total_pixels, 1.0):.2f}%"
            status = (
                f"Solução 1: segmentação YOLO concluída com {weights_path.name} "
                f"e confiança {float(confidence):.2f}."
            )
            if crack_count == 0:
                status = (
                    f"Nenhuma fissura acima da confiança {float(confidence):.2f}. "
                    "Verifique se está usando o best.pt treinado ou reduza a confiança."
                )

            return (
                "YOLO",
                str(crack_count),
                area_percent,
                status,
                array_to_data_url(cv2_mod, result.plot()),
                {"display": "none"},
            )
        except Exception as exc:  # noqa: BLE001
            return (
                "YOLO",
                "-",
                "-",
                f"Falha ao executar YOLO: {exc}",
                array_to_data_url(cv2_mod, image),
                {"display": "none"},
            )


@app.callback(
    Output("unet-engine-label", "children"),
    Output("unet-crack-count", "children"),
    Output("unet-area-percent", "children"),
    Output("unet-status-label", "children"),
    Output("unet-annotated-image", "src"),
    Output("unet-empty-state", "style"),
    Input("unet-upload-image", "contents"),
    Input("unet-weights", "value"),
    State("unet-upload-image", "filename"),
)
def run_unet_preview(contents, weights, filename):
    """Prévia da página U-Net enquanto a inferência final não está conectada."""

    if not contents:
        raise PreventUpdate
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        safe_name = Path(filename or "upload.jpg").name
        image_path = temp_path / safe_name
        image_path.write_bytes(uploaded_bytes(contents))
        cv2_mod, _ = require_cv2()
        image = cv2_mod.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read uploaded image: {safe_name}")
        status = "Solução 2: treino e máscaras preparados; inferência visual fica como próximo passo."
        if weights and weights.strip():
            status = f"Modelo informado: {weights}. Ainda falta conectar a rotina de inferência U-Net."
        return "U-Net", "-", "-", status, array_to_data_url(cv2_mod, image), {"display": "none"}


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "8062")))
