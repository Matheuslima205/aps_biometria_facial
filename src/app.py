import json
import re
import unicodedata
from pathlib import Path

import cv2
import gradio as gr

from evaluate import DETAIL_HEADERS, SUMMARY_HEADERS, evaluate
from recognition import THRESHOLD, identify
from enroll import build_database
from pipeline import acquire, detect_face, preprocess

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = [[str(p)] for p in sorted((ROOT / "data" / "test").glob("*.jpg"))]
STEP_HEIGHT = "calc((100vh - 190px) / 2)"
INPUT_HEIGHT = "20vh"

DARK_MODE = """
() => {
    if (!document.body.classList.contains('dark')) {
        document.body.classList.add('dark');
    }
}
"""

CSS = """
html, body { height: 100%; overflow: hidden !important; }
body, .gradio-container, .dark .gradio-container {
    background-color: #141414 !important;
    background-image: radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px) !important;
    background-size: 14px 14px !important;
}
.gradio-container { padding: 12px 16px 0 16px !important; }
#app { max-width: 1320px; margin: 0 auto; gap: 8px; }
#header { padding: 0; }
#header h2 { margin: 0 0 2px 0; }
#header p { margin: 0; opacity: 0.75; font-size: 0.9rem; }
#main-row { height: calc(100vh - 92px); align-items: center; gap: 20px; }
#panel {
    background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
    border-radius: 12px; padding: 14px; gap: 12px;
    height: calc(100vh - 176px); overflow: hidden; align-self: center;
    display: flex; flex-direction: column; flex-wrap: nowrap;
}
#panel > * { flex-shrink: 0; }
#panel > #status-wrap { flex: 1 1 auto; min-height: 0; display: flex; }
#status-wrap * { box-sizing: border-box; }
#status-wrap > div, #status-wrap > div > div { flex: 1 1 auto; display: flex; min-height: 0; width: 100%; }
#status-wrap .status { flex: 1; height: 100%; overflow-y: auto; box-sizing: border-box; }
#panel .block { box-shadow: none; }
#steps { align-self: center; justify-content: center; gap: 14px; }
#steps .row { gap: 14px; }
.step-img img, .step-img canvas { object-fit: contain !important; }
#panel .gallery { display: flex !important; flex-wrap: wrap; gap: 6px; }
#panel .gallery-item { padding: 0 !important; width: 54px !important; min-width: 0 !important; height: 54px; overflow: hidden; }
#panel .gallery-item img { width: 52px !important; height: 52px !important; object-fit: cover; }
.status { padding: 0; border-radius: var(--block-radius); border: 1px solid var(--border-color-primary);
          background: var(--block-background-fill); position: relative; }
.status-label { display: inline-flex; align-items: center; gap: 6px; font-size: var(--block-label-text-size);
                color: var(--block-label-text-color); background: var(--block-label-background-fill);
                border-right: 1px solid var(--border-color-primary); border-bottom: 1px solid var(--border-color-primary);
                border-radius: 0 0 var(--radius-sm) 0; padding: 4px 10px; }
.status-body { padding: 8px 14px 10px 14px; }
.status-title { font-weight: 600; font-size: 0.95rem; margin: 0 0 6px 0; }
.status.ok .status-title { color: #4ade80; }
.status.denied .status-title { color: #f87171; }
.status.invalid .status-title, .status.busy .status-title { color: var(--color-accent); }
.status.idle .status-title { color: var(--body-text-color-subdued); font-weight: 500; }
.status table, .status tbody, .status tr, .status td { border: none !important; background: transparent !important; box-shadow: none !important; }
.status table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 0 !important; }
.status td { padding: 3px 0 !important; vertical-align: top; }
.status td:first-child { color: var(--body-text-color-subdued); width: 42%; }
.status ul { margin: 4px 0 0 16px; padding: 0; font-size: 0.85rem; }
.status hr { border: none; border-top: 1px solid var(--border-color-primary); margin: 8px 0 6px 0; }
.status li { margin: 2px 0; line-height: 1.3; }
.status .muted { color: var(--body-text-color-subdued); font-size: 0.82rem; margin: 0; }
#run-btn { max-width: 100%; }
#status-wrap, #status-wrap > div { padding: 0 !important; margin: 0 !important; }
.eta-bar, .progress-text { display: none !important; }
#header-row { align-items: center; }
#eval-btn { max-width: 160px; margin-left: auto; }
#eval-modal {
    position: fixed !important; top: 50%; left: 50%; transform: translate(-50%, -50%);
    width: min(920px, 92vw); max-height: 86vh; overflow-y: auto; z-index: 1000;
    background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
    border-radius: 12px; padding: 18px 20px; gap: 12px;
    box-shadow: 0 0 0 100vmax rgba(0, 0, 0, 0.65);
}
#eval-modal-head { align-items: center; }
#eval-modal-head h3 { margin: 0; }
#eval-close { min-width: 32px !important; max-width: 32px; height: 32px; padding: 0 !important; margin-left: auto;
              font-size: 1rem; background: transparent; border: none; box-shadow: none; }
#eval-close:hover { background: var(--button-secondary-background-fill-hover); }
#eval-modal-head > * { flex-grow: 0; }
#eval-modal-head > :first-child { flex-grow: 1; }
#eval-acc p { margin: 0; font-size: 0.95rem; }

/* Cadastro: upload compacto sem cortar as instrucoes */
#person-photos-upload {
    height: 104px !important;
    min-height: 104px !important;
    overflow: visible !important;
}
#person-photos-upload > div,
#person-photos-upload .wrap,
#person-photos-upload .file-upload {
    box-sizing: border-box !important;
    min-height: 0 !important;
}
#person-photos-upload .upload-container,
#person-photos-upload [data-testid="file-upload"],
#person-photos-upload .file-upload {
    height: 76px !important;
    min-height: 76px !important;
    padding: 5px 8px !important;
    gap: 4px !important;
    box-sizing: border-box !important;
}
#person-photos-upload svg {
    width: 19px !important;
    height: 19px !important;
}
#person-photos-upload p,
#person-photos-upload span {
    font-size: 12px !important;
    line-height: 1.2 !important;
    margin-top: 1px !important;
    margin-bottom: 1px !important;
}
"""

LEVEL_LABELS = {
    1: "Informações gerais e consultas públicas",
    2: "Fiscalização e ocorrências ambientais",
    3: "Informações estratégicas e decisões",
}


def box(kind, title, body):
    return (f'<div class="status {kind}"><div class="status-label">Resultado</div>'
            f'<div class="status-body"><p class="status-title">{title}</p>{body}</div></div>')


def rows(pairs):
    return "<table>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in pairs) + "</table>"


IDLE = box("idle", "Aguardando imagem", '<p class="muted">Envie uma foto ou capture uma imagem pela webcam.</p>')
BUSY = box("busy", "Processando...", '<p class="muted">Executando o pipeline de reconhecimento.</p>')


def to_rgb(img):
    if img is None:
        return None
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def status_box(r):
    if r["status"] == "invalid":
        return box("invalid", "Entrada inválida",
                   rows([("Rosto detectado", "Não"), ("Acesso", "Negado")]))

    if r["status"] == "unknown":
        return box("denied", "Acesso negado",
                   rows([("Usuário", "Não reconhecido"),
                         ("Mais parecido", r["user"]),
                         ("Similaridade", f'{r["score"]:.3f} (limiar {THRESHOLD:.2f})')]))

    items = "".join(
        f"<li>Nível {n} - {LEVEL_LABELS[n]}</li>"
        for n in range(1, r["level"] + 1)
    )

    return box("ok", "Acesso liberado",
               rows([("Usuário", f'<b>{r["user"]}</b>'),
                     ("Similaridade", f'{r["score"]:.3f} (limiar {THRESHOLD:.2f})'),
                     ("Nível de acesso", f'<b>{r["level"]}</b>')])
               + f'<hr><p class="muted">Informações liberadas (fictícias)</p><ul>{items}</ul>')


def run_evaluation():
    summary, details, accuracy = evaluate()
    return f"Acurácia global: **{accuracy:.1%}** ({summary[-1][2]} de {summary[-1][1]} testes)", summary, details


def run(path):
    """
    Recebe o caminho de uma imagem enviada ou capturada pela webcam
    e executa o mesmo processo de identificação facial.
    """
    if path is None:
        return None, None, None, None, IDLE

    result = identify(path)

    return (
        to_rgb(result["original"]),
        to_rgb(result["preprocessed"]),
        to_rgb(result["detection"]),
        to_rgb(result["roi"]),
        status_box(result),
    )


def _person_key(name):
    """Create a stable folder/database key from a display name."""
    normalized = unicodedata.normalize("NFKD", name.strip().lower())
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")


def register_person(name, photos, consent, access_level):
    """
    Registra fotografias autorizadas e atualiza o nível de acesso
    da pessoa no users.json.
    """
    if not consent:
        return "Para continuar, confirme que tem autorização para usar as fotos."

    if not name or not name.strip():
        return "Informe o nome da pessoa."

    if not photos:
        return "Selecione pelo menos uma fotografia."

    try:
        access_level = int(access_level)
    except (TypeError, ValueError):
        return "Selecione um nível de acesso válido."

    if access_level not in LEVEL_LABELS:
        return "O nível de acesso deve ser 1, 2 ou 3."

    key = _person_key(name)

    if not key:
        return "O nome informado não gerou um identificador válido. Tente outro nome."

    enrollment_dir = ROOT / "data" / "enroll" / key
    enrollment_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        p for p in enrollment_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    next_index = len(existing) + 1
    saved_count = 0
    rejected = []

    if isinstance(photos, (str, Path)):
        photos = [photos]

    for source in photos:
        image = cv2.imread(str(source))

        if image is None:
            rejected.append(Path(str(source)).name + " (arquivo não pôde ser lido)")
            continue

        # Aplica o mesmo detector utilizado no pipeline de reconhecimento.
        _, gray = acquire(str(source))
        roi, _ = detect_face(preprocess(gray))

        if roi is None:
            rejected.append(Path(str(source)).name + " (nenhum rosto detectado)")
            continue

        target = enrollment_dir / f"{next_index:03d}.jpg"

        while target.exists():
            next_index += 1
            target = enrollment_dir / f"{next_index:03d}.jpg"

        if not cv2.imwrite(str(target), image):
            rejected.append(Path(str(source)).name + " (não foi possível salvar)")
            continue

        saved_count += 1
        next_index += 1

    if saved_count == 0:
        if not any(enrollment_dir.iterdir()):
            enrollment_dir.rmdir()

        return "Nenhuma foto foi cadastrada. Envie imagens nítidas com um rosto visível."

    users_path = ROOT / "users.json"

    try:
        with users_path.open(encoding="utf-8") as file:
            users = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        users = {}

    # Atualiza o nível de acesso, inclusive para pessoas já cadastradas.
    users[key] = access_level

    with users_path.open("w", encoding="utf-8") as file:
        json.dump(users, file, ensure_ascii=False, indent=2)

    # Reconstrói o banco de embeddings FaceNet após o cadastro.
    registered_users = build_database("facenet")

    message = (
        f"**Cadastro concluído:** {name.strip()} — "
        f"{saved_count} foto(s) adicionada(s).\n\n"
        f"**Nível de acesso:** {access_level} — {LEVEL_LABELS[access_level]}.\n\n"
        f"Banco FaceNet atualizado ({registered_users} pessoa(s))."
    )

    if rejected:
        message += "\n\n**Fotos ignoradas:** " + "; ".join(rejected)

    return message


def step_image(label):
    return gr.Image(
        label=label,
        height=STEP_HEIGHT,
        interactive=False,
        buttons=["fullscreen"],
        elem_classes="step-img",
    )


with gr.Blocks(title="Identificação Biométrica Facial", fill_width=True) as demo:
    with gr.Column(elem_id="app"):
        with gr.Row(elem_id="header-row"):
            gr.Markdown(
                "## Identificação Biométrica Facial\n"
                "Secretaria de Monitoramento Ambiental — protótipo acadêmico com dados fictícios",
                elem_id="header",
            )

            eval_btn = gr.Button(
                "Avaliação",
                variant="secondary",
                elem_id="eval-btn",
            )

        with gr.Row(equal_height=False, elem_id="main-row"):
            with gr.Column(scale=1, min_width=340, elem_id="panel"):

                # Agora permite selecionar uma imagem ou capturar pela webcam.
                inp = gr.Image(
                    type="filepath",
                    sources=["upload", "webcam"],
                    label="Imagem de entrada",
                    height=INPUT_HEIGHT,
                    buttons=["fullscreen"],
                    elem_classes="step-img",
                )

                btn = gr.Button(
                    "Identificar",
                    variant="primary",
                    elem_id="run-btn",
                )

                status = gr.HTML(
                    IDLE,
                    padding=False,
                    container=False,
                    elem_id="status-wrap",
                )

                with gr.Accordion("Cadastrar pessoa", open=False):
                    person_name = gr.Textbox(
                        label="Nome da pessoa",
                    )

                    # Seleção do nível de acesso durante o cadastro.
                    access_level = gr.Dropdown(
                        choices=[
                            ("Nível 1 - Informações gerais", 1),
                            ("Nível 2 - Fiscalização e ocorrências", 2),
                            ("Nível 3 - Informações estratégicas", 3),
                        ],
                        value=1,
                        type="value",
                        label="Nível de acesso",
                    )

                    person_photos = gr.File(
                        label="Fotografias (uma ou mais)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                        height=104,
                        elem_id="person-photos-upload",
                    )

                    consent = gr.Checkbox(
                        label="Tenho autorização para utilizar estas fotos neste protótipo acadêmico.",
                        value=False,
                    )

                    register_btn = gr.Button(
                        "Cadastrar / atualizar pessoa",
                        variant="secondary",
                    )

                    register_status = gr.Markdown()

                    register_btn.click(
                        register_person,
                        inputs=[
                            person_name,
                            person_photos,
                            consent,
                            access_level,
                        ],
                        outputs=register_status,
                        show_progress="minimal",
                    )

                gr.Examples(
                    EXAMPLES,
                    inputs=inp,
                    label="Imagens de teste",
                    examples_per_page=18,
                )

            with gr.Column(scale=2, elem_id="steps"):
                with gr.Row(equal_height=True):
                    out1 = step_image("1. Original")
                    out2 = step_image("2. Pré-processada")

                with gr.Row(equal_height=True):
                    out3 = step_image("3. Rosto detectado")
                    out4 = step_image("4. ROI (160x160)")

        with gr.Column(visible=False, elem_id="eval-modal") as modal:
            with gr.Row(elem_id="eval-modal-head"):
                gr.Markdown("### Avaliação de desempenho")

                close_btn = gr.Button(
                    "✕",
                    size="sm",
                    variant="secondary",
                    elem_id="eval-close",
                )

            acc = gr.Markdown(
                "Executando os testes...",
                elem_id="eval-acc",
            )

            summary_df = gr.Dataframe(
                headers=SUMMARY_HEADERS,
                interactive=False,
                label="Resumo por cenário",
            )

            details_df = gr.Dataframe(
                headers=DETAIL_HEADERS,
                interactive=False,
                label="Resultado por imagem",
                max_height=320,
            )

        eval_btn.click(
            lambda: (gr.Column(visible=True), "Executando os testes..."),
            outputs=[modal, acc],
            show_progress="hidden",
        ).then(
            run_evaluation,
            outputs=[acc, summary_df, details_df],
        )

        close_btn.click(
            lambda: gr.Column(visible=False),
            outputs=modal,
            show_progress="hidden",
        )

        btn.click(
            lambda: BUSY,
            outputs=status,
            show_progress="hidden",
        ).then(
            run,
            inputs=inp,
            outputs=[out1, out2, out3, out4, status],
            show_progress_on=[out1, out2, out3, out4],
        )


if __name__ == "__main__":
    demo.launch(
        css=CSS,
        js=DARK_MODE,
        footer_links=[],
    )