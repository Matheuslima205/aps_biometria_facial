# Identificação Biométrica Facial

Protótipo acadêmico de reconhecimento facial desenvolvido em Python para a disciplina de Processamento Digital de Imagens.

O sistema utiliza técnicas de processamento de imagens e o modelo pré-treinado FaceNet para identificar pessoas cadastradas, rejeitar indivíduos desconhecidos e validar entradas nas quais nenhum rosto é detectado.

## Tecnologias utilizadas

- **Python** — linguagem principal.
- **OpenCV** — processamento de imagens e detecção facial.
- **FaceNet** — extração de características faciais (*embeddings*).
- **NumPy** — manipulação de matrizes e vetores.
- **Gradio** — interface web.
- **Docker** — ambiente de execução isolado.

## Funcionalidades

- Identificação facial de pessoas cadastradas.
- Cadastro de pessoas a partir de fotografias.
- Comparação facial utilizando similaridade de cosseno.
- Controle de acesso baseado em níveis.
- Rejeição de pessoas não cadastradas.
- Identificação de imagens sem rostos detectáveis.
- Visualização das etapas de processamento da imagem.
- Avaliação dos resultados do reconhecimento.

## Estrutura do projeto

```text
aps-biometria/
├── data/
├── docs/
├── outputs/
├── src/
│   ├── app.py
│   ├── enroll.py
│   ├── evaluate.py
│   ├── features.py
│   ├── pipeline.py
│   ├── prepare_dataset.py
│   └── recognition.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-docker.txt
```

## Requisitos

Para executar o projeto localmente:

- Python instalado.
- Git (opcional).
- Conexão com a internet na primeira execução, para baixar os modelos necessários.

Para executar com Docker:

- Docker instalado.
- Docker Compose disponível.

## Execução local (Python)

Clone o repositório:

```bash
git clone URL_DO_REPOSITORIO
cd aps-biometria
```

Crie um ambiente virtual:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
python src/app.py
```

Após a inicialização, acesse pelo navegador:

```text
http://127.0.0.1:7860
```

## Execução com Docker

Com o Docker instalado e em execução, acesse a pasta raiz do projeto e execute:

```bash
docker compose up --build
```

O comando realiza a construção da imagem e inicia o contêiner da aplicação.

Após a inicialização, acesse:

```text
http://localhost:7860
```

Para encerrar a aplicação, utilize:

```bash
docker compose down
```

## Observações

- Na primeira execução, o download dos modelos de reconhecimento pode levar alguns minutos.
- Os arquivos de cadastro, banco de características e informações dos usuários são armazenados localmente.
- O desempenho pode variar conforme o hardware utilizado.
- O projeto possui finalidade exclusivamente acadêmica e demonstrativa. Não deve ser utilizado como sistema real de autenticação ou controle de acesso.

## Contexto acadêmico

Projeto desenvolvido como parte da Atividade Prática Supervisionada (APS) do curso de Ciência da Computação.

**Objetivo:** aplicar conceitos de Processamento Digital de Imagens e reconhecimento biométrico facial por meio da integração de bibliotecas e modelos pré-treinados.