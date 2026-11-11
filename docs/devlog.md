# Diário de desenvolvimento — APS Biometria (PDI/VC)

Registro do que foi feito, das decisões técnicas e dos testes, para usar na escrita do trabalho.
Cada item indica a seção do documento onde o conteúdo pode entrar.

---

## 1. Definições do projeto

| Item | Escolha | Justificativa | Seção do documento |
|---|---|---|---|
| Modalidade | Reconhecimento facial | Bases públicas disponíveis e detecção facial pronta no OpenCV | Objetivo / Plano |
| Finalidade | Identificação 1:N | O sistema compara a amostra com todos os cadastrados e descobre a identidade, ou rejeita | Objetivo / Fundamentos |
| Linguagem | Python | Recomendação principal do professor | Plano |
| Interface | Gradio (Hugging Face) | Interface web local, roda só no computador do grupo | Plano |
| Base de imagens | LFW (pública) | Base autorizada para pesquisa, sem uso de imagens próprias | Plano / Ética |
| Armazenamento | Pastas + `users.json` | Arquivos locais, como o documento exige | Projeto do programa |

### Bibliotecas

| Biblioteca | Papel no projeto |
|---|---|
| OpenCV (4.x) | Leitura, conversão, filtros, CLAHE, detecção facial (Haar Cascade) |
| scikit-image | Imagens de exemplo sem rosto (entradas inválidas); LBP (opcional) |
| scikit-learn | Download da LFW; métricas de avaliação |
| NumPy | Manipulação das imagens como matrizes |
| PyTorch + Transformers | Executar o modelo DINOv2 do Hugging Face |
| facenet-pytorch + torchvision | Modelo FaceNet (embedding facial) |
| Gradio | Interface do protótipo |

---

## 2. Ambiente (Etapa 1)

- Windows + VS Code + PowerShell, Python 3.14, ambiente virtual `.venv`.
- Problema 1: erro de digitação no `requirements.txt` (`tourch` em vez de `torch`). Corrigido.
- Problema 2: o pip instalou o OpenCV 5.0, que tirou o detector Haar Cascade (`CascadeClassifier`) do pacote principal e não traz os arquivos `.xml` dos classificadores. Solução: fixar `opencv-python<5` no `requirements.txt` (versão 4.14). Vale citar no documento como limitação/decisão de ferramenta.

Estrutura de pastas:

```
aps-biometria/
├── data/enroll/     fotos de cadastro (1 pasta por usuário)
├── data/test/       fotos de teste
├── src/             código-fonte
├── outputs/         imagens geradas pelo pipeline
├── docs/            este diário
├── users.json       usuário → nível de acesso
└── requirements.txt
```

---

## 3. Base de imagens (Etapa 2) — `src/prepare_dataset.py`

- **Base:** LFW (*Labeled Faces in the Wild*), baixada pelo `fetch_lfw_people` do scikit-learn.
- **Filtro:** apenas pessoas com 70 fotos ou mais (7 pessoas).
- **Divisão** (semente fixa `42`, para resultados reproduzíveis):
  - 5 pessoas cadastradas: 5 fotos de cadastro + 2 fotos de teste cada;
  - 2 pessoas não cadastradas: 4 fotos de teste (`unknown_*`), cenário "usuário diferente";
  - 4 imagens sem rosto (`invalid_*`): xícara, gato, imagem preta e ruído, cenário "entrada inválida".
- **Níveis de acesso:** 3, 2, 1, 1, 2, definidos no `users.json`.
- **Decisão técnica:** usar a imagem completa da LFW (250×250, colorida) em vez do recorte de rosto que a base já oferece. Assim a localização do rosto é feita pela etapa de detecção do próprio sistema.

> Preencher após rodar: nomes das pessoas cadastradas e níveis →

---

## 4. Pipeline de PDI (Etapa 3) — `src/pipeline.py`

### 4A. Aquisição — `acquire()`

1. Leitura com `cv2.imread`. A imagem vira uma matriz BGR com formato `(altura, largura, 3)`.
2. Redimensionamento para largura fixa de 400 px, mantendo a proporção. Isso padroniza entradas de tamanhos diferentes.
3. Conversão para escala de cinza (`cv2.cvtColor`): de 3 canais para 1 canal de intensidade (0 = preto, 255 = branco).

**Teste:** colorida `(266, 400, 3)` → cinza `(266, 400)`.
**Conceito:** o "3" é o número de **canais** (Azul, Verde, Vermelho). A imagem em cinza tem 1/3 dos dados e mantém a informação de claro/escuro, que é a que a detecção facial usa.

### 4B. Pré-processamento — `preprocess()`

| Técnica | Tipo exigido | Parâmetros | Efeito |
|---|---|---|---|
| Filtro Gaussiano | Filtragem espacial | Kernel 5×5, sigma automático | Cada pixel vira a média ponderada dos vizinhos, o que reduz ruído e detalhes pequenos |
| CLAHE | Melhoria / transformação | `clipLimit=2.0`, blocos 8×8 | Equalização de histograma por regiões; aumenta o contraste local sem estourar o ruído |

**Ordem:** filtro → CLAHE. Se o CLAHE viesse primeiro, ele realçaria também o ruído.

#### Experimento 1: variação do `clipLimit` do CLAHE

| clipLimit | Observação |
|---|---|
| 2.0 (padrão) | Contraste melhorado de forma natural: olhos, óculos, boca e contorno do rosto mais definidos, fundo continua liso e com pouco ruído |
| 8.0 | Contraste muito forte: textura da pele e do fundo exageradas, aspecto "desenhado/metálico", ruído do fundo realçado |

**Conclusão parcial:** um valor alto realça detalhes que não fazem parte do rosto (fundo, textura, ruído). O valor 2.0 foi mantido como padrão. _(confirmar depois do teste de detecção: o valor alto aumenta falsas detecções?)_

Imagens geradas: `outputs/1_original.jpg`, `outputs/2_gray.jpg`, `outputs/3_preprocessed.jpg`.

### 4C. Detecção do rosto (ROI) — `detect_face()`

- **Técnica:** classificador em cascata de Haar (Viola & Jones, 2001), modelo pré-treinado `haarcascade_frontalface_default.xml`, que vem com o OpenCV.
- **Como funciona:** uma janela percorre a imagem em várias escalas e calcula *Haar features*, que são diferenças de brilho entre retângulos vizinhos (ex.: a região dos olhos é mais escura que a das bochechas). Uma "cascata" de classificadores rápidos descarta cedo as regiões que claramente não são rosto, e só as candidatas passam pelos estágios seguintes.
- **Entrada:** a imagem pré-processada (cinza + Gaussiano + CLAHE).

| Parâmetro | Valor | Significado |
|---|---|---|
| `scaleFactor` | 1.1 | A cada nível da pirâmide a imagem diminui 10% (detecta rostos de vários tamanhos) |
| `minNeighbors` | 5 | Detecções sobrepostas necessárias para confirmar um rosto; maior = menos falsos positivos |
| `minSize` | 60×60 | Ignora "rostos" muito pequenos, geralmente falsos positivos |

- Se houver mais de um rosto, fica o **maior** (provavelmente o usuário em primeiro plano).
- A ROI é recortada e padronizada em **160×160 px**, para que todas as amostras tenham o mesmo tamanho na extração de características.
- Se nenhum rosto for encontrado, a função retorna `None`. É esse o caminho do cenário **entrada inválida**.

**Teste inicial (imagens de exemplo):** rosto detectado na imagem com pessoa; nenhuma detecção em gato, xícara e ruído aleatório.

Imagens geradas: `outputs/4_detection.jpg` (retângulo verde sobre o rosto) e `outputs/5_roi.jpg` (rosto recortado).

**Teste na LFW:** `colinpowell_01.jpg` → rosto detectado em `(x=98, y=105, w=188, h=188)`. Em uma imagem de 400 px de largura, o rosto ocupa quase metade, o que é coerente com o enquadramento da LFW.

### 4D. Extração de características — `src/features.py`

Dois métodos foram implementados para comparação (discussão técnica):

| Método | Tipo | Como funciona | Tamanho do vetor |
|---|---|---|---|
| **LBP** (*Local Binary Patterns*) | PDI clássico (scikit-image) | Cada pixel é comparado com 8 vizinhos e vira um código binário que descreve a textura local. A face é dividida em 4×4 blocos; o histograma de cada bloco é concatenado | 160 |
| **DINOv2** (`facebook/dinov2-small`, Hugging Face) | Deep learning (Vision Transformer pré-treinado) | A ROI passa pela rede, e o token CLS da última camada é usado como *embedding*, um vetor que resume a imagem | 384 |

- Os vetores são **normalizados (norma L2 = 1)**, então a similaridade de cosseno vira um simples produto escalar.
- **Similaridade de cosseno:** mede o ângulo entre dois vetores. 1 = mesma direção (muito parecidos), 0 = sem relação.
- O DINOv2 é um modelo de visão **genérico**, não treinado especificamente para rostos (como FaceNet/ArcFace). Isso é uma limitação a discutir no documento.

### 4E. Cadastro (*enrollment*) — `src/enroll.py`

- Cada foto de `data/enroll/<usuário>/` passa pelo pipeline completo: aquisição → pré-processamento → detecção → extração.
- O **template** do usuário é a **média** dos vetores das suas fotos, o que é mais robusto que usar uma foto só.
- Fotos sem rosto detectado são descartadas, e o script avisa.
- O resultado é salvo em `data/db_<método>.npz`, um "banco vetorial" simples em arquivo local (nomes + templates).

**Resultado do cadastro (igual para os 3 métodos, pois a detecção é a mesma):**

| Usuário | Fotos aproveitadas |
|---|---|
| colinpowell | 5 de 5 |
| donaldrumsfeld | 5 de 5 |
| georgewbush | 4 de 5 (`3.jpg`: nenhum rosto detectado pelo Haar) |
| gerhardschroeder | 5 de 5 |
| tonyblair | 5 de 5 |

### 4F. Problemas encontrados na implementação

| Problema | Sintoma | Causa | Correção |
|---|---|---|---|
| `enroll.py` não fazia nada | Nenhuma saída e `db_lbp.npz` não criado | Faltava o bloco `if __name__ == "__main__":` que chama `build_database()` | Bloco adicionado |
| Vetor LBP com tamanho 10 | Todos os scores entre 0.995 e 1.000 | `grid=1`: histograma do rosto inteiro, sem informação espacial | `grid=4` (4×4 blocos → 160 valores) |
| Erro no DINOv2 | `Unable to infer channel dimension format` | `np.stack(..., axis=1)` colocou os canais no lugar errado | Conversão com `Image.fromarray(roi).convert("RGB")` |
| Atributo inexistente | `last_hidden_stats` | Erro de digitação | `last_hidden_state` |

**Observação para o documento:** com `grid=1` o LBP perde a informação de **onde** cada textura está no rosto. O resultado foi praticamente o mesmo vetor para todas as pessoas (scores ≈ 1.0), o que mostra a importância da divisão espacial em blocos.

### 4G. Experimento 2: comparação dos extratores (LBP × DINOv2)

Teste com 10 imagens de usuários cadastrados, 4 de pessoas não cadastradas e 4 imagens sem rosto.

**LBP (vetor 160):**

| Imagem | Melhor candidato | Score | Correto? |
|---|---|---|---|
| colinpowell_01 | colinpowell | 0.993 | ✔ |
| colinpowell_02 | georgewbush | 0.994 | ✘ |
| donaldrumsfeld_01 | tonyblair | 0.989 | ✘ |
| donaldrumsfeld_02 | gerhardschroeder | 0.990 | ✘ |
| georgewbush_01 | tonyblair | 0.992 | ✘ |
| georgewbush_02 | tonyblair | 0.993 | ✘ |
| gerhardschroeder_01 | georgewbush | 0.971 | ✘ |
| gerhardschroeder_02 | georgewbush | 0.969 | ✘ |
| tonyblair_01 | tonyblair | 0.991 | ✔ |
| tonyblair_02 | tonyblair | 0.993 | ✔ |
| unknown_01 | georgewbush | 0.993 | — |
| unknown_02 | donaldrumsfeld | 0.994 | — |
| unknown_03 | tonyblair | 0.994 | — |
| unknown_04 | tonyblair | 0.996 | — |

**DINOv2 (vetor 384):**

| Imagem | Melhor candidato | Score | Correto? |
|---|---|---|---|
| colinpowell_01 | gerhardschroeder | 0.844 | ✘ |
| colinpowell_02 | colinpowell | 0.795 | ✔ |
| donaldrumsfeld_01 | colinpowell | 0.814 | ✘ |
| donaldrumsfeld_02 | gerhardschroeder | 0.798 | ✘ |
| georgewbush_01 | georgewbush | 0.729 | ✔ |
| georgewbush_02 | tonyblair | 0.810 | ✘ |
| gerhardschroeder_01 | gerhardschroeder | 0.765 | ✔ |
| gerhardschroeder_02 | donaldrumsfeld | 0.826 | ✘ |
| tonyblair_01 | tonyblair | 0.792 | ✔ |
| tonyblair_02 | tonyblair | 0.820 | ✔ |
| unknown_01 | tonyblair | 0.833 | — |
| unknown_02 | georgewbush | 0.732 | — |
| unknown_03 | donaldrumsfeld | 0.703 | — |
| unknown_04 | colinpowell | 0.869 | — |

As 4 imagens inválidas foram rejeitadas na detecção pelos dois métodos (nenhum rosto detectado).

**Resumo:**

| Método | Acertos (cadastrados) | Faixa de score (cadastrados) | Faixa de score (desconhecidos) |
|---|---|---|---|
| LBP | 3/10 (30%) | 0.969 – 0.994 | 0.993 – 0.996 |
| DINOv2 | 5/10 (50%) | 0.729 – 0.844 | 0.703 – 0.869 |

**Análise:**
- Os dois métodos ficaram muito abaixo do esperado. A chance de acertar "no chute" entre 5 pessoas é de 20%.
- Pior: os **desconhecidos tiveram scores iguais ou maiores** que os cadastrados. Assim, **nenhum limiar** consegue separar "é a pessoa" de "não é", e o sistema aceitaria impostores.
- **Por que o LBP falhou:** ele descreve textura local (pele, cabelo, fundo), que é parecida entre pessoas diferentes. A LFW tem grande variação de pose, iluminação e expressão entre as fotos da mesma pessoa, o que pesa mais que a diferença entre pessoas.
- **Por que o DINOv2 falhou:** é um modelo de visão **genérico**, treinado para distinguir categorias de objetos (rosto × carro × animal), e não indivíduos. Para ele, rostos de homens adultos de terno são muito parecidos.
- **Conclusão:** para identificar *indivíduos*, é necessário um modelo de embedding **treinado especificamente para rostos**, como os citados na proposta da APS (FaceNet, ArcFace, InsightFace).

### 4H. Decisão técnica: adoção do FaceNet

- **Modelo:** FaceNet (Schroff; Kalenichenko; Philbin, 2015), arquitetura Inception-ResNet-v1, pré-treinado na base VGGFace2. Biblioteca `facenet-pytorch`.
- **Por quê:** sugerido explicitamente na proposta da APS. Foi treinado com *triplet loss* para que fotos da **mesma pessoa** fiquem próximas e de **pessoas diferentes** fiquem distantes no espaço vetorial, que é exatamente o que faltou ao LBP e ao DINOv2.
- **Vetor:** 512 valores (já normalizado).
- **Entrada:** ROI 160×160 do nosso pipeline, com pixels normalizados para a faixa [-1, 1] ((pixel − 127.5) / 128).
- LBP e DINOv2 foram **mantidos** no código, para a comparação no documento.

#### Experimento 3: resultado do FaceNet

| Imagem | Melhor candidato | Score | Correto? |
|---|---|---|---|
| colinpowell_01 | colinpowell | 0.874 | ✔ |
| colinpowell_02 | colinpowell | 0.847 | ✔ |
| donaldrumsfeld_01 | donaldrumsfeld | 0.831 | ✔ |
| donaldrumsfeld_02 | donaldrumsfeld | 0.731 | ✔ |
| georgewbush_01 | georgewbush | 0.856 | ✔ |
| georgewbush_02 | georgewbush | 0.897 | ✔ |
| gerhardschroeder_01 | gerhardschroeder | 0.796 | ✔ |
| gerhardschroeder_02 | gerhardschroeder | 0.832 | ✔ |
| tonyblair_01 | tonyblair | 0.708 | ✔ |
| tonyblair_02 | tonyblair | 0.819 | ✔ |
| unknown_01 | georgewbush | 0.239 | — |
| unknown_02 | georgewbush | 0.304 | — |
| unknown_03 | colinpowell | 0.186 | — |
| unknown_04 | colinpowell | 0.302 | — |

Inválidas: as 4 rejeitadas na detecção (nenhum rosto).

**Comparação final dos 3 extratores:**

| Método | Tipo | Vetor | Acertos (cadastrados) | Score cadastrados | Score desconhecidos | Separáveis por limiar? |
|---|---|---|---|---|---|---|
| LBP | PDI clássico | 160 | 3/10 (30%) | 0.969 – 0.994 | 0.993 – 0.996 | Não |
| DINOv2 | Deep, genérico | 384 | 5/10 (50%) | 0.729 – 0.844 | 0.703 – 0.869 | Não |
| **FaceNet** | **Deep, facial** | **512** | **10/10 (100%)** | **0.708 – 0.897** | **0.186 – 0.304** | **Sim** |

**Análise:** com o FaceNet, além de acertar todos os cadastrados, existe uma **margem clara** entre os scores: o menor score de um cadastrado (0.708) fica muito acima do maior score de um desconhecido (0.304). É isso que permite definir um limiar de decisão confiável. O modelo treinado especificamente para rostos resolveu o problema que o LBP e o DINOv2 não resolveram.

**Método adotado no protótipo: FaceNet.**

### 4I. Regra de decisão (identificação 1:N)

1. A amostra é comparada com **todos** os templates cadastrados (similaridade de cosseno).
2. O usuário com **maior score** é o candidato.
3. Se o score for **≥ 0.50** (limiar), o usuário é **reconhecido** e recebe seu nível de acesso; se for menor, o resultado é **"não reconhecido"** (acesso negado).
4. Se nenhum rosto for detectado, o resultado é **"entrada inválida"**.

**Escolha do limiar (0.50):** os scores observados ficaram em 0.186–0.304 para desconhecidos e 0.708–0.897 para cadastrados. O valor 0.50 fica aproximadamente no meio dessa faixa vazia, dando folga dos dois lados: uma foto de cadastrado um pouco pior ainda passa, e um desconhecido um pouco mais parecido ainda é barrado.

- Limiar **mais alto**: mais seguro (menos impostores aceitos), mas aumenta o risco de rejeitar o usuário verdadeiro.
- Limiar **mais baixo**: mais permissivo, com risco de aceitar impostores.
- **Limitação:** o limiar foi definido com poucas imagens (14 com rosto). Em um sistema real seria calibrado com uma base bem maior.

---

## 5. Protótipo (Etapa 5)

### 5A. Reconhecimento e controle de acesso — `src/recognition.py`

- `identify(path)` executa o pipeline completo: aquisição → pré-processamento → detecção → extração (FaceNet) → comparação com todos os templates → limiar de 0.50.
- Retorna o status (`recognized`, `unknown` ou `invalid`), o usuário candidato, o score, o nível de acesso e as imagens de cada etapa.
- Nível de acesso lido do `users.json`. Um usuário de nível N acessa as informações dos níveis 1 até N (hierárquico).

| Nível | Informação ambiental fictícia liberada |
|---|---|
| 1 | Informações ambientais gerais: dados de propriedades rurais e consultas públicas |
| 2 | Relatórios de fiscalização, ocorrências ambientais e análises técnicas |
| 3 | Informações estratégicas, relatórios consolidados e decisões administrativas |

| Usuário | Nível |
|---|---|
| georgewbush | 3 |
| donaldrumsfeld | 2 |
| colinpowell | 2 |
| tonyblair | 1 |
| gerhardschroeder | 1 |

### 5B. Interface — `src/app.py` (Gradio)

- Interface web **local** (`http://127.0.0.1:7860`), sem servidor externo, como exige a proposta.
- Entrada: upload de arquivo local (imagens da base de teste).
- Exibe as etapas visuais: original, pré-processada, rosto detectado (retângulo) e ROI 160×160.
- Mensagens:
  - ✅ **Acesso liberado**: usuário, similaridade, nível e informações liberadas;
  - ⛔ **Acesso negado**: usuário não reconhecido (score abaixo do limiar);
  - ⚠️ **Entrada inválida**: nenhum rosto detectado.
- As imagens de `data/test` aparecem como exemplos clicáveis, o que facilita a demonstração.
- **Layout final:** tema escuro com a cor de destaque laranja. À esquerda, um card com a imagem de entrada, o botão Identificar, o quadro "Resultado" e as imagens de teste; à direita, as 4 etapas em grade 2×2. O card e a grade têm a mesma altura e a página cabe na tela sem rolagem.
- **Quadro "Resultado":** título colorido (verde = acesso liberado, vermelho = acesso negado, laranja = entrada inválida), com usuário, similaridade, limiar, nível de acesso e as informações liberadas.
- **Botão "Avaliação":** abre uma janela flutuante que roda o `evaluate.py` e mostra a acurácia global, o resumo por cenário e o resultado por imagem. Fecha no ✕.

**Mapa dos requisitos mínimos do protótipo (item 4 da proposta):**

| Requisito | Onde |
|---|---|
| Cadastro de usuários fictícios | `prepare_dataset.py` + `enroll.py` |
| Associação a um nível de acesso | `users.json` |
| Processamento de imagem de teste pelo pipeline | `recognition.identify()` |
| Exibição das 3 etapas visuais | `app.py` (4 imagens) |
| Reconhecimento do usuário | FaceNet + similaridade de cosseno + limiar |
| Mensagem com nível e informação liberada | `app.py` (acesso liberado) |
| Mensagem de acesso negado / erro | `app.py` (negado / entrada inválida) |

---

### 5C. Execução em Docker

- `Dockerfile` baseado em `python:3.12-slim`, com PyTorch **CPU** (imagem menor, sem CUDA) e `opencv-python-headless` (sem dependências de interface gráfica).
- Os pesos do FaceNet são baixados **durante o build**, então o container funciona sem internet depois de pronto.
- A base (`data/`, com o banco `db_facenet.npz`) e o `users.json` são copiados para a imagem.
- `docker-compose.yml` expõe a interface em `http://localhost:7860` e monta `outputs/` para os `.csv` da avaliação ficarem no computador.
- O container usa só o FaceNet; os experimentos com DINOv2 rodam fora dele (o `transformers` não foi incluído na imagem).

Comandos:

```
docker compose up --build                          # build + sobe a interface
docker compose exec app python src/evaluate.py     # roda a avaliação dentro do container
docker compose down                                # para
```

---

## 6. Testes e métricas (Etapa 6) — `src/evaluate.py`

- O script percorre todas as imagens de `data/test`, executa o reconhecimento completo e compara o resultado com o **esperado**, que é tirado do nome do arquivo:
  - `<usuário>_NN.jpg` → deve ser **reconhecido como esse usuário** (cenário 1: tentativa correta);
  - `unknown_NN.jpg` → deve ser **"não reconhecido"** (cenário 2: usuário diferente);
  - `invalid_NN.jpg` → deve dar **"sem rosto"** (cenário 3: entrada inválida).
- Gera duas tabelas, impressas no terminal e salvas em `outputs/evaluation_summary.csv` e `outputs/evaluation_details.csv` (abrem no Excel).
- A interface tem um botão **"Avaliação"**, que abre uma janela flutuante com as mesmas tabelas e a acurácia global.
- Como a finalidade é **identificação 1:N**, FAR e FRR não são exigidos pela proposta. Os erros equivalentes aparecem por cenário: um desconhecido aceito seria uma falsa aceitação, e um cadastrado rejeitado seria uma falsa rejeição.

**Resultado (FaceNet, limiar 0.50):**

| Cenário | Testes | Acertos | Erros | Acurácia |
|---|---|---|---|---|
| Usuário cadastrado | 10 | 10 | 0 | 100% |
| Usuário não cadastrado | 4 | 4 | 0 | 100% |
| Entrada inválida | 4 | 4 | 0 | 100% |
| **Total** | **18** | **18** | **0** | **100%** |

- Nenhuma falsa aceitação: os 4 desconhecidos ficaram com score entre 0.186 e 0.304, abaixo do limiar.
- Nenhuma falsa rejeição: o menor score de um cadastrado foi 0.708 (`tonyblair_01`).
- As 4 imagens inválidas (xícara, gato, imagem preta e ruído) foram barradas na etapa de detecção.

> Esta tabela foi montada a partir dos scores do Experimento 3 aplicando o limiar de 0.50. Confirmar com `python src/evaluate.py` (o script usa o mesmo pipeline e deve dar o mesmo resultado).

**Limitações da avaliação (para a discussão):**
- Base pequena (5 usuários, 18 imagens de teste): 100% neste conjunto não significa 100% em geral.
- O limiar foi escolhido olhando os próprios dados de teste; o ideal seria calibrá-lo numa base separada.
- A LFW tem fotos de pessoas públicas, bem iluminadas e quase sempre frontais. Fotos de perfil, com pouca luz ou óculos escuros tendem a falhar na detecção Haar, como aconteceu com `georgewbush/3.jpg` no cadastro.

---

## 7. Referências bibliográficas (ir completando)

- HUANG, G. B. et al. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*. Amherst: University of Massachusetts, 2007. (Technical Report 07-49).
- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011.
- BRADSKI, G. The OpenCV Library. *Dr. Dobb's Journal of Software Tools*, 2000.
- ZUIDERVELD, K. Contrast Limited Adaptive Histogram Equalization. In: HECKBERT, P. (ed.). *Graphics Gems IV*. San Diego: Academic Press, 1994. p. 474–485.
- VIOLA, P.; JONES, M. Rapid Object Detection using a Boosted Cascade of Simple Features. In: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR), 2001, Kauai. *Proceedings* [...]. IEEE, 2001. p. I-511–I-518.
- SCHROFF, F.; KALENICHENKO, D.; PHILBIN, J. FaceNet: A Unified Embedding for Face Recognition and Clustering. In: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR), 2015, Boston. *Proceedings* [...]. IEEE, 2015. p. 815–823.
- CAO, Q. et al. VGGFace2: A dataset for recognising faces across pose and age. In: IEEE INTERNATIONAL CONFERENCE ON AUTOMATIC FACE & GESTURE RECOGNITION, 2018. *Proceedings* [...]. IEEE, 2018. p. 67–74.
- OQUAB, M. et al. DINOv2: Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research*, 2024.
- OJALA, T.; PIETIKÄINEN, M.; MÄENPÄÄ, T. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, v. 24, n. 7, p. 971–987, 2002.
- GONZALEZ, R. C.; WOODS, R. E. *Processamento Digital de Imagens*. 3. ed. São Paulo: Pearson, 2010.
