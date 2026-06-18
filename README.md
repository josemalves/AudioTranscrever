# AudioTranscrever

Aplicação web *self-hosted* para **transcrição automática de áudio e vídeo**, otimizada para português. Faz upload de um ficheiro, escolhe a qualidade e (opcionalmente) a separação de oradores, e recebe a transcrição pronta a descarregar em `.txt`, `.srt` ou `.json`.

Corre inteiramente em casa/servidor próprio (CPU, sem GPU), em Docker, com contas de utilizador, painel de administração e notificações por email.

---

## Funcionalidades

- 🎙️ **Transcrição de áudio e vídeo** com [WhisperX](https://github.com/m-bain/whisperX) (faster-whisper por baixo)
- ⚡ **Dois modos de qualidade:** rápido (modelo `medium`) ou qualidade máxima (`large-v3`)
- 🗣️ **Diarização opcional** — distingue oradores ("Orador 1", "Orador 2"...) via [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- 📄 **Exportação** em texto simples (`.txt`), legendas (`.srt`) e dados estruturados (`.json`)
- 👤 **Contas de utilizador** — login, pedidos de acesso e aprovação por administrador
- 🛠️ **Painel de admin** — aprovar/rejeitar pedidos, ativar/desativar contas, reset de password
- 🕑 **Histórico** de transcrições por utilizador
- 📨 **Notificações por email** quando um trabalho termina ou um acesso é aprovado (opcional)
- 🧹 **Limpeza automática** de ficheiros temporários e de trabalhos antigos
- ⏳ **Fila assíncrona** — processa um trabalho de cada vez, sem sobrecarregar o servidor

---

## Tecnologias

**Backend / Web**
- [Python](https://www.python.org/) 3.12
- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- [SQLAlchemy](https://www.sqlalchemy.org/) sobre [SQLite](https://www.sqlite.org/)
- [Jinja2](https://jinja.palletsprojects.com/) para os templates HTML
- Autenticação por sessão (cookie assinado) com `passlib` + `bcrypt`

**Motor de transcrição**
- [WhisperX](https://github.com/m-bain/whisperX) 3.4.2
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) (diarização)
- [PyTorch](https://pytorch.org/) (build CPU)

**Infraestrutura**
- [Docker](https://www.docker.com/) + Docker Compose

---

## Arquitetura

Dois serviços Docker independentes:

```
┌──────────────────────┐         HTTP          ┌──────────────────────────┐
│   transcrever-app    │  ───────────────────► │   transcrever-whisperx    │
│                      │                       │                           │
│  FastAPI + SQLite    │                       │  FastAPI + WhisperX       │
│  UI, contas, fila,   │ ◄─────────────────── │  faster-whisper, pyannote  │
│  histórico, emails   │      transcrição      │  (CPU, int8)              │
└──────────────────────┘                       └──────────────────────────┘
```

- **`app`** — interface web, autenticação, base de dados, fila de trabalhos e notificações. Coloca os pedidos de transcrição no serviço WhisperX por HTTP interno.
- **`whisperx`** — serviço dedicado que carrega os modelos e faz a transcrição/diarização. Isolado para poder limitar CPU e memória independentemente.

---

## Como correr

**Pré-requisitos:** [Docker](https://docs.docker.com/get-docker/) e Docker Compose.

```bash
# 1. Clonar o repositório
git clone <url-do-repo>
cd AudioTranscrever

# 2. Criar o ficheiro de configuração a partir do exemplo
cp .env.example .env

# 3. Editar o .env e preencher os segredos (ver secção abaixo)
#    - SESSION_SECRET, ADMIN_PASSWORD são obrigatórios
#    - HF_TOKEN só é preciso se quiseres diarização

# 4. Arrancar
docker compose up -d --build
```

A aplicação fica disponível em `http://localhost:8082` (porta configurável via `APP_PORT`).

No primeiro arranque é criada automaticamente uma conta de administrador com as credenciais definidas no `.env`.

> ℹ️ Na primeira transcrição, os modelos do Whisper são descarregados (vários GB) e ficam em cache local em `hf-cache/`. As transcrições seguintes são imediatas.

---

## Configuração

As variáveis principais ficam no `.env` (ver [`.env.example`](.env.example) para a lista completa):

| Variável | Descrição |
|----------|-----------|
| `SESSION_SECRET` | Segredo para assinar os cookies de sessão (gerar com `openssl rand -hex 32`) |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | Conta de admin criada no primeiro arranque |
| `WHISPER_MODEL` | Modelo Whisper por defeito (`medium`, `large-v3`, ...) |
| `WHISPER_LANGUAGE` | Idioma da transcrição (`pt` por defeito) |
| `HF_TOKEN` | Token HuggingFace — necessário **apenas** para diarização |
| `SMTP_*` | Configuração de email para notificações (opcional) |

---

## Licença

[MIT](LICENSE) © José Alves
