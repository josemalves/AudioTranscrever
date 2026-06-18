import json as _json
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from auth import generate_password, hash_password, verify_password
from database import Base, SessionLocal, engine, get_db
from models import AccessRequest, Transcription, User
from notify import notify_access_approved, notify_job_done, notify_job_failed


SESSION_SECRET = os.environ.get("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET env var is required")

TRANSCRIBE_URL = os.environ.get("TRANSCRIBE_URL", "http://whisperx:8000")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "pt")
MODEL_FAST = os.environ.get("WHISPER_MODEL_FAST", "medium")
MODEL_QUALITY = os.environ.get("WHISPER_MODEL_QUALITY", "large-v3")

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@local")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

ALLOWED_PREFIXES = ("audio/", "video/")
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
AUTO_DELETE_FAILED_DAYS = int(os.environ.get("AUTO_DELETE_FAILED_DAYS", "30"))
AUTO_DELETE_DONE_DAYS = int(os.environ.get("AUTO_DELETE_DONE_DAYS", "0"))  # 0 = nunca

# In-memory login attempt tracker. Keyed by IP — resets on restart.
LOGIN_ATTEMPTS: dict = {}
LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _login_rate_limited(ip: str) -> bool:
    now = time.time()
    attempts = [t for t in LOGIN_ATTEMPTS.get(ip, []) if now - t < LOGIN_WINDOW_SECONDS]
    LOGIN_ATTEMPTS[ip] = attempts
    return len(attempts) >= LOGIN_MAX_ATTEMPTS


def _login_record(ip: str):
    LOGIN_ATTEMPTS.setdefault(ip, []).append(time.time())


# Same in-memory pattern for /request-access — 3 per hour per IP.
ACCESS_ATTEMPTS: dict = {}
ACCESS_MAX_ATTEMPTS = 3
ACCESS_WINDOW_SECONDS = 3600


def _access_rate_limited(ip: str) -> bool:
    now = time.time()
    attempts = [t for t in ACCESS_ATTEMPTS.get(ip, []) if now - t < ACCESS_WINDOW_SECONDS]
    ACCESS_ATTEMPTS[ip] = attempts
    return len(attempts) >= ACCESS_MAX_ATTEMPTS


def _access_record(ip: str):
    ACCESS_ATTEMPTS.setdefault(ip, []).append(time.time())


def utcnow() -> datetime:
    """Return naive UTC datetime — replaces deprecated utcnow()."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


def verify_csrf(request: Request, token: Optional[str]):
    expected = request.session.get("csrf_token")
    if not expected or not token or not secrets.compare_digest(expected, token):
        raise HTTPException(status_code=403, detail="CSRF token inválido.")


def sanitize_error(msg: Optional[str], limit: int = 200) -> str:
    if not msg:
        return "Erro desconhecido."
    first_line = msg.split("\n")[0].strip()
    return first_line[:limit] + ("…" if len(first_line) > limit else "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite()

    db = SessionLocal()
    try:
        stuck = db.query(Transcription).filter_by(status="running").all()
        for j in stuck:
            j.status = "failed"
            j.error_message = "Interrompido por reinício do servidor."
            j.completed_at = utcnow()
        if stuck:
            db.commit()
    finally:
        db.close()

    threading.Thread(target=_worker_loop, daemon=True).start()

    db = SessionLocal()
    try:
        if not db.query(User).filter_by(is_admin=True).first():
            if not ADMIN_PASSWORD:
                raise RuntimeError("ADMIN_PASSWORD env var is required to bootstrap the first admin")
            admin = User(
                username=ADMIN_USERNAME,
                email=ADMIN_EMAIL,
                password_hash=hash_password(ADMIN_PASSWORD),
                is_admin=True,
                is_active=True,
            )
            db.add(admin)
            db.commit()
            print(f"[bootstrap] admin '{ADMIN_USERNAME}' criado.")
    finally:
        db.close()

    yield


app = FastAPI(title="AudioTranscrever", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=os.environ.get("COOKIE_SECURE", "true").lower() == "true",
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _migrate_sqlite():
    """Add new columns to existing transcriptions table if missing."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if "transcriptions" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("transcriptions")}
    with engine.begin() as conn:
        if "segments_json" not in cols:
            conn.execute(text("ALTER TABLE transcriptions ADD COLUMN segments_json TEXT"))
        if "diarized" not in cols:
            conn.execute(text("ALTER TABLE transcriptions ADD COLUMN diarized BOOLEAN DEFAULT 0 NOT NULL"))
        if "status" not in cols:
            conn.execute(text("ALTER TABLE transcriptions ADD COLUMN status VARCHAR DEFAULT 'done' NOT NULL"))
        if "audio_path" not in cols:
            conn.execute(text("ALTER TABLE transcriptions ADD COLUMN audio_path VARCHAR"))
        if "error_message" not in cols:
            conn.execute(text("ALTER TABLE transcriptions ADD COLUMN error_message TEXT"))
        if "started_at" not in cols:
            conn.execute(text("ALTER TABLE transcriptions ADD COLUMN started_at DATETIME"))
        if "completed_at" not in cols:
            conn.execute(text("ALTER TABLE transcriptions ADD COLUMN completed_at DATETIME"))
        if "model" not in cols:
            conn.execute(text("ALTER TABLE transcriptions ADD COLUMN model VARCHAR"))


def _process_job(job_id: int):
    db = SessionLocal()
    try:
        job = db.get(Transcription, job_id)
        if not job:
            return
        job.status = "running"
        job.started_at = utcnow()
        db.commit()

        audio_path = job.audio_path
        diarize_flag = bool(job.diarized)
        filename = job.filename or "audio"
        owner = db.get(User, job.user_id)

        try:
            if not audio_path or not os.path.exists(audio_path):
                raise RuntimeError("Ficheiro de áudio não encontrado.")

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            with httpx.Client(timeout=httpx.Timeout(None)) as client:
                r = client.post(
                    f"{TRANSCRIBE_URL}/transcribe",
                    files={"file": (filename, audio_bytes, "application/octet-stream")},
                    data={
                        "diarize": "true" if diarize_flag else "false",
                        "language": WHISPER_LANGUAGE,
                        "model": job.model or MODEL_FAST,
                    },
                )
                r.raise_for_status()
                result = r.json()

            job.text = (result.get("text") or "").strip()
            job.segments_json = _json.dumps(result.get("segments") or [])
            duration = result.get("duration")
            job.duration_seconds = int(duration) if duration is not None else None
            job.status = "done"
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)[:2000]
            print(f"[worker] job {job.id} failed: {e}")
        finally:
            job.completed_at = utcnow()
            try:
                if job.audio_path and os.path.exists(job.audio_path):
                    os.unlink(job.audio_path)
            except OSError:
                pass
            job.audio_path = None
            db.commit()

            if owner and owner.email:
                processing = None
                if job.started_at and job.completed_at:
                    processing = int((job.completed_at - job.started_at).total_seconds())
                if job.status == "done":
                    notify_job_done(
                        owner.email, owner.username, filename, job.id,
                        job.duration_seconds, processing,
                    )
                elif job.status == "failed":
                    notify_job_failed(
                        owner.email, owner.username, filename,
                        sanitize_error(job.error_message),
                    )
    finally:
        db.close()


def _cleanup_old_jobs():
    """Apaga jobs `failed` mais antigos que AUTO_DELETE_FAILED_DAYS dias.
    Opcionalmente também `done` se AUTO_DELETE_DONE_DAYS > 0."""
    try:
        db = SessionLocal()
        try:
            now = utcnow()
            removed = 0
            if AUTO_DELETE_FAILED_DAYS > 0:
                cutoff = now - timedelta(days=AUTO_DELETE_FAILED_DAYS)
                old = db.query(Transcription).filter(
                    Transcription.status == "failed",
                    Transcription.created_at < cutoff,
                ).all()
                for j in old:
                    if j.audio_path and os.path.exists(j.audio_path):
                        try: os.unlink(j.audio_path)
                        except OSError: pass
                    db.delete(j)
                removed += len(old)
            if AUTO_DELETE_DONE_DAYS > 0:
                cutoff = now - timedelta(days=AUTO_DELETE_DONE_DAYS)
                old = db.query(Transcription).filter(
                    Transcription.status == "done",
                    Transcription.created_at < cutoff,
                ).all()
                for j in old:
                    db.delete(j)
                removed += len(old)
            if removed:
                db.commit()
                print(f"[cleanup] removed {removed} old jobs")
        finally:
            db.close()
    except Exception as e:
        print(f"[cleanup] old jobs error: {e}")


def _cleanup_orphans():
    """Apaga ficheiros em UPLOAD_DIR que (a) não estão referenciados por jobs
    pending/running, (b) têm mais de 1 hora. Evita acumulação por crashes."""
    try:
        if not UPLOAD_DIR.exists():
            return
        db = SessionLocal()
        try:
            referenced = {
                row[0]
                for row in db.query(Transcription.audio_path)
                .filter(Transcription.audio_path.isnot(None))
                .all()
            }
        finally:
            db.close()

        threshold = time.time() - 3600  # 1h
        for f in UPLOAD_DIR.iterdir():
            if not f.is_file():
                continue
            if str(f) in referenced:
                continue
            try:
                if f.stat().st_mtime < threshold:
                    f.unlink()
                    print(f"[cleanup] removed orphan {f.name}")
            except OSError as e:
                print(f"[cleanup] failed to remove {f}: {e}")
    except Exception as e:
        print(f"[cleanup] error: {e}")


def _worker_loop():
    print("[worker] started")
    last_cleanup = 0.0
    while True:
        try:
            db = SessionLocal()
            try:
                job = (
                    db.query(Transcription)
                    .filter_by(status="pending")
                    .order_by(Transcription.created_at)
                    .first()
                )
                job_id = job.id if job else None
            finally:
                db.close()

            if job_id is None:
                if time.time() - last_cleanup > 3600:
                    _cleanup_orphans()
                    _cleanup_old_jobs()
                    last_cleanup = time.time()
                time.sleep(2)
                continue

            _process_job(job_id)
        except Exception as e:
            print(f"[worker] loop error: {e}")
            time.sleep(5)


def current_user(request: Request, db: Session) -> Optional[User]:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    user = db.get(User, user_id)
    if not user or not user.is_active:
        request.session.clear()
        return None
    return user


def render(request: Request, template: str, **ctx):
    user = ctx.pop("user", None)
    flash = ctx.pop("flash", None) or request.session.pop("flash", None)
    return templates.TemplateResponse(
        template,
        {
            "request": request,
            "user": user,
            "flash": flash,
            "csrf_token": csrf_token(request),
            **ctx,
        },
    )


def flash(request: Request, message: str, level: str = "info"):
    request.session["flash"] = {"message": message, "level": level}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    return RedirectResponse("/app" if user else "/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    if current_user(request, db):
        return RedirectResponse("/app", status_code=303)
    return render(request, "login.html")


@app.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    verify_csrf(request, csrf_token)
    ip = _client_ip(request)
    if _login_rate_limited(ip):
        flash(
            request,
            "Demasiadas tentativas falhadas. Espera 5 minutos antes de tentar de novo.",
            "error",
        )
        return RedirectResponse("/login", status_code=303)

    user = (
        db.query(User)
        .filter((User.username == username) | (User.email == username.lower()))
        .first()
    )
    if not user or not user.is_active or not verify_password(password, user.password_hash):
        _login_record(ip)
        print(f"[auth] failed login from {ip} for username={username!r}")
        flash(request, "Credenciais inválidas.", "error")
        return RedirectResponse("/login", status_code=303)
    LOGIN_ATTEMPTS.pop(ip, None)
    request.session["user_id"] = user.id
    user.last_login_at = utcnow()
    db.commit()
    return RedirectResponse("/app", status_code=303)


@app.post("/logout")
def logout(request: Request, csrf_token: str = Form("")):
    verify_csrf(request, csrf_token)
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/request-access", response_class=HTMLResponse)
def request_access_page(request: Request):
    return render(request, "request_access.html")


@app.post("/request-access")
def request_access(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    motivo: str = Form(""),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    verify_csrf(request, csrf_token)

    ip = _client_ip(request)
    if _access_rate_limited(ip):
        flash(
            request,
            "Demasiados pedidos recentes. Tenta novamente daqui a 1 hora.",
            "error",
        )
        return RedirectResponse("/request-access", status_code=303)

    name = name.strip()
    email = email.strip().lower()
    if not name or not email:
        flash(request, "Nome e email são obrigatórios.", "error")
        return RedirectResponse("/request-access", status_code=303)
    existing = db.query(AccessRequest).filter_by(email=email, status="pending").first()
    if existing:
        flash(request, "Já existe um pedido pendente para este email.", "info")
        return RedirectResponse("/login", status_code=303)
    if db.query(User).filter_by(email=email).first():
        flash(request, "Já existe uma conta com este email. Tenta entrar.", "info")
        return RedirectResponse("/login", status_code=303)
    req = AccessRequest(name=name, email=email, motivo=motivo.strip() or None)
    db.add(req)
    db.commit()
    _access_record(ip)
    flash(request, "Pedido enviado. Será contactado por email assim que for aprovado.", "success")
    return RedirectResponse("/login", status_code=303)


@app.get("/app", response_class=HTMLResponse)
def app_page(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return render(request, "app.html", user=user)


@app.post("/transcribe")
def transcribe(
    request: Request,
    file: UploadFile = File(...),
    diarize: str = Form("false"),
    quality: str = Form("false"),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    verify_csrf(request, csrf_token)
    user = current_user(request, db)
    if not user:
        return JSONResponse({"error": "Não autenticado."}, status_code=401)

    ctype = file.content_type or ""
    if ctype and not ctype.startswith(ALLOWED_PREFIXES):
        return JSONResponse(
            {"error": f"Tipo de ficheiro não suportado: {ctype}"}, status_code=400
        )

    diarize_flag = diarize.lower() in ("true", "on", "1", "yes")
    quality_flag = quality.lower() in ("true", "on", "1", "yes")
    model_name = MODEL_QUALITY if quality_flag else MODEL_FAST

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + 1024 * 1024:
        return JSONResponse(
            {"error": f"Ficheiro acima do limite ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB). Para ficheiros maiores, contacta o admin."},
            status_code=413,
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    audio_id = uuid.uuid4().hex
    audio_path = UPLOAD_DIR / f"{audio_id}{suffix}"

    size = 0
    too_big = False
    with open(audio_path, "wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                too_big = True
                break
            out.write(chunk)

    if too_big:
        try:
            audio_path.unlink()
        except OSError:
            pass
        return JSONResponse(
            {"error": f"Ficheiro acima do limite ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB). Para ficheiros maiores, contacta o admin."},
            status_code=413,
        )

    if size == 0:
        try:
            audio_path.unlink()
        except OSError:
            pass
        return JSONResponse({"error": "Ficheiro vazio."}, status_code=400)

    job = Transcription(
        user_id=user.id,
        filename=file.filename or "audio",
        diarized=diarize_flag,
        model=model_name,
        saved=True,
        status="pending",
        audio_path=str(audio_path),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return JSONResponse({"job_id": job.id, "status": "pending", "model": model_name})


@app.get("/api/jobs")
def api_jobs(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    if not user:
        return JSONResponse({"error": "Não autenticado."}, status_code=401)
    items = (
        db.query(Transcription)
        .filter_by(user_id=user.id)
        .order_by(Transcription.created_at.desc())
        .all()
    )
    return JSONResponse(
        [
            {
                "id": it.id,
                "filename": it.filename,
                "status": it.status,
                "diarized": bool(it.diarized),
                "model": it.model,
                "duration_seconds": it.duration_seconds,
                "error_message": it.error_message if it.status == "failed" else None,
                "created_at": it.created_at.isoformat() if it.created_at else None,
                "started_at": it.started_at.isoformat() if it.started_at else None,
                "completed_at": it.completed_at.isoformat() if it.completed_at else None,
            }
            for it in items
        ]
    )


@app.get("/history", response_class=HTMLResponse)
def history(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return render(request, "history.html", user=user)


@app.get("/history/{tid}", response_class=HTMLResponse)
def history_detail(tid: int, request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    item = db.query(Transcription).filter_by(id=tid, user_id=user.id).first()
    if not item:
        raise HTTPException(404)
    if item.status != "done":
        return RedirectResponse("/history", status_code=303)
    segments = _json.loads(item.segments_json) if item.segments_json else []
    data_json = _json.dumps(
        {"text": item.text or "", "segments": segments, "diarized": item.diarized}
    )
    return render(
        request,
        "transcription_detail.html",
        user=user,
        item=item,
        data_json=data_json,
    )


@app.post("/history/{tid}/delete")
def history_delete(
    tid: int,
    request: Request,
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    verify_csrf(request, csrf_token)
    user = current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    item = db.query(Transcription).filter_by(id=tid, user_id=user.id).first()
    if item:
        if item.audio_path and os.path.exists(item.audio_path):
            try:
                os.unlink(item.audio_path)
            except OSError:
                pass
        was_pending = item.status == "pending"
        db.delete(item)
        db.commit()
        flash(request, "Trabalho cancelado." if was_pending else "Transcrição apagada.", "success")
    return RedirectResponse("/history", status_code=303)


def _require_admin(request: Request, db: Session) -> User:
    user = current_user(request, db)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Acesso negado.")
    return user


def _unique_username(db: Session, base: str) -> str:
    base = "".join(c for c in base if c.isalnum() or c in "._-").lower() or "user"
    candidate = base
    n = 1
    while db.query(User).filter_by(username=candidate).first():
        n += 1
        candidate = f"{base}{n}"
    return candidate


@app.get("/admin", response_class=HTMLResponse)
def admin_panel(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not user.is_admin:
        return RedirectResponse("/app", status_code=303)

    pending = (
        db.query(AccessRequest)
        .filter_by(status="pending")
        .order_by(AccessRequest.created_at.desc())
        .all()
    )
    users = db.query(User).order_by(User.created_at.desc()).all()

    stats = []
    for u in users:
        total = db.query(Transcription).filter_by(user_id=u.id).count()
        saved = db.query(Transcription).filter_by(user_id=u.id, saved=True).count()
        last = (
            db.query(Transcription)
            .filter_by(user_id=u.id)
            .order_by(Transcription.created_at.desc())
            .first()
        )
        stats.append(
            {
                "user": u,
                "total": total,
                "saved": saved,
                "last_used": last.created_at if last else None,
            }
        )

    return render(
        request, "admin.html",
        user=user,
        pending=pending,
        stats=stats,
    )


@app.post("/admin/requests/{rid}/approve")
def admin_approve(
    rid: int,
    request: Request,
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    verify_csrf(request, csrf_token)
    _require_admin(request, db)
    req = db.get(AccessRequest, rid)
    if not req or req.status != "pending":
        raise HTTPException(404)
    if db.query(User).filter_by(email=req.email).first():
        flash(request, f"Já existe um utilizador com email {req.email}.", "error")
        return RedirectResponse("/admin", status_code=303)
    base_username = req.email.split("@")[0]
    username = _unique_username(db, base_username)
    password = generate_password()
    user = User(
        username=username,
        email=req.email,
        password_hash=hash_password(password),
        is_admin=False,
        is_active=True,
    )
    db.add(user)
    req.status = "approved"
    db.commit()

    sent = notify_access_approved(req.email, req.name or username, username, password)
    if sent:
        flash(
            request,
            f"Conta criada e credenciais enviadas por email para {req.email} (utilizador: {username}).",
            "success",
        )
    else:
        flash(
            request,
            f"Conta criada — utilizador: {username} | password: {password} (envia ao colega; SMTP não configurado, password não voltará a ser mostrada)",
            "success",
        )
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/requests/{rid}/reject")
def admin_reject(
    rid: int,
    request: Request,
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    verify_csrf(request, csrf_token)
    _require_admin(request, db)
    req = db.get(AccessRequest, rid)
    if not req or req.status != "pending":
        raise HTTPException(404)
    req.status = "rejected"
    db.commit()
    flash(request, "Pedido rejeitado.", "info")
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{uid}/toggle")
def admin_toggle(
    uid: int,
    request: Request,
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    verify_csrf(request, csrf_token)
    admin = _require_admin(request, db)
    target = db.get(User, uid)
    if not target:
        raise HTTPException(404)
    if target.id == admin.id:
        flash(request, "Não podes desativar a própria conta.", "error")
        return RedirectResponse("/admin", status_code=303)
    target.is_active = not target.is_active
    db.commit()
    flash(
        request,
        f"Utilizador {target.username}: {'ativo' if target.is_active else 'desativado'}.",
        "info",
    )
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{uid}/reset-password")
def admin_reset_password(
    uid: int,
    request: Request,
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    verify_csrf(request, csrf_token)
    _require_admin(request, db)
    target = db.get(User, uid)
    if not target:
        raise HTTPException(404)
    password = generate_password()
    target.password_hash = hash_password(password)
    db.commit()
    flash(
        request,
        f"Nova password para {target.username}: {password} (envia ao utilizador; não vai ser mostrada outra vez)",
        "success",
    )
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/users/{uid}/delete")
def admin_delete_user(
    uid: int,
    request: Request,
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    verify_csrf(request, csrf_token)
    admin = _require_admin(request, db)
    target = db.get(User, uid)
    if not target:
        raise HTTPException(404)
    if target.id == admin.id:
        flash(request, "Não podes apagar a própria conta.", "error")
        return RedirectResponse("/admin", status_code=303)
    db.delete(target)
    db.commit()
    flash(request, f"Utilizador {target.username} apagado.", "info")
    return RedirectResponse("/admin", status_code=303)
