#!/usr/bin/env python3
"""
🏟️ Arena of Models v2.4
✅ Судья | ✅ Анти-повторение | ✅ Жёсткая русификация | ✅ Без эмодзи | ✅ .env конфиг
"""
import json, asyncio, httpx, uuid, sqlite3, re, os
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

# ========== ЗАГРУЗКА КОНФИГА ИЗ .ENV ==========
load_dotenv()

# ========== КОНФИГ ==========
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
DEFAULT_MODELS = [m.strip() for m in os.getenv("DEFAULT_MODELS", "llama3.2:latest,ruadapt-qwen2.5-14b:latest").split(",")]
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "ruadapt-qwen2.5-14b:latest")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
MAX_TOKENS_JUDGE = int(os.getenv("MAX_TOKENS_JUDGE", "800"))
DB_PATH = os.getenv("DB_PATH", "arena.db")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# ========== БАЗА ДАННЫХ ==========
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS arena_sessions (
        id TEXT PRIMARY KEY, model_a TEXT, model_b TEXT, topic TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, ended_at TIMESTAMP,
        turns_count INTEGER DEFAULT 0, status TEXT DEFAULT 'active',
        judge_verdict TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS arena_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
        model TEXT, content TEXT, turn_num INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_msg_sess ON arena_messages(session_id, turn_num)')
    conn.commit(); conn.close()
    print(f"✅ БД: {DB_PATH}")

def create_session(sid, ma, mb, topic):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO arena_sessions (id,model_a,model_b,topic) VALUES (?,?,?,?)',(sid,ma,mb,topic))
    conn.commit(); conn.close()

def save_message(sid, model, content, turn):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO arena_messages (session_id,model,content,turn_num) VALUES (?,?,?,?)',(sid,model,content,turn))
    conn.execute('UPDATE arena_sessions SET turns_count=turns_count+1 WHERE id=?',(sid,))
    conn.commit(); conn.close()

def end_session(sid, status, verdict=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE arena_sessions SET ended_at=CURRENT_TIMESTAMP,status=?,judge_verdict=? WHERE id=?',(status,verdict,sid))
    conn.commit(); conn.close()

def get_history(sid):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT model,content,turn_num,created_at FROM arena_messages WHERE session_id=? ORDER BY turn_num',(sid,)).fetchall()
    conn.close()
    return [{"model":r[0],"content":r[1],"turn":r[2],"time":r[3]} for r in rows]

def get_all_sessions(limit=50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT id,model_a,model_b,topic,started_at,ended_at,turns_count,status,judge_verdict FROM arena_sessions ORDER BY started_at DESC LIMIT ?',(limit,)).fetchall()
    conn.close()
    return [{"id":r[0],"model_a":r[1],"model_b":r[2],"topic":r[3],"started":r[4],"ended":r[5],"turns":r[6],"status":r[7],"verdict":r[8]} for r in rows]

# ========== СЕССИИ ==========
ACTIVE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Arena v2.3 — запуск"); init_db(); yield; print("🛑 Завершение")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root(): return FileResponse("index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico") if os.path.exists("favicon.ico") else JSONResponse(status_code=204)

@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get("http://127.0.0.1:11434/api/tags")
        if r.status_code==200:
            models = sorted(set(m.get('name') or m.get('model') for m in r.json().get('models',[]) if m.get('name') or m.get('model')))
            return JSONResponse({"models": models or DEFAULT_MODELS})
    except Exception as e: print(f"⚠️ Models: {e}")
    return JSONResponse({"models": DEFAULT_MODELS})

@app.get("/api/sessions")
async def list_sessions():
    return JSONResponse({"sessions": get_all_sessions()})

@app.get("/api/session/{sid}")
async def get_session(sid: str):
    return JSONResponse({"session_id": sid, "messages": get_history(sid)})

# ========== УСИЛЕННЫЕ ПРОМПТЫ ==========
def _get_system_prompt(current: str, opponent: str) -> str:
    return f"""Ты — Модель {current}. Интеллектуальный спор с Модель {opponent}.

📜 **ЖЁСТКИЕ ПРАВИЛА:**
1. ТОЛЬКО РУССКИЙ ЯЗЫК. Английские слова ЗАПРЕЩЕНЫ даже в транслитерации.
2. ЗАПРЕЩЕНО использовать эмодзи (🤔, 👀, 📖 и любые другие).
3. НЕ ПОВТОРЯЙ структуру и фразы собеседника. Будь оригинален.
4. Краткость: 2-4 предложения.
5. ИЗБЕГАЙ шаблонов: "Я согласен", "Интересно", "Может быть", "Возможно".
6. НЕ ДОБАВЛЯЙ префиксы [A]:, [Б]: в начало ответа.
7. Если не знаешь факт — честно признайся.

🎯 **ЗАДАЧА:** Показать эрудицию и способность развивать тему без повторений."""

def _get_judge_prompt(topic: str, dialog: list) -> str:
    dialog_text = chr(10).join(
        f"[{m.get('speaker', '??')}]: {m.get('content', '')}" 
        for m in dialog if m.get('speaker')
    )
    
    return f"""Ты — независимый эксперт. Проанализируй диалог на тему: "{topic}"

📋 ДИАЛОГ:
{dialog_text}

🔍 **КРИТЕРИИ:**
- Оригинальность (не повторяет ли модель оппонента?)
- Глубина аргументов
- Чистота русского языка (отсутствие англицизмов и эмодзи)
- Логичность

📝 **ФОРМАТ ОТВЕТА (СТРОГО):**
Модель А сильные стороны: [текст]
Модель А слабые стороны: [текст]
Модель Б сильные стороны: [текст]
Модель Б слабые стороны: [текст]
ВЕРДИКТ: [Победа А / Победа Б / Ничья]
ОБОСНОВАНИЕ: [2-3 предложения]

⚠️ Только русский язык. Без Markdown и эмодзи."""

# ========== ЖЁСТКАЯ ЧИСТКА ==========
def _clean(content: str) -> str:
    # 1. Удаляем все эмодзи
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # смайлики
        "\U0001F300-\U0001F5FF"  # символы и пиктограммы
        "\U0001F680-\U0001F6FF"  # транспорт и символы
        "\U0001F1E0-\U0001F1FF"  # флаги
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    content = emoji_pattern.sub('', content)
    
    # 2. Удаляем префиксы
    content = re.sub(r'^\s*\[?[АAБB]\]?\s*:?\s*', '', content.strip())
    
    # 3. Расширенный словарь замены англицизмов
    replacements = {
        # Английские слова
        'interesting': 'любопытно',
        'interessno': 'любопытно',
        'human element': 'человеческий фактор',
        'tech-issues': 'технические проблемы',
        'artificial intelligence': 'искусственный интеллект',
        'machine learning': 'машинное обучение',
        'data analytics': 'анализ данных',
        'personalized': 'персонализированный',
        'cooperative thinking': 'совместное мышление',
        'experiential learning': 'практическое обучение',
        'focusiratsya': 'сосредоточиться',
        'mechalzoit': 'завораживает',
        'strategию': 'стратегию',
        'facts': 'факты',
        
        # Транслит
        'personalizirovanny': 'персонализированный',
        'adaptablenykh': 'адаптивных',
        'sophisticated': 'сложный',
        'intuitivnykh': 'интуитивных',
        'user-': 'пользовательско-',
        'très': 'очень',
        'exactnie': 'именно',
        'fantastic': 'отлично',
        'completely': 'полностью',
        'zagадка': 'загадка',
    }
    
    for eng, rus in replacements.items():
        content = re.sub(eng, rus, content, flags=re.IGNORECASE)
    
    # 4. Удаляем слова-паразиты в начале
    stop_words = [
        "действительно", "правильно", "верно", "точно", "безусловно", 
        "интересно", "согласен", "понятно", "я думаю", "мне кажется", 
        "возможно", "может быть", "наверное", "видимо", "похоже",
        "interessno", "interesting"
    ]
    
    for w in stop_words:
        pattern = r'^' + re.escape(w) + r'\s*[,:.\-!]?\s*'
        content = re.sub(pattern, '', content, flags=re.IGNORECASE).strip()
    
    # 5. Капитализация первого символа
    if content and content[0].islower():
        content = content[0].upper() + content[1:]
    
    return content.strip()

def _build_prompt(history: list, current: str, opponent: str) -> list:
    sys = _get_system_prompt(current, opponent)
    messages = [{"role":"system","content":sys}]
    
    anti_repeat = "⚠️ Скажи что-то НОВОЕ. Не повторяй оппонента. Без эмодзи."
    
    for i, msg in enumerate(history):
        content = _clean(msg["content"])
        if i == 0 and msg["role"] == "user":
            messages.append({"role":"user","content": f"[Тема]: {content}\n{anti_repeat}"})
        else:
            spk = msg.get("speaker")
            if spk == ("B" if current == "A" else "A"):
                messages.append({"role":"user","content": f"[{opponent}]: {content}"})
            else:
                messages.append({"role":"assistant","content": f"[{current}]: {content}"})
    
    if messages[-1]["role"] == "user":
        messages[-1]["content"] += f"\n\n{anti_repeat}"
    
    return messages

async def _call_ollama(model: str, messages: list, max_tokens: int = MAX_TOKENS) -> str:
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(OLLAMA_URL, json={
                "model": model, "messages": messages, "stream": False,
                "options": {"temperature": TEMPERATURE, "num_predict": max_tokens}
            })
        if resp.status_code == 200:
            return _clean(resp.json().get("message", {}).get("content", "").strip())
    except httpx.ConnectError:
        print(f"❌ ConnectError к {OLLAMA_URL}")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
    return ""

async def _call_judge(topic: str, dialog: list) -> str:
    if not dialog:
        return "⚠️ Диалог пуст — нечего анализировать"
    
    messages = [{"role":"user","content": _get_judge_prompt(topic, dialog)}]
    verdict = await _call_ollama(JUDGE_MODEL, messages, max_tokens=MAX_TOKENS_JUDGE)
    return verdict if verdict else "⚠️ Вердикт не получен"

@app.post("/api/arena/start")
async def start(req: Request):
    data = await req.json()
    sid = str(uuid.uuid4())
    create_session(sid, data.get("model_a", DEFAULT_MODELS[0]), data.get("model_b", DEFAULT_MODELS[1]), data.get("topic", ""))
    ACTIVE[sid] = {
        "history": [{"role":"user","content": data.get("topic", "")}],
        "model_a": data.get("model_a", DEFAULT_MODELS[0]),
        "model_b": data.get("model_b", DEFAULT_MODELS[1]),
        "stop": False,
        "max_turns": data.get("max_turns", MAX_TURNS),
        "turn": 0
    }
    print(f"🎭 Старт: {sid}")
    return JSONResponse({"session_id": sid})

@app.get("/api/arena/stream/{sid}")
async def stream(sid: str):
    if sid not in ACTIVE:
        raise HTTPException(404)
    sess = ACTIVE[sid]
    
    async def gen():
        while sess["turn"] < sess["max_turns"] and not sess["stop"]:
            for model_key, model_name in [("A", sess["model_a"]), ("B", sess["model_b"])]:
                if sess["stop"]:
                    break
                
                yield f"data: {json.dumps({'type':'typing','model':model_key})}\n\n"
                
                messages = _build_prompt(sess["history"], model_key, "B" if model_key == "A" else "A")
                content = await _call_ollama(model_name, messages)
                
                if content and not sess["stop"]:
                    sess["history"].append({"role":"assistant","content": content, "speaker": model_key})
                    save_message(sid, model_key, content, sess["turn"] + 1)
                    yield f"data: {json.dumps({'type':'message','model':model_key,'content':content})}\n\n"
                    print(f"{'🟢' if model_key == 'A' else '🔵'} {model_key}: {content[:60]}...")
                else:
                    yield f"data: {json.dumps({'type':'error','model':model_key,'error':'Пустой ответ'})}\n\n"
                await asyncio.sleep(0.3)
            sess["turn"] += 1
            if sess["stop"]:
                break
        
        status = "stopped" if sess["stop"] else "finished"
        topic = sess["history"][0]["content"] if sess["history"] else "Без темы"
        dialog = [m for m in sess["history"] if m.get("speaker")]
        
        if dialog and status == "finished":
            yield f"data: {json.dumps({'type':'judge_thinking'})}\n\n"
            print("⚖️ Судья анализирует...")
            verdict = await _call_judge(topic, dialog)
            yield f"data: {json.dumps({'type':'verdict','content':verdict})}\n\n"
            end_session(sid, status, verdict)
            print(f"⚖️ Вердикт: {verdict[:100]}...")
        else:
            end_session(sid, status, None)
        
        yield f"data: {json.dumps({'type':status})}\n\n"
        print(f"✅ {sid} — {status}")
        if sid in ACTIVE:
            del ACTIVE[sid]
    
    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    })

@app.post("/api/arena/stop/{sid}")
async def stop(sid: str):
    if sid in ACTIVE:
        ACTIVE[sid]["stop"] = True
        end_session(sid, "stopped", None)
        print(f"⏹ Останов: {sid}")
    return JSONResponse({"status":"ok"})

@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM arena_messages WHERE session_id=?', (sid,))
    conn.execute('DELETE FROM arena_sessions WHERE id=?', (sid,))
    conn.commit(); conn.close()
    if sid in ACTIVE:
        del ACTIVE[sid]
    print(f"🗑️ Удалена сессия: {sid}")
    return JSONResponse({"status":"ok"})

if __name__ == "__main__":
    print(f"🏟️ Arena v2.4 — запуск на http://{HOST}:{PORT}")
    print(f"⚖️ Судья: {JUDGE_MODEL}")
    print(f"🎭 Модели: {DEFAULT_MODELS[0]} ↔ {DEFAULT_MODELS[1]}")
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)
