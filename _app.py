#!/usr/bin/env python3
"""
🏟️ AI Hub: Арена Моделей + Голосовой Чат с Памятью
✅ Судья | ✅ Анти-повторение | ✅ TTS/ASR | ✅ Долговременная память
"""
import os, json, base64, tempfile, subprocess, re, sqlite3, httpx, uuid, asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from collections import OrderedDict
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import soundfile as sf
from pydub import AudioSegment
from faster_whisper import WhisperModel

# ========== КОНФИГ ==========
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
USE_GPU = torch.cuda.is_available()
WHISPER_MODEL = "small"
HISTORY_DIR = "conversations"
MEMORY_DB = "long_term_memory.db"
ARENA_DB = "arena.db"

DEFAULT_MODELS = ["llama3.2:latest", "ruadapt-qwen2.5-14b:latest"]
JUDGE_MODEL = "ruadapt-qwen2.5-14b:latest"
TEMPERATURE = 0.8
MAX_TURNS = 8
MAX_TOKENS = 400
MAX_TOKENS_JUDGE = 800

os.makedirs(HISTORY_DIR, exist_ok=True)

asr_model = None
tts_model = None
device = torch.device('cuda' if USE_GPU else 'cpu')

# ========== LRU-КЭШ ДЛЯ TTS ==========
class LRUCache:
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size
    def get(self, key: str) -> str | None:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    def set(self, key: str, value: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value

tts_cache = LRUCache(max_size=100)

# ========== БАЗА ДАННЫХ АРЕНЫ ==========
def init_arena_db():
    conn = sqlite3.connect(ARENA_DB)
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
    print(f"✅ БД Арены: {ARENA_DB}")

def create_arena_session(sid, ma, mb, topic):
    conn = sqlite3.connect(ARENA_DB)
    conn.execute('INSERT INTO arena_sessions (id,model_a,model_b,topic) VALUES (?,?,?,?)',(sid,ma,mb,topic))
    conn.commit(); conn.close()

def save_arena_message(sid, model, content, turn):
    conn = sqlite3.connect(ARENA_DB)
    conn.execute('INSERT INTO arena_messages (session_id,model,content,turn_num) VALUES (?,?,?,?)',(sid,model,content,turn))
    conn.execute('UPDATE arena_sessions SET turns_count=turns_count+1 WHERE id=?',(sid,))
    conn.commit(); conn.close()

def end_arena_session(sid, status, verdict=None):
    conn = sqlite3.connect(ARENA_DB)
    conn.execute('UPDATE arena_sessions SET ended_at=CURRENT_TIMESTAMP,status=?,judge_verdict=? WHERE id=?',(status,verdict,sid))
    conn.commit(); conn.close()

def get_arena_history(sid):
    conn = sqlite3.connect(ARENA_DB)
    rows = conn.execute('SELECT model,content,turn_num,created_at FROM arena_messages WHERE session_id=? ORDER BY turn_num',(sid,)).fetchall()
    conn.close()
    return [{"model":r[0],"content":r[1],"turn":r[2],"time":r[3]} for r in rows]

def get_all_arena_sessions(limit=50):
    conn = sqlite3.connect(ARENA_DB)
    rows = conn.execute('SELECT id,model_a,model_b,topic,started_at,ended_at,turns_count,status,judge_verdict FROM arena_sessions ORDER BY started_at DESC LIMIT ?',(limit,)).fetchall()
    conn.close()
    return [{"id":r[0],"model_a":r[1],"model_b":r[2],"topic":r[3],"started":r[4],"ended":r[5],"turns":r[6],"status":r[7],"verdict":r[8]} for r in rows]

# ========== БАЗА ПАМЯТИ ==========
def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fact_type TEXT, content TEXT, keywords TEXT,
        conversation_id TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_message TEXT, bot_response TEXT, embedding TEXT,
        conversation_id TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit(); conn.close()
    print(f"✅ БД Памяти: {MEMORY_DB}")

def extract_facts(text: str) -> list:
    facts = []
    patterns = [
        r'(?:меня зовут|зовут|называют|я\s+—\s+)([А-Яа-яA-Za-z\s-]+)',
        r'(?:я из|живу в|нахожусь в|в городе|локация\s+)([А-Яа-яA-Za-z\s-]+)',
        r'(?:работаю|занимаюсь|делаю проект)\s+(?:над|в|с)?([А-Яа-яA-Za-z0-9\s-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            if len(val) > 1 and val.lower() not in ['привет', 'тест', 'да', 'нет']:
                facts.append(("info", val))
                break
    return facts

def save_fact(fact_type: str, content: str, conversation_id: str):
    conn = sqlite3.connect(MEMORY_DB)
    existing = conn.execute('SELECT id FROM facts WHERE fact_type=? AND content=?', (fact_type, content)).fetchone()
    if not existing:
        conn.execute('INSERT INTO facts (fact_type, content, keywords, conversation_id) VALUES (?, ?, ?, ?)',
                    (fact_type, content, content.lower()[:100], conversation_id))
        conn.commit()
    conn.close()

def get_relevant_facts(query: str, limit: int = 5) -> list:
    conn = sqlite3.connect(MEMORY_DB)
    keywords = [k for k in query.lower().split() if len(k) > 3]
    facts = []
    for kw in keywords:
        rows = conn.execute('SELECT fact_type, content FROM facts WHERE keywords LIKE ? LIMIT ?', (f'%{kw}%', limit)).fetchall()
        for row in rows:
            facts.append({"type": row[0], "content": row[1]})
    conn.close()
    seen, unique = set(), []
    for f in facts:
        key = f"{f['type']}:{f['content']}"
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique[:limit]

def save_memory(user_message: str, bot_response: str, conversation_id: str):
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute('INSERT INTO memories (user_message, bot_response, conversation_id) VALUES (?, ?, ?)',
                (user_message[:500], bot_response[:500], conversation_id))
    conn.commit(); conn.close()

def get_similar_memories(query: str, limit: int = 3) -> list:
    conn = sqlite3.connect(MEMORY_DB)
    keywords = [k for k in query.lower().split() if len(k) > 3]
    memories = []
    for kw in keywords:
        rows = conn.execute('SELECT user_message, bot_response FROM memories WHERE user_message LIKE ? LIMIT ?', (f'%{kw}%', limit)).fetchall()
        for row in rows:
            memories.append({"user": row[0], "bot": row[1]})
    conn.close()
    return memories[:limit]

# ========== МОДЕЛИ ==========
def get_available_models():
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get('http://localhost:11434/api/tags')
        if resp.status_code == 200:
            data = resp.json()
            models = sorted(set(m.get('name') or m.get('model') for m in data.get('models', []) if m.get('name') or m.get('model')))
            return models
    except Exception as e:
        print(f"Ошибка получения моделей: {e}")
    return DEFAULT_MODELS

# ========== БЕСЕДЫ (ЧАТ) ==========
def get_all_conversations():
    conversations = []
    for fn in os.listdir(HISTORY_DIR):
        if fn.endswith('.json'):
            try:
                with open(os.path.join(HISTORY_DIR, fn), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                title = "Новый диалог"
                for msg in data:
                    if msg.get('role') == 'user':
                        title = msg['content'][:50] + ('...' if len(msg['content']) > 50 else '')
                        break
                conversations.append({"id": fn[:-5], "title": title, "created_at": data[0].get('timestamp', '') if data else '', "message_count": len(data)})
            except: pass
    return sorted(conversations, key=lambda x: x.get('created_at', ''), reverse=True)

def load_conversation(conv_id: str) -> list:
    path = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return []
    return []

def save_conversation(conv_id: str, messages: list):
    with open(os.path.join(HISTORY_DIR, f"{conv_id}.json"), 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def create_new_conversation() -> str:
    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(4).hex()
    save_conversation(conv_id, [])
    return conv_id

def add_message(conv_id: str, role: str, content: str, model: str = None):
    messages = load_conversation(conv_id)
    messages.append({"role": role, "content": content, "model": model, "timestamp": datetime.now().isoformat()})
    save_conversation(conv_id, messages)

def delete_conversation(conv_id: str):
    path = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    if os.path.exists(path): os.unlink(path)

# ========== ASR ==========
def init_asr():
    global asr_model
    if asr_model is None:
        device_asr = "cuda" if USE_GPU else "cpu"
        compute_type = "float16" if USE_GPU else "int8"
        asr_model = WhisperModel(WHISPER_MODEL, device=device_asr, compute_type=compute_type)
        print(f"✅ Whisper на {device_asr}")

def speech_to_text(audio_bytes: bytes) -> str:
    if len(audio_bytes) < 100: return ""
    if asr_model is None: init_asr()
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            tmp.write(audio_bytes)
            webm = tmp.name
        wav = webm.replace('.webm', '.wav')
        subprocess.run(['ffmpeg', '-i', webm, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-loglevel', 'quiet', '-y', wav], capture_output=True)
        if not os.path.exists(wav): return ""
        segments, _ = asr_model.transcribe(wav, language="ru", beam_size=3)
        text = " ".join(seg.text for seg in segments).strip()
        os.unlink(webm); os.unlink(wav)
        return text if len(text) >= 3 else ""
    except Exception as e:
        print(f"ASR error: {e}")
        return ""

# ========== TTS ==========
def init_tts():
    global tts_model
    if tts_model is None:
        tts_model, _ = torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v4_ru')
        tts_model.to(device)
        print(f"✅ TTS на {device}")

def text_to_speech(text: str) -> str | None:
    if not text or len(text.strip()) < 2: return None
    if tts_model is None: init_tts()
    cached = tts_cache.get(text)
    if cached: return cached
    try:
        if len(text) > 400: text = text[:400]
        audio = tts_model.apply_tts(text=text, speaker='eugene', sample_rate=24000, put_accent=True, put_yo=True)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio.cpu().numpy(), 24000)
            wav = tmp.name
        ogg = wav.replace('.wav', '.ogg')
        AudioSegment.from_wav(wav).export(ogg, format="ogg")
        with open(ogg, 'rb') as f: b64 = base64.b64encode(f.read()).decode()
        os.unlink(wav); os.unlink(ogg)
        tts_cache.set(text, b64)
        return b64
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# ========== LLM ДЛЯ ЧАТА ==========
async def ask_llm_with_memory(message: str, history: list, model: str, conv_id: str) -> str:
    try:
        facts = get_relevant_facts(message)
        similar = get_similar_memories(message)
        memory_context = ""
        if facts:
            memory_context += "\n[ПАМЯТЬ]:\n" + "\n".join(f"- {f['type']}: {f['content']}" for f in facts) + "\n"
        if similar:
            memory_context += "\n[ИСТОРИЯ]:\n" + "\n".join(f"- Вы: {m['user']}\n  Я: {m['bot']}" for m in similar) + "\n"
        context = [{"role": "user" if m["role"]=="user" else "assistant", "content": m["content"][:500]} for m in history[-10:]]
        system = f"""Ты — голосовой ассистент с памятью.{memory_context}
Отвечай кратко, по-русски, дружелюбно."""
        context.insert(0, {"role": "system", "content": system})
        context.append({"role": "user", "content": message[:500]})
        payload = {"model": model, "messages": context, "stream": False, "options": {"temperature": 0.7, "num_predict": 512}}
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get('message', {}).get('content', '').strip()
            if not content: content = "Извините, не удалось получить ответ."
            for ft, fc in extract_facts(message): save_fact(ft, fc, conv_id)
            save_memory(message[:500], content[:500], conv_id)
            return content
        return f"Ошибка API: {resp.status_code}"
    except Exception as e:
        print(f"LLM error: {e}")
        return f"Ошибка: {str(e)[:100]}"

# ========== ПРОМПТЫ ДЛЯ АРЕНЫ ==========
def _get_system_prompt(current: str, opponent: str) -> str:
    return f"""Ты — Модель {current}. Интеллектуальный спор с Модель {opponent}.

📜 **ЖЁСТКИЕ ПРАВИЛА:**
1. ТОЛЬКО РУССКИЙ ЯЗЫК. Английские слова ЗАПРЕЩЕНЫ.
2. ЗАПРЕЩЕНО использовать эмодзи.
3. НЕ ПОВТОРЯЙ структуру и фразы собеседника.
4. Краткость: 2-4 предложения.
5. ИЗБЕГАЙ шаблонов: "Я согласен", "Интересно".
6. НЕ ДОБАВЛЯЙ префиксы [A]:, [Б]: в начало ответа."""

def _get_judge_prompt(topic: str, dialog: list) -> str:
    dialog_text = chr(10).join(f"[{m.get('speaker', '??')}]: {m.get('content', '')}" for m in dialog if m.get('speaker'))
    return f"""Ты — независимый эксперт. Проанализируй диалог на тему: "{topic}"

📋 ДИАЛОГ:
{dialog_text}

📝 **ФОРМАТ ОТВЕТА (СТРОГО):**
Модель А сильные стороны: [текст]
Модель А слабые стороны: [текст]
Модель Б сильные стороны: [текст]
Модель Б слабые стороны: [текст]
ВЕРДИКТ: [Победа А / Победа Б / Ничья]
ОБОСНОВАНИЕ: [2-3 предложения]

⚠️ Только русский язык. Без Markdown и эмодзи."""

def _clean(content: str) -> str:
    emoji_pattern = re.compile("[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF" "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
    content = emoji_pattern.sub('', content)
    content = re.sub(r'^\s*\[?[АAБB]\]?\s*:?\s*', '', content.strip())
    return content.strip()

def _build_prompt(history: list, current: str, opponent: str) -> list:
    sys = _get_system_prompt(current, opponent)
    messages = [{"role":"system","content":sys}]
    anti_repeat = "⚠️ Скажи что-то НОВОЕ. Не повторяй оппонента."
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
    return messages

async def _call_ollama(model: str, messages: list, max_tokens: int = MAX_TOKENS) -> str:
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(OLLAMA_URL, json={"model": model, "messages": messages, "stream": False, "options": {"temperature": TEMPERATURE, "num_predict": max_tokens}})
        if resp.status_code == 200:
            return _clean(resp.json().get("message", {}).get("content", "").strip())
    except Exception as e:
        print(f"❌ Ollama error: {e}")
    return ""

async def _call_judge(topic: str, dialog: list) -> str:
    if not dialog: return "⚠️ Диалог пуст"
    messages = [{"role":"user","content": _get_judge_prompt(topic, dialog)}]
    verdict = await _call_ollama(JUDGE_MODEL, messages, max_tokens=MAX_TOKENS_JUDGE)
    return verdict if verdict else "⚠️ Вердикт не получен"

# ========== СЕССИИ АРЕНЫ ==========
ACTIVE_ARENA = {}

# ========== FASTAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("="*50)
    print("🚀 AI Hub: Арена + Голосовой Чат")
    init_arena_db()
    init_memory_db()
    init_asr()
    init_tts()
    print(f"📋 Модели: {get_available_models()}")
    print("="*50)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return FileResponse("index.html") if os.path.exists("index.html") else HTMLResponse("<h1>Index.html not found</h1>")

@app.get("/api/models")
async def list_models():
    return JSONResponse({"models": get_available_models()})

# ========== ЧАТ ЭНДПОИНТЫ ==========
@app.get("/api/conversations")
async def conversations():
    return JSONResponse({"conversations": get_all_conversations()})

@app.post("/api/conversation/new")
async def new_conv():
    return JSONResponse({"conversation_id": create_new_conversation()})

@app.get("/api/conversation/{conv_id}")
async def get_conv(conv_id: str):
    return JSONResponse({"conversation_id": conv_id, "messages": load_conversation(conv_id)})

@app.delete("/api/conversation/{conv_id}")
async def del_conv(conv_id: str):
    delete_conversation(conv_id)
    return JSONResponse({"status": "ok"})

@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        message = body.get('message', '').strip()
        conv_id = body.get('conversation_id', '') or create_new_conversation()
        model = body.get('model', DEFAULT_MODELS[0])
        if not message: return JSONResponse({"error": "Пустое сообщение"}, status_code=400)
        history = load_conversation(conv_id)
        add_message(conv_id, "user", message, model)
        response = await ask_llm_with_memory(message, history, model, conv_id)
        add_message(conv_id, "assistant", response, model)
        audio = text_to_speech(response)
        return JSONResponse({"response": response, "audio": audio, "conversation_id": conv_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/voice")
async def voice(audio: UploadFile = File(...), conv_id: str = Form("")):
    try:
        text = speech_to_text(await audio.read())
        return JSONResponse({"text": text or None, "conversation_id": conv_id})
    except Exception as e:
        return JSONResponse({"text": None})

@app.post("/api/tts")
async def tts_only(request: Request):
    try:
        text = (await request.json()).get('text', '').strip()
        if not text: return JSONResponse({"error": "Нет текста"}, status_code=400)
        audio = text_to_speech(text)
        return JSONResponse({"audio": audio})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ========== АРЕНА ЭНДПОИНТЫ ==========
@app.get("/api/sessions")
async def list_sessions():
    return JSONResponse({"sessions": get_all_arena_sessions()})

@app.get("/api/session/{sid}")
async def get_session(sid: str):
    return JSONResponse({"session_id": sid, "messages": get_arena_history(sid)})

@app.post("/api/arena/start")
async def start_arena(request: Request):
    data = await request.json()
    sid = str(uuid.uuid4())
    model_a = data.get("model_a", DEFAULT_MODELS[0])
    model_b = data.get("model_b", DEFAULT_MODELS[1])
    topic = data.get("topic", "")
    create_arena_session(sid, model_a, model_b, topic)
    ACTIVE_ARENA[sid] = {
        "history": [{"role":"user","content": topic}],
        "model_a": model_a,
        "model_b": model_b,
        "stop": False,
        "max_turns": data.get("max_turns", MAX_TURNS),
        "turn": 0
    }
    print(f"🎭 Старт Арены: {sid}")
    return JSONResponse({"session_id": sid})

@app.get("/api/arena/stream/{sid}")
async def stream_arena(sid: str):
    if sid not in ACTIVE_ARENA:
        raise HTTPException(404)
    sess = ACTIVE_ARENA[sid]
    
    async def gen():
        while sess["turn"] < sess["max_turns"] and not sess["stop"]:
            for model_key, model_name in [("A", sess["model_a"]), ("B", sess["model_b"])]:
                if sess["stop"]: break
                yield f"data: {json.dumps({'type':'typing','model':model_key})}\n\n"
                messages = _build_prompt(sess["history"], model_key, "B" if model_key == "A" else "A")
                content = await _call_ollama(model_name, messages)
                if content and not sess["stop"]:
                    sess["history"].append({"role":"assistant","content": content, "speaker": model_key})
                    save_arena_message(sid, model_key, content, sess["turn"] + 1)
                    yield f"data: {json.dumps({'type':'message','model':model_key,'content':content})}\n\n"
                await asyncio.sleep(0.3)
            sess["turn"] += 1
        
        status = "stopped" if sess["stop"] else "finished"
        topic = sess["history"][0]["content"] if sess["history"] else "Без темы"
        dialog = [m for m in sess["history"] if m.get("speaker")]
        if dialog and status == "finished":
            yield f"data: {json.dumps({'type':'judge_thinking'})}\n\n"
            verdict = await _call_judge(topic, dialog)
            yield f"data: {json.dumps({'type':'verdict','content':verdict})}\n\n"
            end_arena_session(sid, status, verdict)
        else:
            end_arena_session(sid, status, None)
        yield f"data: {json.dumps({'type':status})}\n\n"
        if sid in ACTIVE_ARENA: del ACTIVE_ARENA[sid]
    
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

@app.post("/api/arena/stop/{sid}")
async def stop_arena(sid: str):
    if sid in ACTIVE_ARENA:
        ACTIVE_ARENA[sid]["stop"] = True
        end_arena_session(sid, "stopped", None)
    return JSONResponse({"status":"ok"})

@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    conn = sqlite3.connect(ARENA_DB)
    conn.execute('DELETE FROM arena_messages WHERE session_id=?', (sid,))
    conn.execute('DELETE FROM arena_sessions WHERE id=?', (sid,))
    conn.commit(); conn.close()
    if sid in ACTIVE_ARENA: del ACTIVE_ARENA[sid]
    return JSONResponse({"status":"ok"})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico") if os.path.exists("favicon.ico") else JSONResponse(status_code=204)

if __name__ == "__main__":
    print("🏟️ AI Hub запуск на http://0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
