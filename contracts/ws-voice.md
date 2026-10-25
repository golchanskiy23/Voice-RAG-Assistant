# WebSocket `/ws/voice` (FR-5.3)

Голосовой диалог в реальном времени: клиент шлёт аудио, сервер — распознанный текст, текст ответа по предложениям
и аудио ответа по мере синтеза.

## Клиент → сервер

| Кадр | Содержимое |
|---|---|
| бинарный | PCM 16-bit little-endian, 16 кГц, mono, любой длины (рекомендуется 20–100 мс) — при `format=pcm16`; байты файла — при `format=ogg` / `wav` |
| `{"type": "start", "format": "pcm16" \| "ogg" \| "wav"}` | Начало сессии/записи. По умолчанию `pcm16`. Необязателен для `pcm16` |
| `{"type": "end"}` | Конец фразы вручную (кнопка «стоп»). Для `ogg`/`wav` обязателен: файл декодируется ffmpeg и распознаётся целиком |
| `{"type": "text", "question": "..."}` | Вопрос текстом без STT (отладка, eval задержки) |
| `{"type": "cancel"}` | Прервать текущий ответ |

Для `pcm16` конец фразы определяет VAD на сервере (FR-5.2), `end` не нужен.
Если во время ответа VAD слышит новую речь — ответ прерывается (barge-in, FR-5.6).

## Сервер → клиент

Все события — текстовые JSON-кадры.

| Событие | Поля | Когда |
|---|---|---|
| `ready` | `sample_rate_in`, `sample_rate_out`, `stt`, `tts` | Сразу после подключения |
| `partial_transcript` | `text` | Во время речи, примерно раз в `STT_PARTIAL_INTERVAL` секунд аудио |
| `final_transcript` | `text`, `audio_seconds` | После конца фразы, перед ответом |
| `answer_text` | `text` | Очередное предложение ответа (то, что озвучивается; без нормализации для TTS) |
| `audio_chunk` | `seq`, `sample_rate`, `format: "pcm16"`, `data` (base64) | Аудио очередного предложения |
| `sources` | `sources: [{filename, page, section, quote}]`, `confidence` | После текста ответа |
| `done` | `latency: {stage: ms}` | Ответ завершён; метки этапов от конца речи |
| `interrupted` | `reason: "barge_in" \| "cancel"` | Текущий ответ прерван |
| `error` | `error: {code, message, request_id}` | Ошибка; соединение остаётся открытым |

### `done.latency`

Миллисекунды от конца речи (`speech_end`; для `text` — от получения вопроса):

`stt` — текст распознан, `retrieve` — поиск завершён, `first_token` — первый токен LLM, `first_sentence` — первое
предложение готово к синтезу, `first_audio` — первый `audio_chunk` отправлен, `total` — всё отправлено.

## Пример

```
→ {"type": "start", "format": "pcm16"}
→ <binary 3200 B> × N
← {"type": "partial_transcript", "text": "какой максимальный"}
← {"type": "final_transcript", "text": "Какой максимальный размер файла?", "audio_seconds": 2.1}
← {"type": "answer_text", "text": "Размер файла не должен превышать 20 МБ."}
← {"type": "audio_chunk", "seq": 0, "sample_rate": 24000, "format": "pcm16", "data": "..."}
← {"type": "sources", "sources": [...], "confidence": 0.9}
← {"type": "done", "latency": {"stt": 310, "retrieve": 352, "first_token": 980, "first_sentence": 1210, "first_audio": 1530, "total": 2400}}
```
