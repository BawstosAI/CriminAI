.PHONY: audio_ws text_ws llm stt_tts stop_audio_ws stop_text_ws stop_llm stop_stt_tts frontend

audio_ws:
	@echo "Building Audio WebSocket Service..."
	bash -c "source /home/akpale/hackatons/prep_first_hackaton_ever/CriminAI/.venv/bin/activate && \
	uv run python /home/akpale/hackatons/prep_first_hackaton_ever/CriminAI/src/audio_websocket_server.py" &

text_ws:
	@echo "Building Text WebSocket Service..."
	bash -c "source /home/akpale/hackatons/prep_first_hackaton_ever/CriminAI/.venv/bin/activate && \
	uv run python /home/akpale/hackatons/prep_first_hackaton_ever/CriminAI/src/text_websocket_server.py" &

llm:
	@echo "starting LLM Server..."
	bash -c "source /home/akpale/hackatons/prep_first_hackaton_ever/CriminAI/.venv/bin/activate && \
	uv run python /home/akpale/hackatons/prep_first_hackaton_ever/CriminAI/src/gemini_openai_server.py" &

stt_tts:
	@echo "starting STT and TTS Server..."
	bash -c "source /home/akpale/hackatons/prep_first_hackaton_ever/CriminAI/.venv/bin/activate && \
	moshi-server worker --config /home/akpale/hackatons/prep_first_hackaton_ever/CriminAI/delayed-streams-modeling/configs/config-unified-stt-tts.toml" &

stop_audio_ws:
	@echo "Stopping Audio WebSocket Service..."
	-pkill -f audio_websocket_server.py

stop_text_ws:
	@echo "Stopping Text WebSocket Service..."
	-pkill -f text_websocket_server.py

stop_llm:
	@echo "Stopping LLM Server..."
	-pkill -f gemini_openai_server.py

stop_stt_tts:
	@echo "Stopping STT and TTS Server..."
	-pkill -f moshi-server

all: audio_ws text_ws llm stt_tts
stop: stop_audio_ws stop_text_ws stop_llm stop_stt_tts

audio: audio_ws llm stt_tts
text: text_ws llm stt_tts

frontend:
	@echo "Starting Frontend..."
	cd /home/akpale/hackatons/prep_first_hackaton_ever/CriminAI/frontend && npm run dev