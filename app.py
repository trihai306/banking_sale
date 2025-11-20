import gradio as gr
import torch
import whisper
import edge_tts
import asyncio
import io
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer
from threading import Thread
from queue import Queue
import tempfile
import os


# Load model và tokenizer
MODEL_NAME = "hainguyen306201/bank-model"

# Khởi tạo model và tokenizer một lần khi app khởi động
model = None
tokenizer = None
whisper_model = None

def load_models():
    """Load tất cả models"""
    global model, tokenizer, whisper_model
    try:
        print("Đang tải model bank-model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("Model đã được tải thành công!")
        
        # Load Whisper model cho STT
        print("Đang tải Whisper model cho STT...")
        whisper_model = whisper.load_model("base")
        print("Whisper model đã được tải thành công!")
        return True
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return False

# Load models
load_models()

# System message mặc định
DEFAULT_SYSTEM_MESSAGE = "You are a helpful banking and finance assistant specialized in providing financial advice and banking services information. Respond in Vietnamese when the user speaks Vietnamese."


def speech_to_text(audio):
    """
    STT: Chuyển đổi audio thành text
    """
    if audio is None:
        return None
    
    if whisper_model is None:
        return "Lỗi: Whisper model chưa được tải. Vui lòng đợi..."
    
    try:
        # Whisper xử lý audio file
        result = whisper_model.transcribe(audio, language="vi")
        text = result["text"].strip()
        return text
    except Exception as e:
        print(f"Lỗi STT: {e}")
        return None


def generate_response_stream(
    message,
    history: list,
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    LLM: Tạo response từ LLM với streaming
    """
    if model is None or tokenizer is None:
        yield "Lỗi: Model chưa được tải. Vui lòng đợi hoặc refresh trang..."
        return
    
    # Chuẩn bị messages
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    else:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_MESSAGE})
    
    # Thêm lịch sử chat
    messages.extend(history)
    
    # Thêm message hiện tại
    messages.append({"role": "user", "content": message})
    
    # Áp dụng chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize input
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Cấu hình generation
    generation_config = GenerationConfig(
        max_new_tokens=min(max_tokens, 16384),
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
    )
    
    # Tạo streamer để stream response theo thời gian thực
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Cấu hình generation với streamer
    generation_kwargs = {
        **model_inputs,
        "generation_config": generation_config,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    
    # Chạy generation trong thread riêng
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream response
    response_text = ""
    for new_text in streamer:
        response_text += new_text
        yield response_text


async def text_to_speech_async(text, voice="vi-VN-HoaiMyNeural"):
    """
    TTS: Chuyển đổi text thành audio (async)
    """
    try:
        # Sử dụng edge-tts để tạo audio
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        # Tạo file audio tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(audio_data)
            return tmp_file.name
    except Exception as e:
        print(f"Lỗi TTS: {e}")
        return None


def text_to_speech(text):
    """
    TTS wrapper (sync)
    """
    if not text:
        return None
    try:
        return asyncio.run(text_to_speech_async(text))
    except Exception as e:
        print(f"Lỗi TTS: {e}")
        return None


def process_voice_input(
    audio,
    history: list,
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    Xử lý voice input: STT → LLM → TTS
    """
    # STT: Chuyển audio thành text
    user_text = speech_to_text(audio)
    if not user_text:
        return history, None, "Không thể nhận diện giọng nói. Vui lòng thử lại."
    
    # Thêm user message vào history
    history.append({"role": "user", "content": user_text})
    
    # LLM: Tạo response với streaming
    response_text = ""
    for partial_text in generate_response_stream(
        user_text, history[:-1], system_message, max_tokens, temperature, top_p
    ):
        response_text = partial_text
    
    # Thêm assistant response vào history
    history.append({"role": "assistant", "content": response_text})
    
    # TTS: Chuyển response thành audio
    audio_output = text_to_speech(response_text)
    
    return history, audio_output, response_text


def process_text_input(
    message,
    history: list,
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    Xử lý text input: LLM → TTS (optional)
    """
    if not message:
        return history, None, ""
    
    # Thêm user message vào history
    history.append({"role": "user", "content": message})
    
    # LLM: Tạo response với streaming
    response_text = ""
    for partial_text in generate_response_stream(
        message, history[:-1], system_message, max_tokens, temperature, top_p
    ):
        response_text = partial_text
    
    # Thêm assistant response vào history
    history.append({"role": "assistant", "content": response_text})
    
    return history, response_text


def chat_with_voice(
    audio,
    message,
    history: list,
    system_message,
    max_tokens,
    temperature,
    top_p,
    enable_tts,
):
    """
    Hàm chính xử lý cả voice và text input
    """
    if audio is not None:
        # Xử lý voice input
        history, audio_output, response_text = process_voice_input(
            audio, history, system_message, max_tokens, temperature, top_p
        )
        return history, response_text, audio_output if enable_tts else None
    elif message:
        # Xử lý text input
        history, response_text = process_text_input(
            message, history, system_message, max_tokens, temperature, top_p
        )
        audio_output = text_to_speech(response_text) if enable_tts else None
        return history, response_text, audio_output
    else:
        return history, "", None


def stream_chat_response(
    message,
    history: list,
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    Stream response cho text chat
    """
    if not message:
        return history, ""
    
    # Thêm user message vào history
    history.append({"role": "user", "content": message})
    
    # Stream response từ LLM
    response_text = ""
    for partial_text in generate_response_stream(
        message, history[:-1], system_message, max_tokens, temperature, top_p
    ):
        response_text = partial_text
        # Cập nhật history với partial response
        temp_history = history.copy()
        temp_history.append({"role": "assistant", "content": response_text})
        yield temp_history, response_text
    
    # Cập nhật history cuối cùng
    history.append({"role": "assistant", "content": response_text})


# Tạo Gradio interface
with gr.Blocks(title="Bank Model Voice Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏦 Bank Model Voice Chat
    
    Ứng dụng tư vấn ngân hàng và tài chính với hỗ trợ giọng nói.
    
    **Kiến trúc**: STT (Speech-to-Text) → LLM → TTS (Text-to-Speech)
    
    - 🎤 **Voice Input**: Nói vào microphone để đặt câu hỏi
    - 💬 **Text Input**: Gõ câu hỏi bằng văn bản
    - 🔊 **Voice Output**: Nghe câu trả lời bằng giọng nói (tùy chọn)
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_copy_button=True,
            )
            
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Nói vào đây",
                    show_label=True,
                )
                text_input = gr.Textbox(
                    label="💬 Hoặc gõ câu hỏi",
                    placeholder="Nhập câu hỏi của bạn...",
                    lines=2,
                )
            
            with gr.Row():
                submit_voice_btn = gr.Button("🎤 Gửi (Voice)", variant="primary")
                submit_text_btn = gr.Button("💬 Gửi (Text)", variant="primary")
                clear_btn = gr.Button("🗑️ Xóa lịch sử", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Cài đặt")
            
            system_msg = gr.Textbox(
                value=DEFAULT_SYSTEM_MESSAGE,
                label="System Message",
                lines=3,
            )
            
            max_tokens = gr.Slider(
                minimum=1,
                maximum=16384,
                value=2048,
                step=1,
                label="Max Tokens",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            )
            
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.8,
                step=0.05,
                label="Top-p",
            )
            
            enable_tts = gr.Checkbox(
                value=True,
                label="🔊 Bật TTS (Text-to-Speech)",
            )
            
            audio_output = gr.Audio(
                label="🔊 Câu trả lời bằng giọng nói",
                type="filepath",
                autoplay=True,
            )
            
            with gr.Accordion("ℹ️ Thông tin", open=False):
                gr.Markdown("""
                **Model**: hainguyen306201/bank-model
                - Fine-tuned từ Qwen3-4B-Instruct-2507
                - Chuyên về tư vấn ngân hàng và tài chính
                - Hỗ trợ đa ngôn ngữ (ưu tiên tiếng Việt)
                
                **STT**: OpenAI Whisper (base)
                **TTS**: Edge-TTS (vi-VN-HoaiMyNeural)
                """)
    
    # State để lưu lịch sử chat
    history_state = gr.State(value=[])
    
    # Event handlers
    def submit_voice(audio, history, system_msg, max_tok, temp, top_p_val, tts_enabled):
        if audio is None:
            return history, "", None, history
        new_history, response_text, audio_out = chat_with_voice(
            audio, None, history, system_msg, max_tok, temp, top_p_val, tts_enabled
        )
        return new_history, response_text, audio_out, new_history
    
    def submit_text_stream(message, history, system_msg, max_tok, temp, top_p_val, tts_enabled):
        if not message:
            return history, "", None, history
        
        # Thêm user message vào history
        history.append({"role": "user", "content": message})
        
        # Stream response từ LLM
        response_text = ""
        for partial_text in generate_response_stream(
            message, history[:-1], system_msg, max_tok, temp, top_p_val
        ):
            response_text = partial_text
            # Cập nhật history với partial response
            temp_history = history.copy()
            temp_history.append({"role": "assistant", "content": response_text})
            yield temp_history, "", None, temp_history
        
        # Cập nhật history cuối cùng
        history.append({"role": "assistant", "content": response_text})
        
        # Tạo audio nếu TTS được bật
        audio_out = text_to_speech(response_text) if tts_enabled else None
        yield history, "", audio_out, history
    
    def clear_chat():
        return [], "", None, []
    
    # Bind events
    submit_voice_btn.click(
        fn=submit_voice,
        inputs=[audio_input, history_state, system_msg, max_tokens, temperature, top_p, enable_tts],
        outputs=[chatbot, text_input, audio_output, history_state],
    ).then(
        fn=lambda: None,
        outputs=[audio_input],
    )
    
    submit_text_btn.click(
        fn=submit_text_stream,
        inputs=[text_input, history_state, system_msg, max_tokens, temperature, top_p, enable_tts],
        outputs=[chatbot, text_input, audio_output, history_state],
    ).then(
        fn=lambda: "",
        outputs=[text_input],
    )
    
    text_input.submit(
        fn=submit_text_stream,
        inputs=[text_input, history_state, system_msg, max_tokens, temperature, top_p, enable_tts],
        outputs=[chatbot, text_input, audio_output, history_state],
    ).then(
        fn=lambda: "",
        outputs=[text_input],
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, text_input, audio_output, history_state],
    )


if __name__ == "__main__":
    # Kiểm tra models đã được load chưa
    if model is None or tokenizer is None or whisper_model is None:
        print("⚠️ Cảnh báo: Một số models chưa được tải. App vẫn sẽ chạy nhưng có thể gặp lỗi.")
    
    # Hugging Face Spaces sẽ tự động xử lý server configuration
    demo.launch()
