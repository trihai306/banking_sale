import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer
from threading import Thread


# Load model và tokenizer
MODEL_NAME = "hainguyen306201/bank-model"

# Khởi tạo model và tokenizer một lần khi app khởi động
print("Đang tải model bank-model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Model đã được tải thành công!")


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    """
    Hàm xử lý chat với model bank-model
    Model này được fine-tuned từ Qwen3-4B-Instruct-2507 cho các tác vụ liên quan đến ngân hàng và tài chính
    """
    # Chuẩn bị messages
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
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
    
    # Cấu hình generation theo khuyến nghị của Qwen3-4B-Instruct-2507 (base model)
    # Khuyến nghị: Temperature=0.7, TopP=0.8, TopK=20, MinP=0, max_new_tokens=16384
    generation_config = GenerationConfig(
        max_new_tokens=min(max_tokens, 16384),  # Giới hạn tối đa 16384 theo khuyến nghị
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=20,  # Khuyến nghị TopK=20
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


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(
            value="You are a helpful banking and finance assistant specialized in providing financial advice and banking services information.", 
            label="System message"
        ),
        gr.Slider(
            minimum=1, 
            maximum=16384, 
            value=16384, 
            step=1, 
            label="Max new tokens"
        ),
        gr.Slider(
            minimum=0.1, 
            maximum=2.0, 
            value=0.7, 
            step=0.1, 
            label="Temperature (khuyến nghị: 0.7)"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.8,
            step=0.05,
            label="Top-p (khuyến nghị: 0.8)",
        ),
    ],
    title="Bank Model Chat",
    description="Model bank-model được fine-tuned từ Qwen3-4B-Instruct-2507, chuyên về tư vấn ngân hàng và tài chính. Hỗ trợ đa ngôn ngữ và ngữ cảnh dài (256K tokens).",
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
