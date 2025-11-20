import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer
from threading import Thread


# Load model và tokenizer
MODEL_NAME = "WeiboAI/VibeThinker-1.5B"

# Khởi tạo model và tokenizer một lần khi app khởi động
print("Đang tải model VibeThinker-1.5B...")
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
    Hàm xử lý chat với model VibeThinker-1.5B
    Model này được tối ưu cho các bài toán toán học và lập trình cạnh tranh
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
    
    # Cấu hình generation theo khuyến nghị của model
    generation_config = GenerationConfig(
        max_new_tokens=min(max_tokens, 40960),  # Giới hạn tối đa 40960 theo khuyến nghị
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=None,  # Bỏ qua top_k theo khuyến nghị
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
            value="You are a helpful assistant specialized in competitive-style math and algorithm coding problems.", 
            label="System message"
        ),
        gr.Slider(
            minimum=1, 
            maximum=40960, 
            value=2048, 
            step=1, 
            label="Max new tokens"
        ),
        gr.Slider(
            minimum=0.1, 
            maximum=2.0, 
            value=0.6, 
            step=0.1, 
            label="Temperature (khuyến nghị: 0.6 hoặc 1.0)"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    title="VibeThinker-1.5B Chat",
    description="Model tối ưu cho các bài toán toán học và lập trình cạnh tranh. Khuyến nghị đặt câu hỏi bằng tiếng Anh để có kết quả tốt nhất.",
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
