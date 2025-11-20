---
title: Banking Sale Voice Chat
emoji: 🏦
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: false
short_description: Tư vấn ngân hàng với voice chat (STT→LLM→TTS)
---

# 🏦 Banking Sale Voice Chat

Ứng dụng tư vấn ngân hàng và tài chính với hỗ trợ giọng nói, sử dụng kiến trúc **STT → LLM → TTS**.

## ✨ Tính năng

- 🎤 **Voice Input (STT)**: Nhận diện giọng nói bằng OpenAI Whisper
- 🤖 **LLM**: Sử dụng model `hainguyen306201/bank-model` (fine-tuned từ Qwen3-4B-Instruct-2507)
- 🔊 **Voice Output (TTS)**: Chuyển đổi text thành giọng nói bằng Edge-TTS
- 💬 **Text Chat**: Hỗ trợ chat bằng văn bản
- ⚡ **Streaming**: Stream response trực tiếp từ LLM

## 🏗️ Kiến trúc

```
Audio Input (Microphone)
    ↓
[STT: Whisper]
    ↓
Text Input
    ↓
[LLM: bank-model]
    ↓
Text Response (Streaming)
    ↓
[TTS: Edge-TTS]
    ↓
Audio Output
```

## 🚀 Sử dụng

1. **Voice Input**: Click vào microphone và nói câu hỏi của bạn
2. **Text Input**: Gõ câu hỏi vào ô text và nhấn Enter hoặc click "Gửi"
3. **Voice Output**: Bật TTS để nghe câu trả lời bằng giọng nói

## 📦 Dependencies

- `gradio>=5.42.0`: UI framework
- `transformers>=4.51.0`: Hugging Face transformers
- `torch`: PyTorch
- `openai-whisper`: Speech-to-Text
- `edge-tts`: Text-to-Speech
- `accelerate`: Model acceleration
- `huggingface_hub`: Hugging Face Hub integration

## 🎯 Model

- **Base Model**: `hainguyen306201/bank-model`
- **STT Model**: OpenAI Whisper (base)
- **TTS**: Edge-TTS (vi-VN-HoaiMyNeural)

## 📝 License

Apache-2.0
