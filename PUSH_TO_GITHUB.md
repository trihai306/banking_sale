# Hướng dẫn đẩy dự án lên GitHub

## Bước 1: Tạo repository trên GitHub

1. Truy cập https://github.com/new
2. Điền thông tin:
   - **Repository name**: `banking_sale`
   - **Description**: Banking Sale Voice Chat - STT → LLM → TTS
   - **Visibility**: Public hoặc Private (tùy bạn)
   - **KHÔNG** tích vào "Initialize this repository with a README"
3. Click **Create repository**

## Bước 2: Push code lên GitHub

Sau khi tạo repository, chạy các lệnh sau trong terminal:

```bash
cd /Users/hainc/duan/AIcall/banking_sale

# Thêm remote GitHub
git remote add github https://github.com/hainguyen306201/banking_sale.git

# Push code lên GitHub
git push -u github main
```

Hoặc nếu repository của bạn có tên khác, thay đổi URL:

```bash
git remote add github https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u github main
```

## Bước 3: Xác nhận

Sau khi push thành công, bạn có thể:
- Truy cập repository tại: https://github.com/hainguyen306201/banking_sale
- Sử dụng URL này trong notebook Colab để clone dự án

## Sử dụng trong Colab

Sau khi đã push lên GitHub, bạn có thể clone trong Colab:

```python
!git clone https://github.com/hainguyen306201/banking_sale.git
```

Notebook đã được cập nhật để hỗ trợ clone từ GitHub tự động!

