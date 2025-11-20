#!/bin/bash

# Script để push code lên GitHub
# Sử dụng: ./push_to_github.sh YOUR_GITHUB_USERNAME YOUR_REPO_NAME

GITHUB_USERNAME=${1:-hainguyen306201}
REPO_NAME=${2:-banking_sale}

echo "🚀 Đang đẩy code lên GitHub..."
echo "   Username: $GITHUB_USERNAME"
echo "   Repository: $REPO_NAME"
echo ""

# Kiểm tra xem remote github đã tồn tại chưa
if git remote | grep -q "^github$"; then
    echo "⚠️  Remote 'github' đã tồn tại, đang xóa..."
    git remote remove github
fi

# Thêm remote GitHub
echo "📝 Đang thêm remote GitHub..."
git remote add github "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# Push code
echo "⬆️  Đang push code lên GitHub..."
if git push -u github main; then
    echo ""
    echo "✅ Đã push code lên GitHub thành công!"
    echo "   Repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
    echo "📋 Bạn có thể sử dụng URL này trong Colab:"
    echo "   https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
else
    echo ""
    echo "❌ Lỗi khi push code!"
    echo "   Vui lòng kiểm tra:"
    echo "   1. Repository đã được tạo trên GitHub chưa?"
    echo "   2. Bạn có quyền push vào repository không?"
    echo "   3. Username và repository name có đúng không?"
    exit 1
fi

