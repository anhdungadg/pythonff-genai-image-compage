#!/bin/bash

# Script để kích hoạt môi trường ảo và cài đặt các thư viện cần thiết

# Kích hoạt môi trường ảo
echo "Kích hoạt môi trường ảo myenv..."
source myenv/bin/activate

# Cài đặt các thư viện từ requirements.txt
echo "Cài đặt các thư viện cần thiết..."
pip install -r requirements.txt

echo ""
echo "Môi trường ảo đã được kích hoạt và các thư viện đã được cài đặt."
echo "Để sử dụng môi trường ảo, hãy chạy: source myenv/bin/activate"
echo "Để thoát môi trường ảo, hãy chạy: deactivate"
