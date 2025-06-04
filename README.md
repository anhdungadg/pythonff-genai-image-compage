# GenAI Image Compare

Công cụ đánh giá và so sánh chi tiết các mô hình AI tạo hình ảnh.

## Mô tả

GenAI Image Compare là một framework toàn diện để đánh giá và so sánh các mô hình AI tạo hình ảnh khác nhau như Amazon Nova Canvas và Stable Diffusion XL. Công cụ này cung cấp các phép đo định lượng và trực quan để giúp người dùng hiểu rõ điểm mạnh và điểm yếu của từng mô hình trong các tình huống khác nhau.

## Tính năng

- **Đánh giá đa mô hình**: So sánh Amazon Nova Canvas và Stable Diffusion XL
- **Đa dạng prompts**: Kiểm tra với các prompt đơn giản, trung bình, phức tạp và chứa nhiều văn bản
- **Đánh giá tự động**: Tính toán các chỉ số như CLIP score, thời gian tạo hình ảnh
- **Đánh giá con người**: Hỗ trợ thu thập và phân tích đánh giá từ người dùng
- **Báo cáo chi tiết**: Tạo báo cáo và biểu đồ trực quan
- **Tùy chỉnh linh hoạt**: Dễ dàng mở rộng để thêm mô hình và chỉ số đánh giá mới

## Yêu cầu

- Python 3.8+
- Các thư viện Python (xem `requirements.txt`)
- API keys cho AWS
- GPU (tùy chọn, nhưng được khuyến nghị cho Stable Diffusion)

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/yourusername/genai-image-compare.git
cd genai-image-compare
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Cấu hình API keys:
   - Tạo file `.env` trong thư mục gốc
   - Thêm các API keys của bạn:
   ```
   AWS_ACCESS_KEY_ID=your_aws_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret
   AWS_REGION=us-east-1
   ```

## Cách sử dụng

### Chạy đánh giá đầy đủ

```bash
python main.py
```

### Chạy đánh giá nhanh (ít prompts hơn)

```bash
python main.py --quick-test
```

### Chỉ đánh giá một mô hình cụ thể

```bash
python main.py --models SDXL
```

### Chỉ sử dụng một số loại prompts

```bash
python main.py --categories simple complex
```

### Chỉ định thư mục đầu ra tùy chỉnh

```bash
python main.py --output ./custom_output
```

## Cấu trúc dự án

```
genai-image-compare/
├── config.py           # Cấu hình và tham số
├── evaluation.py       # Pipeline đánh giá chính
├── generators.py       # Các module tạo hình ảnh
├── main.py             # Điểm vào chính
├── metrics.py          # Các chỉ số đánh giá
├── utils.py            # Các hàm tiện ích
├── outputs/            # Hình ảnh được tạo ra
├── reference_images/   # Hình ảnh tham chiếu
└── reports/            # Báo cáo và biểu đồ
```

## Các chỉ số đánh giá

- **CLIP Score**: Đo lường sự phù hợp ngữ nghĩa giữa prompt và hình ảnh
- **Thời gian tạo hình ảnh**: Đo thời gian cần thiết để tạo ra hình ảnh
- **Đánh giá con người**: Các tiêu chí như tính chân thực, chất lượng chi tiết, độ tuân thủ prompt
- **FID Score**: Đo lường sự khác biệt giữa phân phối của hình ảnh thật và hình ảnh được tạo ra (tùy chọn)

## Mở rộng

### Thêm mô hình mới

1. Tạo một class mới trong `generators.py` kế thừa từ interface chung
2. Thêm mô hình vào factory function `get_generator()`
3. Cập nhật danh sách mô hình trong `main.py`

### Thêm chỉ số đánh giá mới

1. Thêm implementation của chỉ số trong `metrics.py`
2. Cập nhật `ModelEvaluationPipeline` trong `evaluation.py` để sử dụng chỉ số mới

## Đóng góp

Đóng góp và báo cáo lỗi luôn được chào đón! Vui lòng tạo issue hoặc pull request trên GitHub.

## Giấy phép

MIT License

## DISCLAIMER
- For educational/reference purposes only

- Not production-ready, use at your own risk

- No warranty provided - test thoroughly before use

- Author not liable for any damages or issues
