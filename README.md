# App nhận diện số viết tay ANN

## 1. Cài thư viện

```bash
pip install -r requirements.txt
```

## 2. Train model

```bash
python train_model.py
```

Sau khi train xong sẽ có:
- train.keras
- train.h5

## 3. Chạy app

```bash
streamlit run app.py
```

## Lưu ý

Nếu anh đặt model tên là `train` trước đó thì nên đổi thành:
- `train.keras`, hoặc
- `train.h5`

Trong code app đã tự load `train.keras`, nếu không có thì load `train.h5`.

Logic xử lý ảnh:
- chuyển grayscale
- đưa về 28x28
- căn giữa số
- reshape thành 1x784
- normalize `/255.0`
- model ANN dự đoán