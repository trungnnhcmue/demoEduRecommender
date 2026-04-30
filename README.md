# Hệ thống gợi ý khóa học sử dụng Temporal Knowledge Graph

## Giới thiệu

Dự án này xây dựng một **hệ thống gợi ý khóa học cho sinh viên** dựa trên **Đồ thị Tri thức theo thời gian (Temporal Knowledge Graph - TKG)** và các mô hình nhúng tiên tiến.

Mục tiêu là cải thiện khả năng **dự đoán tương tác theo thời gian**, từ đó đưa ra gợi ý khóa học chính xác hơn.

---

## Tính năng chính

* Mô hình hóa dữ liệu bằng Temporal Knowledge Graph
* Áp dụng các mô hình nhúng được thực nghiệm
* Demo hệ thống gợi ý khóa học

---

## Phương pháp

### Biểu diễn dữ liệu

Dữ liệu được biểu diễn dưới dạng bộ bốn:

```id="tkg"
(thực thể nguồn, quan hệ, thực thể đích, thời gian)
```

## Cấu trúc dự án

```id="structure"
├── data/               # Dữ liệu
├── models/             # Cài đặt mô hình
├── training/           # Huấn luyện
├── evaluation/         # Đánh giá
├── demo/               # Ứng dụng demo
├── utils/              # Công cụ hỗ trợ
└── README.md
```

---

## Cài đặt

```bash id="install"
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

---

## Cách sử dụng

### Huấn luyện mô hình

```bash id="train"
python train.py --model TBicomR
```

### Đánh giá

```bash id="eval"
python evaluate.py
```

### Chạy demo

```bash id="demo"
python app.py
```

---

## Dữ liệu

* Dữ liệu đăng ký học phần được biểu diễn dưới dạng đồ thị tri thức đã được ẩn danh
* Dataset riêng phục vụ bài toán gợi ý khóa học

---

## Demo

Link demo:
https://nntrung.pythonanywhere.com/

---

## Cơ sở nghiên cứu

Dự án dựa trên đề tài khoa học:

> *“Xây dựng thuật toán nhúng cải tiến cho dự đoán tương tác trong đồ thị tri thức theo thời gian và ứng dụng vào gợi ý khóa học cho sinh viên”* 

---

## Tác giả

* Nguyễn Ngọc Trung
* Nhóm nghiên cứu – ĐH Sư phạm TP.HCM & ĐH KHTN

---

## License

MIT License

---
