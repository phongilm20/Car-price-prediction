# 🚗 Dự Án Hồi Quy: Dự Đoán Giá Xe Ô tô Cũ (Car Price Prediction)

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-v1.4.1-orange.svg)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 1. Tổng Quan Dự Án (Project Overview)

Đây là dự án Machine Learning cá nhân, tập trung vào xây dựng và triển khai một mô hình hồi quy để dự đoán giá bán của xe ô tô cũ. Dự án nhấn mạnh vào việc áp dụng các tiêu chuẩn **AI Engineering** thông qua việc sử dụng **Scikit-learn Pipeline** để đảm bảo tính nhất quán (Consistency) và khả năng triển khai (Deployability) của mô hình.

* **Mục tiêu:** Dự đoán giá bán dựa trên các yếu tố như năm sản xuất, số km đã đi, thương hiệu, loại nhiên liệu, v.v.
* **Mô hình Chính:** Random Forest Regressor.
* **Điểm mạnh Kỹ thuật:** **ML Pipeline** và **Xử lý Dữ liệu Ngoại lai** (`handle_unknown` trong One-Hot Encoding).

## 🛠️ 2. Ngăn Xếp Công Nghệ (Technology Stack)

| Lĩnh vực | Công cụ/Thư viện | Mục đích Kỹ thuật |
| :--- | :--- | :--- |
| **Data Science Core** | `Pandas`, `NumPy` | Thao tác, làm sạch và xử lý dữ liệu số. |
| **ML Engineering** | `Scikit-learn` | Xây dựng **`Pipeline`**, `ColumnTransformer` (Preprocessor), và mô hình. |
| **Triển khai** | `Joblib` / `Pickle` | **Model Serialization** (Lưu trữ toàn bộ quy trình). |
| **EDA/Visual** | `Matplotlib`, `Seaborn` | Phân tích mối quan hệ và phân phối dữ liệu. |

## 💡 3. Phương Pháp Luận Học Thuật (Academic Methodology)

### 3.1. Kỹ Thuật Tính Năng (Feature Engineering)

Toàn bộ quá trình tiền xử lý được gói gọn trong một `ColumnTransformer` để áp dụng các biến đổi khác nhau cho các loại cột:

* **Dữ liệu Phân loại (Categorical):** Sử dụng `OneHotEncoder` với tham số quan trọng `handle_unknown='ignore'`. **Điều này giải quyết triệt để lỗi `ValueError` về Feature names trong dữ liệu mới.**
* **Dữ liệu Số (Numerical):** Sử dụng `StandardScaler` để chuẩn hóa các biến số (`year`, `km_driven`), tránh tình trạng mô hình ưu tiên các biến có thang đo lớn hơn.

### 3.2. Xây dựng ML Pipeline

**Sử dụng `sklearn.pipeline.Pipeline`**

```python
# Cấu trúc kỹ thuật chính:
model_pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(...)), # Bước tiền xử lý nhất quán
    ('regressor', RandomForestRegressor())    # Mô hình hồi quy
])
