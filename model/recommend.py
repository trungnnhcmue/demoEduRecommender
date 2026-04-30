import os
import torch
import pandas as pd

from model.utils import load_data, build_graph
from model.ftpcomplex import FTPComplex

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "model", "saved", "ftpcomplex.pt")


def _norm_id(x):
    """Chuẩn hoá id thành string đã strip để so khớp nhất quán."""
    return str(x).strip()


def recommend_courses(student_id, top_k=5, emb_dim=64, num_layers=2):
    # ===== 1) Chuẩn hoá student_id =====
    sid = _norm_id(student_id)

    # ===== 2) Load data (đọc bằng đường dẫn tuyệt đối) =====
    students, courses, enrollments = load_data(
        os.path.join(DATA_DIR, "students.csv"),
        os.path.join(DATA_DIR, "courses.csv"),
        os.path.join(DATA_DIR, "enrollments.csv"),
    )

    # ===== 3) Chuẩn hoá kiểu cột để tránh mismatch =====
    # student_id: luôn string
    if "student_id" in students.columns:
        students["student_id"] = students["student_id"].astype(str).str.strip()
    if "student_id" in enrollments.columns:
        enrollments["student_id"] = enrollments["student_id"].astype(str).str.strip()

    # course_id: giữ nguyên kiểu trong courses/enrollments nhưng chuẩn hoá
    if "course_id" in courses.columns:
        courses["course_id"] = courses["course_id"].astype(str).str.strip()
    if "course_id" in enrollments.columns:
        enrollments["course_id"] = enrollments["course_id"].astype(str).str.strip()

    # ===== 4) Build graph =====
    data = build_graph(students, courses, enrollments)

    # ===== 5) Load model =====
    model = FTPComplex(
        num_nodes=data.num_nodes,
        embedding_dim=emb_dim,
        num_layers=num_layers
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

    # map_location giúp chạy được cả CPU
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # ===== 6) Get embeddings =====
    with torch.no_grad():
        emb = model.get_embedding(data.edge_index)

    # ===== 7) Lấy embedding student =====
    # student2idx có thể chứa key string hoặc int
    s_idx = None
    if hasattr(data, "student2idx"):
        if sid in data.student2idx:
            s_idx = data.student2idx[sid]
        else:
            # thử key int nếu có thể
            try:
                sid_int = int(sid)
                if sid_int in data.student2idx:
                    s_idx = data.student2idx[sid_int]
            except ValueError:
                pass

    if s_idx is None:
        # debug rõ: show vài key đầu để bạn kiểm tra
        sample_keys = list(getattr(data, "student2idx", {}).keys())[:10]
        raise KeyError(
            f"student_id='{sid}' không tồn tại trong data.student2idx. "
            f"Ví dụ key hiện có: {sample_keys}"
        )

    student_emb = emb[s_idx]

    # ===== 8) Bỏ qua những môn sinh viên đã học =====
    taken_courses = set(
        enrollments.loc[enrollments["student_id"] == sid, "course_id"].astype(str).str.strip().values
    )

    # ===== 9) Tính điểm cho từng course chưa học =====
    scores = []
    for cid, idx in data.course2idx.items():
        cid_norm = _norm_id(cid)
        if cid_norm in taken_courses:
            continue

        # node index của course thường = idx + num_students
        c_idx = idx + data.num_students
        score = torch.dot(student_emb, emb[c_idx]).item()
        scores.append((cid_norm, score))

    # ===== 10) Top-K =====
    top_courses = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    # ===== 11) Trả về course_name =====
    results = []
    for cid, _ in top_courses:
        row = courses.loc[courses["course_id"] == cid]
        if row.empty:
            results.append(cid)  # fallback: trả course_id
        else:
            results.append(row["course_name"].values[0])

    return results


if __name__ == "__main__":
    # demo
    recs = recommend_courses(student_id="24000001", top_k=5)
    print("Recommend demo:", recs)
