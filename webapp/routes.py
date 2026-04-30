from flask import render_template, session, redirect, url_for
import pandas as pd
import os
import random
import math

from model.recommend import recommend_courses
from visualize.visualize_graph import build_graph, visualize_student_graph

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ---------- Helpers: load CSV with stable dtypes ----------
def _csv_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def _load_students() -> pd.DataFrame:
    df = pd.read_csv(_csv_path("students.csv"), dtype={"student_id": "string"})
    if "student_id" not in df.columns:
        raise ValueError("students.csv thiếu cột student_id")
    df["student_id"] = df["student_id"].astype(str).str.strip()
    return df


def _load_courses() -> pd.DataFrame:
    df = pd.read_csv(_csv_path("courses.csv"), dtype={"course_id": "string"})
    if "course_id" not in df.columns:
        raise ValueError("courses.csv thiếu cột course_id")
    df["course_id"] = df["course_id"].astype(str).str.strip()

    if "course_name" not in df.columns:
        for alt in ["name", "title", "course_title"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "course_name"})
                break

    if "course_name" not in df.columns:
        df["course_name"] = df["course_id"].apply(lambda x: f"Course_{x}")

    return df


def _load_enrollments() -> pd.DataFrame:
    df = pd.read_csv(
        _csv_path("enrollments.csv"),
        dtype={"student_id": "string", "course_id": "string", "semester": "string", "grade": "string"},
    )
    # kiểm tra cột tối thiểu
    for col in ["student_id", "course_id"]:
        if col not in df.columns:
            raise ValueError(f"enrollments.csv thiếu cột {col}")

    df["student_id"] = df["student_id"].astype(str).str.strip()
    df["course_id"] = df["course_id"].astype(str).str.strip()

    if "semester" not in df.columns:
        df["semester"] = ""
    if "grade" not in df.columns:
        df["grade"] = ""

    df["semester"] = df["semester"].astype(str).str.strip()
    df["grade"] = df["grade"].astype(str).str.strip()

    return df


# ---------- XAI text ----------
def generate_explanation(course_name, student):
    # Test XAI for future work
    templates = [
        {
            "intro": "Môn học này được đề xuất vì có mối liên hệ mạnh với các môn bạn đã học gần đây.",
            "reasons": [
                "Cấu trúc đồ thị cho thấy môn có nhiều kết nối với các môn nền tảng bạn đã hoàn thành.",
                "FTPComplex ưu tiên các tương tác gần theo thời gian học tập (học kỳ gần nhất).",
                "Embedding của môn nằm gần embedding của sinh viên trong không gian biểu diễn.",
            ],
        },
        {
            "intro": "Gợi ý này xuất phát từ sự tương đồng trong lộ trình học tập của bạn so với các sinh viên khác.",
            "reasons": [
                "Sinh viên có hồ sơ học tập tương tự thường đăng ký môn này ở giai đoạn hiện tại.",
                "Yếu tố thời gian giúp mô hình nhận diện đúng thời điểm phù hợp để học môn này.",
                "Quan hệ sinh viên–môn học được củng cố qua các học kỳ liên tiếp.",
            ],
        },
        {
            "intro": "Môn học này phù hợp với tiến trình học tập hiện tại của bạn theo phân tích của hệ thống.",
            "reasons": [
                "Đồ thị tương tác phản ánh mối quan hệ học phần kế tiếp từ các môn bạn đã hoàn thành.",
                "FTPComplex làm giảm ảnh hưởng của các tương tác quá xa trong quá khứ.",
                "Biểu diễn học được giúp mô hình đánh giá mức độ phù hợp tổng thể.",
            ],
        },
        {
            "intro": "Hệ thống đề xuất môn này nhằm hỗ trợ bạn cân bằng kiến thức trong giai đoạn hiện tại.",
            "reasons": [
                "Môn học bổ sung kiến thức cho các môn bạn đã đạt kết quả tốt.",
                "Yếu tố thời gian giúp nhận diện sự chuyển tiếp hợp lý giữa các học kỳ.",
                "Embedding của môn cho thấy mức độ liên quan cao với hồ sơ học tập của bạn.",
            ],
        },
    ]

    chosen = random.choice(templates)
    return {"course": course_name, "intro": chosen["intro"], "reasons": chosen["reasons"]}



    
# ---------- Routes ----------
def init_routes(app):
    @app.route("/")
    def home():
        if "role" not in session:
            return redirect(url_for("login"))
        if session["role"] == "student":
            return redirect(url_for("student_dashboard"))
        if session["role"] == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("login"))
        
    @app.route("/student")
    def student_dashboard():
        if session.get("role") != "student":
            return redirect(url_for("login"))

        sid = session.get("student_id", None)
        if sid is None:
            session.clear()
            return redirect(url_for("login"))

        sid = str(sid).strip()

        # Load data
        try:
            students = _load_students()
            courses = _load_courses()
            enrollments = _load_enrollments()
        except Exception as e:
            return f"Lỗi đọc dữ liệu CSV: {e}", 500

        # Find student
        matched = students[students["student_id"] == sid]
        if matched.empty:
            session.clear()
            return redirect(url_for("login"))

        student_info = matched.iloc[0].to_dict()

        # Taken courses
        taken = enrollments[enrollments["student_id"] == sid].copy()

        if taken.empty:
            taken_courses_list = []
        else:
            taken_courses = pd.merge(taken, courses, on="course_id", how="left")

            # nếu course_name bị NaN → fallback
            taken_courses["course_name"] = taken_courses["course_name"].fillna(
                taken_courses["course_id"].apply(lambda x: f"Course_{x}")
            )

            # đảm bảo cột tồn tại
            if "semester" not in taken_courses.columns:
                taken_courses["semester"] = ""
            if "grade" not in taken_courses.columns:
                taken_courses["grade"] = ""

            taken_courses_list = (
                taken_courses[["course_name", "semester", "grade"]]
                .fillna("")
                .to_dict(orient="records")
            )
        
        # Recommendations
        try:
            recommendations = recommend_courses(sid, top_k=5)
        except KeyError:
            recommendations = recommend_courses(int(sid), top_k=5)


        return render_template(
            "index.html",
            student=student_info,
            taken_courses=taken_courses_list,
            recommendations=recommendations,
        )

    @app.route("/explain")
    def explain_page():
        if session.get("role") != "student":
            return redirect(url_for("login"))

        sid = session.get("student_id", None)
        if sid is None:
            session.clear()
            return redirect(url_for("login"))
        sid = str(sid).strip()

        # Load student info
        try:
            students = _load_students()
        except Exception as e:
            return f"Lỗi đọc students.csv: {e}", 500

        matched = students[students["student_id"] == sid]
        if matched.empty:
            session.clear()
            return redirect(url_for("login"))

        student_info = matched.iloc[0].to_dict()

        # Recommendations
        recommendations = recommend_courses(sid, top_k=5)

        # Build explained list (ngẫu nhiên theo templates)
        explained_recommendations = []
        for r in recommendations:
            cname = r.get("course_name") if isinstance(r, dict) else str(r)
            explained_recommendations.append(generate_explanation(cname, student_info))

        return render_template(
            "explain.html",
            student=student_info,
            recommendations=recommendations,  
            explained_recommendations=explained_recommendations, 
        )

    @app.route("/model")
    def model_page():
        if "role" not in session:
            return redirect(url_for("login"))
        return render_template("model.html")

    @app.route("/student/graph")
    def student_graph():
        if session.get("role") != "student":
            return redirect(url_for("login"))

        sid = session.get("student_id", None)
        if sid is None:
            session.clear()
            return redirect(url_for("login"))
        sid = str(sid).strip()

        # Build graph
        G = build_graph()

        # Output to static/
        output_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"student_graph_{sid}.html"
        output_path = os.path.join(output_dir, output_file)

        visualize_student_graph(G, student_id=sid, output_path=output_path)

        return render_template("student_graph.html", graph_file=output_file)
