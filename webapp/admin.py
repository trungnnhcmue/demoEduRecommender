from flask import render_template, request, redirect, url_for, session
import pandas as pd
import os
import math

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


# ---------- Pagination helper ----------
def paginate_df(df, page, per_page):
    page = max(int(page), 1)
    per_page = max(int(per_page), 1)

    total = len(df)
    total_pages = max(math.ceil(total / per_page), 1)

    if page > total_pages:
        page = total_pages

    start = (page - 1) * per_page
    end = start + per_page

    return df.iloc[start:end], page, per_page, total, total_pages


def init_admin(app):

    # ---------- Dashboard ----------
    @app.route("/admin")
    def admin_dashboard():
        if session.get("role") != "admin":
            return redirect(url_for("login"))
        return render_template("admin_dashboard.html")

    # ---------- Students ----------
    @app.route("/admin/students")
    def admin_students():
        if session.get("role") != "admin":
            return redirect(url_for("login"))

        students = pd.read_csv(
            os.path.join(DATA_DIR, "students.csv"),
            dtype={"student_id": "string"}
        )
        students["student_id"] = students["student_id"].astype(str).str.strip()
        students = students.sort_values("student_id")

        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)

        page_df, page, per_page, total, total_pages = paginate_df(
            students, page, per_page
        )

        return render_template(
            "admin_students.html",
            students=page_df.to_dict(orient="records"),
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages
        )

    @app.route("/admin/add_student", methods=["POST"])
    def add_student():
        if session.get("role") != "admin":
            return redirect(url_for("login"))

        students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
        new_id = students["student_id"].astype(int).max() + 1

        new_student = {
            "student_id": new_id,
            "name": request.form["name"],
            "major": request.form["major"],
            "year": int(request.form["year"]),
        }

        students = pd.concat([students, pd.DataFrame([new_student])], ignore_index=True)
        students.to_csv(os.path.join(DATA_DIR, "students.csv"), index=False)

        return redirect(url_for("admin_students"))

    @app.route("/admin/delete_student/<int:sid>")
    def delete_student(sid):
        if session.get("role") != "admin":
            return redirect(url_for("login"))

        students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
        students = students[students["student_id"] != sid]
        students.to_csv(os.path.join(DATA_DIR, "students.csv"), index=False)

        return redirect(url_for("admin_students"))

    # ---------- Courses ----------
    @app.route("/admin/courses")
    def admin_courses():
        if session.get("role") != "admin":
            return redirect(url_for("login"))

        courses = pd.read_csv(
            os.path.join(DATA_DIR, "courses.csv"),
            dtype={"course_id": "string"}
        )
        courses["course_id"] = courses["course_id"].astype(str).str.strip()
        courses = courses.sort_values("course_id")

        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)

        page_df, page, per_page, total, total_pages = paginate_df(
            courses, page, per_page
        )

        return render_template(
            "admin_courses.html",
            courses=page_df.to_dict(orient="records"),
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages
        )

    @app.route("/admin/add_course", methods=["POST"])
    def add_course():
        if session.get("role") != "admin":
            return redirect(url_for("login"))

        courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
        new_id = courses["course_id"].astype(int).max() + 1

        new_course = {
            "course_id": new_id,
            "course_name": request.form["course_name"],
            "major": request.form["major"],
            "credits": int(request.form["credits"]),
            "prerequisite": None,
        }

        courses = pd.concat([courses, pd.DataFrame([new_course])], ignore_index=True)
        courses.to_csv(os.path.join(DATA_DIR, "courses.csv"), index=False)

        return redirect(url_for("admin_courses"))

    @app.route("/admin/delete_course/<int:cid>")
    def delete_course(cid):
        if session.get("role") != "admin":
            return redirect(url_for("login"))

        courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
        courses = courses[courses["course_id"] != cid]
        courses.to_csv(os.path.join(DATA_DIR, "courses.csv"), index=False)

        return redirect(url_for("admin_courses"))
