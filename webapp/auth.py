from flask import render_template, request, redirect, url_for, session
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

def init_auth(app):

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            user_id = request.form.get("user_id", "").strip()
            password = request.form.get("password", "").strip()

            # ===== ADMIN LOGIN =====
            if user_id == "admin" and password == "admin123":
                session.clear()
                session["role"] = "admin"
                return redirect(url_for("admin_dashboard"))

            # ===== STUDENT LOGIN =====            
            students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"), dtype={"student_id": "string"})
            user_row = students[students["student_id"].astype(str) == str(user_id)]

            if password == "123456" and not user_row.empty:
                session.clear()
                session["role"] = "student"
                session["student_id"] = str(user_row.iloc[0]["student_id"]).strip()
                return redirect(url_for("student_dashboard"))

            return render_template(
                "login.html",
                error="User ID hoặc mật khẩu không đúng"
            )

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))
