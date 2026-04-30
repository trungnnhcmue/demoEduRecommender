from flask import Flask
from webapp.routes import init_routes
from webapp.auth import init_auth
from webapp.admin import init_admin

def create_app():
    app = Flask(__name__)
    app.secret_key = "super_secret_key"  # session key

    # Đăng ký routes
    init_auth(app)
    init_routes(app)
    init_admin(app)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
