from flask import Flask
from flask_cors import CORS
from routes.ml_routes import ml_bp
from routes.etl_routes import etl_bp
from routes.data_routes import data_bp

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.register_blueprint(ml_bp, url_prefix="/api/ml")
    app.register_blueprint(etl_bp, url_prefix="/api/etl")
    app.register_blueprint(data_bp, url_prefix="/api/data")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)