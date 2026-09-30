"""Flask app factory."""

from __future__ import annotations

from flask import Flask

from app.config import config
from app.logging import setup_logging

setup_logging()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        DEBUG=config.DEBUG,
        JSON_SORT_KEYS=False,
    )

    from app.chat.routes import bp as chat_bp
    from app.retrieval.routes import bp as retrieval_bp
    from app.questions.routes import bp as questions_bp

    app.register_blueprint(retrieval_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(questions_bp)

    return app