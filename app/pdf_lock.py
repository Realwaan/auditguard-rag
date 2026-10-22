"""Serialize PDFium access across uploads and on-demand image requests."""
from functools import wraps
from threading import RLock

PDF_LOCK = RLock()


def serialized_pdf(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with PDF_LOCK:
            return function(*args, **kwargs)
    return wrapped
