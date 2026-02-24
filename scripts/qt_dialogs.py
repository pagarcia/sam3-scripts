# scripts/qt_dialogs.py
from __future__ import annotations

import sys
from typing import Optional


def pick_file(title: str, file_filter: str, start_dir: str = "") -> Optional[str]:
    """
    Open a native file picker using PyQt5.
    Returns selected path or None.
    """
    from PyQt5 import QtWidgets  # type: ignore

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv)

    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, title, start_dir, file_filter
    )

    if owns_app:
        try:
            app.quit()
        except Exception:
            pass

    return path or None