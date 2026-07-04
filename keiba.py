#!/usr/bin/env python
"""どこからでも・インストール無しで動く起動スクリプト.

使い方（keiba-ai フォルダ内で）::

    python keiba.py calibrate --date 2026-06-26 --track 大井 --race 11
    python keiba.py predict   --date 2026-06-26 --track 大井 --race 11

このファイルが src/ を自動で import パスに追加するので、
`pip install` や PYTHONPATH の設定は不要。
"""

import sys
from pathlib import Path

# このファイルと同じ場所の src/ をパスに通す
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keiba_ai.cli import main

if __name__ == "__main__":
    main()
