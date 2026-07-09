"""Regression: metal_decode_jpeg must not kill the process on malformed input.

Bug (<= 2.31.0): src/metal/metal_decode.mm used libjpeg's DEFAULT error manager, whose
error_exit calls exit() — a malformed JPEG terminated the entire Python interpreter with
no traceback (verified: subprocess exit code 1, empty stdout). Fixed with the standard
setjmp/longjmp handler; malformed input now raises RuntimeError.
"""

import subprocess
import sys

import pytest

import turboloader

pytestmark = pytest.mark.skipif(
    not (hasattr(turboloader, "metal_available") and turboloader.metal_available()),
    reason="Metal not available on this platform",
)


def test_malformed_jpeg_raises_instead_of_killing_process():
    code = (
        "import turboloader as t\n"
        "payloads = [b'\\xff\\xd8\\xff\\xe0' + b'\\x00'*100, b'garbage',\n"
        "            b'\\xff\\xd8' + b'\\xff'*5000]\n"
        "for pl in payloads:\n"
        "    try:\n"
        "        t.metal_decode_jpeg(pl)\n"
        "    except Exception:\n"
        "        pass\n"
        "print('SURVIVED')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (
        f"interpreter died (exit {r.returncode}) — libjpeg error_exit called exit(); "
        f"stderr: {r.stderr[:300]}"
    )
    assert "SURVIVED" in r.stdout


def test_valid_jpeg_still_decodes():
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    import io

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    out = np.asarray(turboloader.metal_decode_jpeg(buf.getvalue()))
    assert out.shape == (64, 64, 3)
