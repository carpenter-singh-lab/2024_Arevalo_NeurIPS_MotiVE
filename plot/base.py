import io

import numpy as np


def to_numpy(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    array = data.reshape((int(h), int(w), -1))
    return array
