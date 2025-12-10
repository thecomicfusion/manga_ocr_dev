import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import pandas as pd
import numpy as np
from manga_ocr_dev.env import ASSETS_PATH

# Increase range and shift center to favor longer text
lengths = np.arange(1, 51)  # Extended to 50
# Shift center from 10 to 18-20 for more multi-line samples
probs = np.exp(-0.5 * ((lengths - 18) / 8) ** 2)
probs = probs / probs.sum()

df = pd.DataFrame({'len': lengths, 'p': probs})
df.to_csv(ASSETS_PATH / 'len_to_p.csv', index=False)
