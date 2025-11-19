import sys
from pathlib import Path

# Add parent directory to Python path so manga_ocr_dev can be imported
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


import pandas as pd
import numpy as np
from manga_ocr_dev.env import ASSETS_PATH

# Create a reasonable distribution (bell curve centered around 10)
lengths = np.arange(1, 31)
# Normal distribution-like probabilities
probs = np.exp(-0.5 * ((lengths - 10) / 5) ** 2)
probs = probs / probs.sum()  # Normalize to sum to 1

df = pd.DataFrame({'len': lengths, 'p': probs})
df.to_csv(ASSETS_PATH / 'len_to_p.csv', index=False)