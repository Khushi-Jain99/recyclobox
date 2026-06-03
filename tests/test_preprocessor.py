import os
import sys
import unittest
import numpy as np

# Add src to system path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing.preprocessor import preprocess_frame, apply_clahe

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a dummy BGR frame (e.g. 100x100x3)
        self.dummy_frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
    def test_preprocess_frame_shape_and_range(self):
        target_size = (224, 224)
        processed = preprocess_frame(self.dummy_frame, target_size=target_size)
        
        # Verify shape
        self.assertEqual(processed.shape, (224, 224, 3))
        
        # Verify values are normalized to [0, 1]
        self.assertTrue(np.all(processed >= 0.0))
        self.assertTrue(np.all(processed <= 1.0))
        self.assertEqual(processed.dtype, np.float32)
        
    def test_preprocess_empty_frame(self):
        empty_frame = np.array([])
        processed = preprocess_frame(empty_frame)
        self.assertIsNone(processed)
        
        none_frame = None
        processed = preprocess_frame(none_frame)
        self.assertIsNone(processed)
        
    def test_apply_clahe_color(self):
        enhanced = apply_clahe(self.dummy_frame)
        self.assertEqual(enhanced.shape, self.dummy_frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

if __name__ == "__main__":
    unittest.main()
