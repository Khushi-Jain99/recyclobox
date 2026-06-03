import os
import sys
import unittest
import tensorflow as tf

# Add src to system path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models.model import build_custom_cnn, build_mobilenet_transfer

class TestModel(unittest.TestCase):
    def test_custom_cnn_output_shape(self):
        input_shape = (224, 224, 3)
        num_classes = 7
        model = build_custom_cnn(input_shape=input_shape, num_classes=num_classes)
        
        # Verify output shape
        self.assertEqual(model.output_shape, (None, num_classes))
        self.assertEqual(model.input_shape, (None,) + input_shape)
        
    def test_mobilenet_transfer_output_shape(self):
        input_shape = (224, 224, 3)
        num_classes = 7
        model = build_mobilenet_transfer(input_shape=input_shape, num_classes=num_classes)
        
        # Verify output shape
        self.assertEqual(model.output_shape, (None, num_classes))
        self.assertEqual(model.input_shape, (None,) + input_shape)

    def test_mobilenet_freezing(self):
        # Base fully frozen
        model = build_mobilenet_transfer(trainable_base_layers=0)
        base_layer = model.layers[0]
        self.assertFalse(base_layer.trainable)
        
        # Base partially trainable
        model_ft = build_mobilenet_transfer(trainable_base_layers=20)
        base_layer_ft = model_ft.layers[0]
        # The base layer object itself is trainable now because we set base_model.trainable = True
        self.assertTrue(base_layer_ft.trainable)

if __name__ == "__main__":
    unittest.main()
