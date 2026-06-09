import unittest
import numpy as np

# Import your custom modules from your package
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler

# Mock classes to simulate a model for pipeline testing
class MockEstimator:
    """A minimal fake estimator to test pipeline routing without needing the full Tree model."""
    def __init__(self):
        self.is_fitted_ = False
    def partial_fit(self, X_chunk, y_chunk, **kwargs):
        self.is_fitted_ = True
        return self
    def predict(self, X_chunk):
        return np.zeros(X_chunk.shape[0], dtype=int)


class TestStreamingPipelineAndPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """
        The setUp method runs automatically BEFORE every single test function below.
        We use it to initialize a fresh pipeline instance and baseline variables.
        """
        # Ensure your custom preprocessing.py has a StandardScaler class implemented!
        self.scaler = StandardScaler()
        self.model = MockEstimator()
        
        # Build a standard pipeline sequence from scratch
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('model', self.model)
        ])
        
        # Create a standard, small healthy data chunk (3 samples, 2 features)
        self.X_healthy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.y_healthy = np.array([0, 1, 0])

    # ---------------------------------------------------------
    # TEST 1: Check if StandardScaler handles an empty chunk (Your Request)
    # ---------------------------------------------------------
    def test_scaler_handles_empty_chunk_safely(self):
        """Verifies that an empty array passed to the scaler does not raise exceptions."""
        empty_chunk = np.empty((0, 2))  # 0 rows, 2 columns
        
        try:
            # Try to run partial_fit on empty data
            self.scaler.partial_fit(empty_chunk)
            transformed = self.scaler.transform(empty_chunk)
        except Exception as e:
            self.fail(f"StandardScaler crashed on an empty chunk with error: {e}")
            
        # Assert that the output shape remains logically correct (0 rows, 2 columns)
        self.assertEqual(transformed.shape, (0, 2))

    # ---------------------------------------------------------
    # TEST 2: Check for TypeError with bad input (Your Request)
    # ---------------------------------------------------------
    def test_pipeline_raises_type_error_for_text_input(self):
        """Verifies that the Pipeline blocks raw strings/text and raises a TypeError."""
        bad_text_X = "This is a raw text string, not a NumPy matrix!"
        bad_text_y = np.array([0, 1, 0])
        
        # self.assertRaises checks if the code inside the block successfully triggers a TypeError
        with self.assertRaises(TypeError) as context:
            self.pipeline.partial_fit(bad_text_X, bad_text_y)
            
        # Verify that your custom error message is informative (HD standard requirement!)
        self.assertIn("Pipeline expects feature inputs as a numpy.ndarray", str(context.exception))

    # ---------------------------------------------------------
    # TEST 3: Edge Case - Shape Mismatch Check
    # ---------------------------------------------------------
    def test_pipeline_raises_value_error_on_row_count_mismatch(self):
        """Verifies that the pipeline flags a data error if X and y have different row counts."""
        # X has 3 rows, but y only has 2 rows
        mismatched_y = np.array([0, 1])
        
        with self.assertRaises(ValueError) as context:
            self.pipeline.partial_fit(self.X_healthy, mismatched_y)
            
        self.assertIn("Row count mismatch", str(context.exception))

    # ---------------------------------------------------------
    # TEST 4: Zero Variance Check (Robust Numerical Stability)
    # ---------------------------------------------------------
    def test_scaler_numerical_stability_with_zero_variance(self):
        """Ensures that identical column data doesn't lead to a division-by-zero NaN explosion."""
        # Feature column 2 contains completely identical values (variance is exactly 0)
        zero_var_X = np.array([[2.0, 5.0], [4.0, 5.0], [6.0, 5.0]])
        
        self.scaler.partial_fit(zero_var_X)
        transformed = self.scaler.transform(zero_var_X)
        
        # Assert that your scaler handles division safely and didn't output infinite numbers or NaNs
        self.assertFalse(np.isnan(transformed).any(), "Scaler generated NaN values during zero-variance division!")
        self.assertFalse(np.isinf(transformed).any(), "Scaler generated Infinite values during zero-variance division!")


# This block allows you to run the script directly from your IDE or file explorer
if __name__ == '__main__':
    unittest.main()
