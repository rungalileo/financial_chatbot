import json
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class FunctionCallingAccuracyMetric(BaseMetric):
    """
    A custom metric to evaluate function calling accuracy for LLM agents.
    
    This metric evaluates:
    1. Whether the correct function was called
    2. Whether the function parameters were correct
    3. Whether a function was called when it should be
    4. Whether a function was incorrectly called when it shouldn't be
    """
    
    def __init__(
        self,
        function_schema: Dict[str, Any],
        threshold: float = 0.8,
        name: str = "Function Calling Accuracy"
    ):
        """
        Initialize the metric.
        
        Args:
            function_schema: A dictionary containing the expected function schemas
            threshold: The minimum score to pass the test
            name: The name of the metric
        """
        super().__init__(threshold, name)
        self.function_schema = function_schema
        self.last_score = 0
        self.details = {}
    
    def _extract_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract function calls from LLM response text."""
        # This is a simplified implementation
        # In practice, you would have a more robust parser based on your system's format
        try:
            # Try to find function call pattern in the text
            if "function_call" in text:
                start_idx = text.find("{")
                end_idx = text.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    function_json = text[start_idx:end_idx]
                    return [json.loads(function_json)]
            return []
        except Exception:
            return []
    
    def _parameter_similarity(
        self, expected_params: Dict[str, Any], actual_params: Dict[str, Any]
    ) -> float:
        """Calculate similarity score between expected and actual parameters."""
        if not expected_params and not actual_params:
            return 1.0
            
        if not expected_params or not actual_params:
            return 0.0
            
        # Count matching keys
        expected_keys = set(expected_params.keys())
        actual_keys = set(actual_params.keys())
        matching_keys = expected_keys.intersection(actual_keys)
        
        if not matching_keys:
            return 0.0
            
        # Simple parameter equality check
        # In a production system, you might want semantic similarity for string values
        correct_params = 0
        for key in matching_keys:
            if expected_params[key] == actual_params[key]:
                correct_params += 1
                
        return correct_params / len(expected_keys)
    
    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure the function calling accuracy.
        
        Args:
            test_case: A test case containing expected and actual responses
            
        Returns:
            The score between 0 and 1
        """
        expected_response = test_case.expected_output
        actual_response = test_case.actual_output
        
        # Extract context and input that might determine if functions should be called
        context = test_case.context
        input_text = test_case.input
        
        # Expected function calls (from test case metadata or expected output)
        expected_function_calls = getattr(test_case, "expected_function_calls", [])
        if not expected_function_calls and hasattr(test_case, "metadata"):
            expected_function_calls = test_case.metadata.get("expected_function_calls", [])
        
        # Extract actual function calls from the response
        actual_function_calls = self._extract_function_calls(actual_response)
        
        # Calculate scores
        if not expected_function_calls and not actual_function_calls:
            # No functions expected and none called - perfect!
            self.last_score = 1.0
            self.details = {"reason": "No functions expected or called"}
            return 1.0
            
        if not expected_function_calls and actual_function_calls:
            # Functions called when none expected
            self.last_score = 0.0
            self.details = {
                "reason": "Functions called when none expected",
                "actual_calls": actual_function_calls
            }
            return 0.0
            
        if expected_function_calls and not actual_function_calls:
            # Functions expected but none called
            self.last_score = 0.0
            self.details = {
                "reason": "Functions expected but none called",
                "expected_calls": expected_function_calls
            }
            return 0.0
        
        # Calculate function calling accuracy
        call_scores = []
        for expected_call in expected_function_calls:
            best_match_score = 0.0
            best_match_details = {}
            
            expected_name = expected_call.get("name", "")
            expected_params = expected_call.get("parameters", {})
            
            for actual_call in actual_function_calls:
                actual_name = actual_call.get("name", "")
                actual_params = actual_call.get("parameters", {})
                
                # Check function name match
                if expected_name == actual_name:
                    # Function name matches, now check parameters
                    param_score = self._parameter_similarity(expected_params, actual_params)
                    call_score = 0.5 + (0.5 * param_score)  # 50% for correct function, 50% for params
                    
                    if call_score > best_match_score:
                        best_match_score = call_score
                        best_match_details = {
                            "expected_call": expected_call,
                            "actual_call": actual_call,
                            "param_score": param_score
                        }
            
            call_scores.append(best_match_score)
            
        if not call_scores:
            self.last_score = 0.0
            self.details = {"reason": "No matching function calls found"}
            return 0.0
            
        # Overall score is the average of all call scores
        final_score = sum(call_scores) / len(expected_function_calls)
        self.last_score = final_score
        self.details = {
            "call_scores": call_scores,
            "expected_calls": len(expected_function_calls),
            "actual_calls": len(actual_function_calls)
        }
        
        return final_score
    
    def get_score(self) -> float:
        """Return the last computed score."""
        return self.last_score
    
    def get_details(self) -> Dict[str, Any]:
        """Return detailed information about the evaluation."""
        return self.details

# Example usage
def evaluate_function_calling(agent_response, expected_calls, function_schemas):
    """Utility function to evaluate function calling in an agent response."""
    # Create a test case
    test_case = LLMTestCase(
        input="What's the weather in New York?",
        actual_output=agent_response,
        expected_output="",  # Not used for this metric
        context="User is asking about weather",
        metadata={"expected_function_calls": expected_calls}
    )
    
    # Initialize and run the metric
    metric = FunctionCallingAccuracyMetric(function_schema=function_schemas)
    score = metric.measure(test_case)
    
    return {
        "score": score,
        "details": metric.get_details(),
        "passed": score >= metric.threshold
    }

# Example demonstration
if __name__ == "__main__":
    # Define function schemas
    weather_function_schema = {
        "weather": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    # Expected function call
    expected_calls = [
        {
            "name": "get_weather",
            "parameters": {
                "location": "New York, NY",
                "unit": "fahrenheit"
            }
        }
    ]
    
    # Example agent response with function call
    agent_response = """
    I'll check the weather for you.
    
    function_call: {
        "name": "get_weather",
        "parameters": {
            "location": "New York, NY",
            "unit": "fahrenheit"
        }
    }
    """
    
    # Evaluate function calling
    result = evaluate_function_calling(
        agent_response=agent_response,
        expected_calls=expected_calls,
        function_schemas=weather_function_schema
    )
    
    print(f"Function Calling Accuracy: {result['score']:.2f}")
    print(f"Passed: {result['passed']}")
    print(f"Details: {json.dumps(result['details'], indent=2)}")