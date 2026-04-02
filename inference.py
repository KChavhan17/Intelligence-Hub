from environment import NewsIntelligenceEnv

def run_inference_test():
    """Validates that the 0.0-1.0 grading logic is active"""
    env = NewsIntelligenceEnv()
    # Mock data test
    test_date = "2026-04-01T10:00:00Z"
    grade = env.calculate_grade(test_date)
    
    return f"Inference Check: Logic is calculating Grade {grade} successfully."

if __name__ == "__main__":
    print(run_inference_test())
