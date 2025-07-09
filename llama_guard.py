def check_text_safety(text):
    blocked_keywords = ["kill", "bomb", "rape", "attack", "shoot"]
    for word in blocked_keywords:
        if word in text.lower():
            return "unsafe"
    return "safe"
