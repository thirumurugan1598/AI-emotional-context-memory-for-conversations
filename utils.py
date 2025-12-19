def truncate_text(text, max_len=200):
    return text if len(text) <= max_len else text[:max_len] + "..."
