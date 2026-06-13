def icl_classification_format(desc: str, examples: str, target: str, options: list) -> str:
    """Build the full ICL prompt for a single test sample."""
    instruction = f"""{desc.strip()}

{examples.strip()}

{target.strip()}
Return ONLY the label as one of: {options} without any explanation
"""
    return "Time Series Classification.\n" + instruction
