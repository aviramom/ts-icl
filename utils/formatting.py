_TSE_OFFICIAL_FORMAT_HINT = (
    "Please answer the question and provide the correct option letter, "
    "e.g., A), B), C), D), and option content at the end of your answer. "
    "All information need to answer the question is given. "
    "If you are unsure, please provide your best guess."
)


def icl_classification_format(desc: str, examples: str, target_ts: str, options: list, prompt_format=None) -> str:
    """Build the full ICL prompt for a single test sample.

    Args:
        desc:          Description / question text (empty string = no description).
        examples:      Pre-formatted few-shot block (the "--- EXAMPLES ---" section).
        target_ts:     The query TS line(s) only — e.g. "New Time Series: <ts><ts/>".
                       The "--- TARGET ---" header is added by this function.
        options:       Valid class labels (used in the "Return ONLY..." instruction).
        prompt_format: One of "no_support", "desc_first", "no_desc", "desc_last", or None.
                       None preserves legacy behavior (desc="" → no_desc, desc non-empty → desc_first).

    Prompt structures:
        no_support:  [desc] → TARGET → query
        desc_first:  [desc] → EXAMPLES → TARGET → query
        no_desc:     EXAMPLES → TARGET → query
        desc_last:   EXAMPLES → TARGET → [desc] → query
        None (legacy): same as desc_first when desc non-empty, same as no_desc when desc empty
    """
    if prompt_format == "tse_official":
        # Official TSE MCQ format — no "Time Series Classification." prefix, no "Return ONLY" suffix.
        if examples.strip():
            return (
                f"Here are some labeled examples:\n"
                f"{examples.strip()}\n\n"
                f"Now answer the following:\n\n"
                f"{desc.strip()}\n\n"
                f"{target_ts.strip()}\n\n"
                f"{_TSE_OFFICIAL_FORMAT_HINT}\n"
            )
        else:
            return (
                f"{desc.strip()}\n\n"
                f"{target_ts.strip()}\n\n"
                f"{_TSE_OFFICIAL_FORMAT_HINT}\n"
            )

    options_str = "[" + ", ".join(str(o) for o in options) + "]"
    suffix = f"Return ONLY the label as one of: {options_str} without any explanation\n"

    if prompt_format == "no_support":
        # Zero-shot: question text + target only, no examples section
        body = (
            f"{desc.strip()}\n\n"
            f"--- TARGET ---\n"
            f"{target_ts.strip()}\n"
            f"{suffix}"
        )
    elif prompt_format == "desc_last":
        # Examples → (TARGET header + description + TS query)
        body = (
            f"\n{examples.strip()}\n\n"
            f"--- TARGET ---\n"
            f"{desc.strip()}\n\n"
            f"{target_ts.strip()}\n"
            f"{suffix}"
        )
    elif prompt_format == "desc_first":
        # Description → examples → target
        body = (
            f"{desc.strip()}\n\n"
            f"{examples.strip()}\n\n"
            f"--- TARGET ---\n"
            f"{target_ts.strip()}\n"
            f"{suffix}"
        )
    elif prompt_format == "no_desc":
        # Examples → target, no description
        body = (
            f"\n{examples.strip()}\n\n"
            f"--- TARGET ---\n"
            f"{target_ts.strip()}\n"
            f"{suffix}"
        )
    else:
        # Legacy (prompt_format=None): desc="" → no_desc layout; desc non-empty → desc_first layout.
        # Preserves original whitespace behavior exactly.
        body = (
            f"{desc.strip()}\n\n"
            f"{examples.strip()}\n\n"
            f"--- TARGET ---\n"
            f"{target_ts.strip()}\n"
            f"{suffix}"
        )

    return "Time Series Classification.\n" + body
