import re


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences while preserving sentence boundaries.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[str]:

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        # If adding this sentence exceeds the target size,
        # finalize the current chunk.
        if (
            current_chunk
            and current_length + sentence_length > chunk_size
        ):
            chunk = " ".join(current_chunk)
            chunks.append(chunk)

            # Keep sentences from the end of the previous
            # chunk for overlap.
            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(current_chunk):
                if overlap_length + len(previous_sentence) > chunk_overlap:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_length += len(previous_sentence)

            current_chunk = overlap_sentences
            current_length = sum(
                len(sentence)
                for sentence in current_chunk
            )

        current_chunk.append(sentence)
        current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks