# Understanding the `positions` Parameter in RAGFlow

In RAGFlow, the `positions` parameter serves a critical role: it **records and restores the physical location of each chunk within the original document** — particularly for PDFs. Each position entry is a list of five integers in the format `[page_number, left, right, top, bottom]`.

This mechanism enables three key capabilities:

1. **Visual Provenance** — When displaying chunk content in the UI, these coordinates are used to highlight the corresponding region on the original document or generate a cropped preview image.
2. **Parsing & Ordering** — During PDF processing, position data helps determine the correct reading order and associate chunks with their surrounding table or image context.
3. **Precise Retrieval Feedback** — Search results carry position metadata, allowing users to jump directly to the exact page and region in the source document.

---

## Data Structure & Validation

At the API layer, `positions` is defined as a nested list `list[list[int]]`. Every sub-list must contain exactly **5 elements**:

| Index | Meaning |
|-------|---------|
| 0 | Page number |
| 1 | Left coordinate |
| 2 | Right coordinate |
| 3 | Top coordinate |
| 4 | Bottom coordinate |

The validation logic enforces this strictly:

```python
positions: list[list[int]] = Field(default_factory=list)

@validator("positions")
def validate_positions(cls, value):
    for sublist in value:
        if len(sublist) != 5:
            raise ValueError("Each sublist in positions must have a length of 5")
    return value
```

> Source: `api/apps/restful_apis/chunk_api.py` (L61-68)

---

## PDF Preview Generation

RAGFlow uses the coordinates stored in `positions` to crop preview images from the original PDF pages:

- The `_crop_pdf_preview` function iterates over each position entry, computes the maximum width across all entries, and extracts the corresponding region from `page_images`.
- To improve readability, the system adds a **context margin** above and below the first and last coordinates, so the preview includes some surrounding content rather than a tight crop.

This means each chunk can show a visual snippet of exactly where it came from in the PDF — a powerful feature for building user trust in RAG-generated answers.

---

## Position Handling During Document Parsing

When documents are parsed using engines like **Docling** or **MinerU**, position information is extracted from inline tags embedded in the text:

- **Position extraction**: A regex matches tags in the format `@@page\tleft\tright\ttop\tbottom##` and converts the captured groups into integer lists.
- **Coordinate transformation**: For multi-page documents or those with layout offsets, local coordinates are transformed into global coordinates to ensure consistency across the entire document.

---

## Database Storage & Retrieval

To support efficient querying, `positions` undergoes a mapping before being persisted:

| Database Field | Purpose |
|----------------|---------|
| `position_int` | Stores the raw coordinate values |
| `page_num_int` | Page number index for filtering |
| `top_int` | Vertical position index for sorting |

When retrieval operations such as `retrieval_by_toc` are executed, the system maps `position_int` back to the `positions` field before returning results to the frontend.

---

## Practical Notes

- **Coordinate system**: Positions are based on the standard PDF coordinate system at 72 DPI. When generating preview images, a `ZOOM` factor (e.g., 3x) is applied to improve clarity.
- **Manual updates**: You can update a chunk's `positions` via the `PATCH` API endpoint:
  ```
  - "positions": (Body parameter), list
    Updated source positions for the chunk.
  ```

---

## Why This Matters

The `positions` parameter is what makes RAGFlow's chunking **traceable**. Unlike many RAG systems that treat chunks as opaque text blobs, RAGFlow preserves the spatial link between each chunk and its source document. This enables:

- **Accurate citation** — Users can verify that the AI's answer actually comes from the referenced document.
- **Visual debugging** — Developers and domain experts can inspect exactly which region was extracted, catching parsing errors early.
- **Richer UX** — The frontend can render highlighted PDF previews, turning a retrieval result into a familiar "find in document" experience.

Understanding this parameter is essential for anyone extending RAGFlow's parsing pipeline or building custom document connectors.

## References

- <https://deepwiki.com/search/positions_180e66da-e091-4548-bd42-029eb9e5804b?mode=fast>