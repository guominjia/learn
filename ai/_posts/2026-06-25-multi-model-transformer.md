---
layout: post
title: "Text Embeddings vs Visual Embeddings in Multimodal Transformers"
date: 2026-06-25
tags: [multimodal, transformer, embedding, vision-transformer, llm]
---

In a multimodal Transformer, both text and images eventually become vectors. Because of that, it is easy to say "text is embedded" and "images are embedded" as if they were produced in the same way.

They are not.

A useful mental model is:

- **Text embeddings are more discrete and table-like.**
- **Visual embeddings are more continuous and dynamically computed from the input.**

This distinction explains why a language model can look up a token embedding directly, while a vision-language model must run an image through preprocessing and a visual encoder before the large language model can reason over it.

---

## 1) Text Side: Token ID to Embedding Table Lookup

Text is first converted into tokens by a tokenizer. Each token has an integer ID:

```text
"cat" -> token_id = 5432
```

The model has an embedding matrix, often shaped like:

$$
E \in \mathbb{R}^{V \times d}
$$

where:

- $V$ is the vocabulary size,
- $d$ is the hidden dimension,
- each row $E_i$ is the vector for token ID $i$.

So the first step is basically a lookup:

```text
token_id -> embedding_matrix[token_id]
```

For the same model checkpoint, the same token ID maps to the same initial embedding vector every time. If token ID `5432` means `cat`, then before contextual layers run, its embedding row is fixed.

Of course, the token representation later changes inside the Transformer layers because self-attention mixes it with surrounding context. But the initial mapping from token ID to vector is discrete and table-based.

### Important nuance: fixed initial vector, not fixed final meaning

The same token can mean different things in different sentences:

```text
I sat by the river bank.
I went to the bank to withdraw cash.
```

The initial token embedding for `bank` is the same, but after contextual Transformer layers, its hidden representation becomes different because the surrounding tokens are different.

So text embedding has two stages:

| Stage | Behavior |
|---|---|
| Token embedding lookup | Fixed for the same token ID in the same model |
| Contextual hidden states | Dynamically computed based on surrounding tokens |

---

## 2) Vision Side: Image Pixels to Dynamically Computed Embeddings

Images do not naturally arrive as token IDs from a fixed vocabulary. An image is a grid of pixel values. Even two photos of the same cat can have different lighting, pose, background, resolution, and noise.

Therefore, the visual side usually does not start with:

```text
image_id -> fixed_embedding_table[image_id]
```

Instead, the common flow is online encoding of the current input image:

```text
image -> preprocessing -> visual encoder -> visual feature vectors
```

The produced vectors depend on the actual pixel content of the input image.

### A typical visual pipeline

In many multimodal systems, the image path looks like this:

1. **Preprocess the image**
	- resize,
	- crop or pad,
	- normalize pixel values,
	- possibly split a large image into tiles.

2. **Convert the image into patches or feature maps**
	- A Vision Transformer (ViT) splits the image into patches.
	- A CNN extracts hierarchical feature maps.
	- Some systems use tiled or multi-resolution image processing.

3. **Run a visual encoder forward pass**
	- Each patch, tile, or feature location is transformed by learned neural network parameters.
	- The output is a sequence of visual feature vectors.

4. **Project visual features into the language model space**
	- A projection layer, adapter, resampler, or cross-attention module maps vision features to the dimension expected by the language model.

5. **Combine visual vectors with text tokens**
	- The model may concatenate visual tokens with text tokens.
	- Or the language model may attend to image features through cross-attention.

Conceptually:

```text
image pixels
  -> resize / normalize / patchify / tile
  -> CNN or ViT visual encoder
  -> visual embeddings
  -> projector / adapter
  -> multimodal Transformer
```

Unlike text token embeddings, the visual embeddings are not fixed rows in a vocabulary table. They are computed for each input image.

---

## 3) Why "Unseen Images" Can Still Produce Vectors

A natural question is: if an image was never seen during training, how can the model produce a vector for it?

The answer is **generalization**.

The visual encoder does not memorize a vector for every possible image. Instead, during training, it learns parameters that define a function:

$$
f_{vision}(image) \rightarrow visual\ embeddings
$$

At inference time, a new image goes through the same preprocessing and forward computation. The encoder applies its learned filters, attention layers, and projections to the new pixels and returns a new set of vectors.

These vectors can be viewed as coordinates in the model's learned semantic space.

For example, images of different dogs may map to nearby regions of that space, even if the exact photos were never in the training set. Images of cars, charts, screenshots, or handwritten notes will land in other regions depending on what the visual encoder has learned.

The quality of this mapping depends on:

- the scale and diversity of training data,
- the visual encoder architecture,
- the image resolution and preprocessing strategy,
- the alignment training between vision and language,
- how far the new image is from the training distribution.

So unseen images can be encoded, but the quality is not guaranteed. A model trained mostly on natural images may struggle with medical scans, satellite imagery, industrial diagrams, or unusual visual domains unless it has seen similar data or has strong transfer ability.

---

## 4) The Key Difference: Discrete Lookup vs Continuous Computation

The contrast can be summarized like this:

| Aspect | Text Embedding | Visual Embedding |
|---|---|---|
| Input form | Token ID | Pixel array |
| First operation | Embedding table lookup | Preprocessing + neural network forward pass |
| Initial vector source | A fixed row in the embedding matrix | Computed from image content |
| Same input behavior | Same token ID gives same initial vector | Same image with same preprocessing gives same computed vector |
| New input handling | New text must be tokenized into known token IDs | New image is encoded by the learned vision function |
| Nature | Discrete vocabulary-based mapping | Continuous content-based mapping |

Text is discrete because the tokenizer maps strings into a finite vocabulary. Vision is continuous because images live in a high-dimensional pixel space, and small pixel changes can produce smoothly changing features.

---

## 5) How Multimodal Transformers Use Visual Embeddings

Once visual embeddings are produced, the multimodal model must connect them with language tokens. Common designs include:

### Concatenation as visual tokens

Some models project image features into the same hidden dimension as text token embeddings and then concatenate them:

```text
[visual_token_1, visual_token_2, ..., text_token_1, text_token_2, ...]
```

The language model then processes a combined sequence. From the Transformer's perspective, image patches become token-like vectors.

### Cross-attention

Other models keep vision features separate and let language hidden states attend to them through cross-attention:

```text
text queries attend to visual keys and values
```

This is common in encoder-decoder or adapter-based multimodal architectures.

### Query-based resampling

Some systems use learned query vectors to compress many image patch features into fewer visual tokens. This reduces context length and makes image reasoning cheaper.

The exact architecture differs by model, but the principle is similar: image pixels are encoded into vectors first, then those vectors are aligned with the language model's representation space.

---

## 6) A Simple Analogy

Text embedding is like looking up a word in a dictionary:

```text
word ID -> dictionary entry
```

Visual embedding is like asking a trained observer to describe a picture in feature space:

```text
picture -> observer's perception -> feature coordinates
```

The dictionary entry is stable for the same word ID. The observer's output depends on what is actually in the picture.

This is why it is misleading to imagine multimodal models as having a fixed vector table for all possible images. The number of possible images is effectively infinite. The model instead learns a function that maps images to vectors.

---

## 7) Practical Takeaways

- Text tokens usually start from a fixed embedding matrix lookup.
- The same token in the same model has the same initial embedding vector.
- Image embeddings are usually computed online from the current image pixels.
- A new image can produce vectors because the visual encoder has learned a general mapping from pixels to semantic features.
- The final quality depends on training data, architecture, preprocessing, and domain match.
- In multimodal Transformers, visual vectors are often treated as token-like inputs or attended to through cross-attention.

The short version:

> Text embedding begins as a discrete lookup. Visual embedding is a continuous computation over the actual input image.

Understanding this difference makes multimodal architectures much easier to reason about.
