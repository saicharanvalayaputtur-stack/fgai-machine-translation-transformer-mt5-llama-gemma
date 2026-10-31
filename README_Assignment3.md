<!-- DELETE THIS LINE. Search this file for the word FILL (Ctrl+F) and replace every marker with your real results before uploading. -->

# Machine Translation (English to French): Transformer vs mT5 vs LLaMA vs Gemma

**Course:** Foundations of Generative AI (ICT4403), Assignment 3
**Author:** SAI CHARAN V, 230911368

## 1. Problem statement and objective
Build machine-translation systems for English to French using four models and compare them: a basic Transformer (encoder-decoder, trained from scratch), mT5 (encoder-decoder, fine-tuned), LLaMA (decoder-only) and Gemma (decoder-only). All four translate the same test set under comparable settings. They are compared on BLEU, chrF, model size, training time and inference time, and the translations are examined for responsible-AI concerns (bias, hallucination, cultural misrepresentation, inappropriate translation of sensitive content).

## 2. Dataset
- **Name:** OPUS Books, English-French (`Helsinki-NLP/opus_books`, config `en-fr`), a publicly available parallel corpus of translated books.
- **Source link:** https://huggingface.co/datasets/Helsinki-NLP/opus_books
- **Size:** **[FILL: raw pair count printed by the notebook]** sentence pairs before filtering.
- **Preprocessing:**
  1. Kept pairs with 3-30 English words and 3-35 French words.
  2. Removed duplicate English sources to avoid train/test leakage.
  3. Shuffled with seed 42.
- **Split:** 10,000 training pairs, 500 validation pairs, 100 test pairs (a manageable subset because of limited compute and time; the notebook has a `QUICK` switch, and setting it to `False` uses 30,000 / 1,000 / 300).
- **Average length:** **[FILL: avg words EN / FR printed by the notebook]**.

## 3. Models implemented

| Model | Architecture | Adaptation | Details |
|---|---|---|---|
| Basic Transformer | Encoder-decoder | Trained from scratch | 3 encoder + 3 decoder layers, d_model 256, 4 heads, feed-forward 1024, pre-norm, tied embeddings, shared SentencePiece BPE vocabulary of 8,000. AdamW (lr 5e-4, 400 warm-up steps, linear decay), label smoothing 0.1, 8 epochs, batch 64, mixed precision, best validation checkpoint kept. |
| mT5-small | Encoder-decoder | Fine-tuned | `google/mt5-small`, input prefix "translate English to French: ", AdamW (lr 3e-4), 1 epoch, batch 16, fp32. |
| LLaMA | Decoder-only | 3-shot prompting (no training) | **[FILL: model actually loaded, `meta-llama/Llama-3.2-1B-Instruct` or the TinyLlama fallback]**, fp16. |
| Gemma | Decoder-only | 3-shot prompting (no training) | **[FILL: model actually loaded, `google/gemma-3-1b-it` or `google/gemma-2-2b-it`]**, fp32. |

Smaller model variants were used because of limited compute (free Colab T4 GPU). LLaMA and Gemma use the same 3 fixed example pairs from the training set in the prompt ("Translate English to French. English: ... French: ...").

## 4. Execution steps and dependencies
1. Create a free Hugging Face account, accept the licences for the LLaMA and Gemma models, create a read token and add it in Colab Secrets as `HF_TOKEN`.
2. Open `FGAI_Assignment3.ipynb` in Google Colab and select Runtime -> Change runtime type -> T4 GPU.
3. Run all cells (about 15-25 minutes in the default quick mode, 45-60 minutes with `QUICK = False`).
4. Dependencies: `torch`, `transformers`, `datasets`, `sentencepiece`, `sacrebleu`, `pandas`, `numpy`, `matplotlib` (everything except `sacrebleu` and `sentencepiece` is preinstalled on Colab; the first cell installs those two).
5. Outputs: `results.csv`, `comparison.png`, `translations.csv`, `probe_translations.csv`.

## 5. Results (test set, 100 sentences)

**[FILL: copy the table from `results.csv`, one row per model]**

| Model | Params (M) | Train time (s) | Inference ms/sentence | BLEU | chrF | Length ratio |
|---|---|---|---|---|---|---|
| Transformer (scratch) | | | | | | |
| mT5-small | | | | | | |
| LLaMA | | | | | | |
| Gemma | | | | | | |

Metrics: corpus BLEU and chrF computed with sacreBLEU against the reference translations. Length ratio is the average of (hypothesis words / reference words). Plot: `comparison.png`.

## 6. Model comparison and evaluation
**[FILL: write 4-6 sentences from your table. Cover these points using your real numbers.]**
- Which model has the highest BLEU and chrF, and which the lowest.
- How the from-scratch Transformer compares with the pretrained models, and why (only 30,000 training pairs and no pretraining).
- Model size versus quality (parameter counts).
- Cost: training time and inference time per sentence (the two decoder-only models were not trained).
- Encoder-decoder versus decoder-only: differences in size, pretraining, tokenization (SentencePiece BPE trained on this data vs the pretrained tokenizers) and adaptation method (training / fine-tuning vs prompting).
- Examples from `translations.csv` that illustrate the numbers.

## 7. Decoding settings (sampling-temperature note)
Machine translation was evaluated with deterministic greedy decoding for all four models (equivalent to temperature 0, no sampling), so that BLEU and chrF are reproducible and comparable. No temperature sweep was run for this assignment. Maximum output length: 64 tokens (Transformer, mT5) and 80 new tokens (LLaMA, Gemma, cut at the first line break).

## 8. Responsible and ethical AI analysis
Twelve probe sentences (`probe_translations.csv`) cover: gender bias with ambiguous pronouns and professions, idioms, culture-specific content, medical dosage and numbers, dates and named entities, politically sensitive wording, politeness/formality, and a request for personal credentials.

**Observations: [FILL: 4-6 sentences from `probe_translations.csv`.]** For example, note which models assign a gender to the doctor or the nurse, which translate the idiom literally, which change numbers, dates or names, and which add information that is not in the source.

**Suggested measures to reduce these concerns:**
- Human review of translations in sensitive domains such as medical, legal and news content, with no fully automatic use.
- Providing gender-alternative translations or flagging ambiguous pronouns instead of silently choosing one.
- Automatic checks that numbers, dates and named entities in the output match the source.
- Glossaries and domain-specific data, plus bias audits with test sentences like the probe set.
- Showing a confidence or "machine-translated" notice to users.
- Avoiding the logging of personal data, and filtering or flagging sensitive content.
- Documenting data sources, limitations and intended use (for example in a model card).

## 9. Conclusion
**[FILL: name the best-performing model for this setup with its BLEU and chrF, and give a one-sentence reason. Then add one sentence on the best trade-off between quality and cost, and one on the main responsible-AI risk you observed.]**

## 10. Limitations
- A subset of the dataset and small model variants were used because of compute limits.
- Each model was trained or prompted once, with a single random seed.
- The test set has only 100 sentences from a literary corpus, so scores are noisy and may not transfer to other domains.
- Quick mode (10,000 training pairs, 8 epochs for the Transformer, 1 epoch for mT5) was used because of the time limit, so the trained models are under-trained compared with the full setting.
- LLaMA and Gemma were prompted with 3 examples and not fine-tuned, so their scores reflect prompting rather than their best possible quality.
- Responsible-AI findings come from 12 hand-written probes and are illustrative, not a full audit.
