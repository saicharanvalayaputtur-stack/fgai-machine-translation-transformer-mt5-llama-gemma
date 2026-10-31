# English to French machine translation comparison

Foundations of Generative AI (ICT4403), Assignment 3.

**Status: implementation prepared; full four-model experiment has not yet been executed. No benchmark scores or winning model are claimed. Do not submit as a completed experiment until the run and human review are finished.**

## Problem and objective

Translate English sentences into French with four systems: a basic encoder-decoder Transformer, mT5, LLaMA and Gemma. Measure translation quality and computational cost on the same held-out sentences, inspect responsible-AI risks, and study the effect of sampling temperature. The decoder-only systems are autoregressive text generators used for translation, satisfying both the main task and the submission wording.

## Dataset and preprocessing

Source: [Helsinki-NLP OPUS Books, en-fr](https://huggingface.co/datasets/Helsinki-NLP/opus_books), with [original OPUS Books description](https://opus.nlpl.eu/Books.php). This is sentence-aligned literary parallel text; it does not represent all modern French or conversational domains. Review the dataset card and original book rights before redistributing source text.

The code pins the verified dataset commit `1f9f6191d0e91a3c539c2595e2fe48fc1420de9b` and records it, normalizes Unicode to NFC, collapses whitespace, keeps punctuation/case/accents, and excludes empty or long pairs (3-400 characters and 1-60 whitespace words per side). It removes repeated English **or** French sentences case-insensitively before splitting. With seed 42, it shuffles and selects **10,000 train / 1,000 validation / 500 test** pairs. Counts and hashes are recorded in `data/manifest.json`. Dataset preparation was executed successfully: 127,085 raw pairs; 5,190 rejected by length/empty filters; 1,534 rejected for a duplicate side; 120,361 retained before subset selection. The exact 10,000/1,000/500 splits were generated and their hashes saved in `results/dataset_manifest.json`. Exact train/test leakage is checked on both languages. Sentence-level splitting does not prevent shared book context or pretrained-model contamination.

The basic model's joint byte-level BPE vocabulary is trained only on the training split. Other systems retain their pretrained tokenizers. Token limits can truncate differently; counts are logged, so compare them before interpreting scores. No test example appears in few-shot prompts.

## Systems and training design

| System | Architecture | Adaptation | Key settings |
| --- | --- | --- | --- |
| Basic Transformer | Encoder-decoder, custom PyTorch module | Random initialization, supervised training | 3 encoder + 3 decoder layers; d_model 256; 4 heads; FFN 1024; dropout 0.1; 8,000-token target BPE vocabulary; AdamW 3e-4; 10 epochs |
| google/mt5-small | Encoder-decoder | Full fine-tuning | Adafactor 1e-4, 3 epochs, float32, gradient checkpointing |
| meta-llama/Llama-3.2-1B-Instruct | Decoder-only | Three-shot prompting | Three short training examples, native chat template |
| google/gemma-3-1b-it | Decoder-only | Three-shot prompting | Exactly the same examples and prompt content, native chat template |

Mini-batch 4, gradient accumulation 4, gradient norm clipping 1.0. Best validation cross-entropy checkpoint is saved, with padding ignored. Decoder inputs are shifted target tokens and causal attention prevents future-token leakage. The encoders attend bidirectionally to English; the decoders attend to previous French tokens and encoder outputs. Prompted models generate French conditioned on the prompt prefix. Full parameter counts are measured from loaded models, not inferred from names.

The small scratch model is a resource-limited baseline and may need more data/epochs for useful quality. Tune only with validation evidence, then freeze the settings before test evaluation. Do not promise that these defaults maximize scores.

## Execution

Use Python 3.11 or 3.12 in a clean environment and a GPU machine. A 16 GB GPU is a reasonable starting target for this configuration, but actual availability and memory use must be checked. Models are loaded sequentially, with no paid service automatically provisioned. Model downloads require internet and disk space. CPU execution is possible but the full experiment can be impractically slow.

1. Open `FGAI_Assignment3.ipynb` in Google Colab and select a GPU runtime. The notebook embeds the source files, so cloning a repository is unnecessary.
2. Install dependencies using the notebook cell. If the runtime already imported old package versions, restart the runtime after installation and resume at project setup.
3. On Hugging Face, review and accept the access terms for [LLaMA](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and [Gemma](https://huggingface.co/google/gemma-3-1b-it), request access if needed, and log in with your own read token. Never paste tokens into committed code or notebook outputs.
4. Run cells in order. Download results/checkpoints before the runtime disconnects. Training time is measured, not promised in advance.

Local alternative:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
huggingface-cli login
python run.py --stage prepare
python run.py --stage train --model basic
python run.py --stage train --model mt5
python run.py --stage evaluate --model basic
python run.py --stage evaluate --model mt5
python run.py --stage evaluate --model llama
python run.py --stage evaluate --model gemma
python run.py --stage report
```

Or run `python run.py --stage all` once. Re-running `all` retrains models. Staged commands let you retain finished checkpoints; training is not resumable mid-epoch. If changing the data size or seed, prepare again and retrain both trained models. Checkpoint manifest checks prevent comparison against different splits. Use a separate copy for development smoke runs so that they cannot overwrite final results.

## Fair evaluation protocol

- Same ordered test IDs for all models; output texts and references saved together.
- Greedy generation, beam size 1, batch size 1, maximum 128 new tokens. Preserve raw decoded content except surrounding whitespace and special tokens. Do not silently remove explanations/refusals that would affect quality.
- Source limits: 254 content tokens for basic, 256 tokens for mT5; decoder prompts must fit 1,024 tokens. Flag truncation and outputs hitting the generation cap.
- Case-sensitive corpus BLEU with SacreBLEU 13a tokenization; chrF with character orders 1-6 and beta 2. Signatures are recorded. Scores are on a 0-100 scale. chrF is the prespecified primary rank metric, BLEU the tie-breaker.
- GPU synchronization surrounds timers. Inference includes tokenization, autoregressive generation and decoding; excludes download, model loading and one warmup. Training time includes validation and best-checkpoint saving. It excludes dataset preparation, model loading and tokenizer training. Prompting has N/A training time, not a fabricated zero-cost pretraining claim.
- Hardware, package versions, dtype, model revision, total parameters and adapted parameters are saved. Basic/mT5 use float32; decoder models use bfloat16 where supported, else float16 on GPU. These precision and system-design differences limit causal claims about architecture.
- Peak CUDA allocated memory is additional evidence, not total GPU reserved/system memory. Multiple inference repetitions and confidence intervals would strengthen a future experiment.

## Results and model comparison

<!-- RESULTS_START -->
Pending the actual four-model run. `python run.py --stage report` inserts measured results here and writes `results/REPORT.md`, `comparison.csv`, temperature tables, figures and translation samples. It refuses to report a winner if any model is missing.
<!-- RESULTS_END -->

## Sampling temperature analysis

The main benchmark uses deterministic greedy decoding. A separate analysis uses the same first 100 test sentences for every model at temperatures 0.3, 0.7 and 1.0, top-p 0.9, no top-k cutoff, and seeds 42/43/44. Temperature changes token probabilities by softmax(logits/T): lower values concentrate probability; higher values flatten it. The report shows mean and seed standard deviation for BLEU/chrF, with greedy scores on that same subset. Never compare a 100-sentence sampled score directly with a 500-sentence headline score. Do not choose a best test setting and retroactively present it as prespecified.

## Responsible AI

See `ETHICS.md`, the eight explicitly synthetic probes in `ethics_probes.json`, and generated `results/human_review.csv`. Review 20 test examples per system plus all probes. Record actual evidence for bias, hallucination, cultural misrepresentation and sensitive-content errors. General risk discussion alone does not complete the required examination of generated translations.

## Conclusion

The best model cannot be identified before measured results exist. After the run, the generated comparison identifies the highest observed chrF model, reports BLEU and cost, and states the limits of the conclusion. Add a final paragraph discussing whether its quality gain justifies its measured inference cost, together with concrete human-review findings.

## Repository contents and submission

`src/` contains data processing, the basic model, and metrics. `run.py` trains/evaluates; `report.py` creates results; `tests/` checks leakage, metrics, causal masking and toy learning. `FGAI_Assignment3.ipynb` is the self-contained execution notebook. `submission/` contains the cover document and PDF draft. `results/local_tests.txt` is a development check, **not translation benchmark evidence**.

Before submission: complete the run, review all outputs, fill the ethics review, commit source/README/results/plots/samples to GitHub, verify evaluator access, insert the real repository URL in the Word cover sheet, export it as PDF, and upload that final PDF to LMS. Do not publish access tokens, downloaded model weights, or the private cover sheet unintentionally. The repository must remain accessible to the evaluator. No repository URL should be invented.

## Sources

- Assignment brief: Foundations of Generative AI ICT4403 Assignment 3, supplied PDF.
- Dataset: https://huggingface.co/datasets/Helsinki-NLP/opus_books
- mT5: https://huggingface.co/google/mt5-small
- LLaMA: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Gemma: https://huggingface.co/google/gemma-3-1b-it
- Transformer: https://arxiv.org/abs/1706.03762
- SacreBLEU: https://github.com/mjpost/sacrebleu
- PyTorch Transformer: https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html

Use AI assistance in accordance with your course policy and be prepared to explain the implementation and measured results.
