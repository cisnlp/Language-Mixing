# Multilingual Knights-and-Knaves Dataset  


## 💡 Introduction
This dataset extends the [Knights-and-Knaves](https://huggingface.co/datasets/K-and-K/knights-and-knaves) logical reasoning dataset into **five additional languages**:  
- **ar** – Arabic  
- **fr** – French  
- **hi** – Hindi  
- **ja** – Japanese  
- **zh** – Chinese  

The dataset was introduced in our EMNLP 2025 paper:  
➡️ [*Language Mixing in Reasoning Language Models: Patterns, Impact, and Internal Causes*](https://arxiv.org/abs/2505.14815)  

The companion code for reproducing our analyses is available in this repository:  👉 [Language-Mixing (coming soon)](https://github.com/boschresearch/Language-Mixing)  


## 📂 Dataset Structure  

- **Source**: The first 200 training samples from the original English-only dataset.  
- **Translations**: Each sample is translated into the five target languages.  
- **Format**: Saved under the path pattern: ``kk_<LANGUAGE>/<x>ppl/train.json``


## ⚙️ Evaluation  

Use the ``kk_evaluation_multi.py`` script to evaluate on the multilingual dataset.


## 📙 Citation  

If you use this dataset or our paper in your research, please cite:  

```latex
@article{wang2025language,
  title={Language Mixing in Reasoning Language Models: Patterns, Impact, and Internal Causes},
  author={Wang, Mingyang and Lange, Lukas and Adel, Heike and Ma, Yunpu and Strötgen, Jannik and Schütze, Hinrich},
  journal={arXiv preprint arXiv:2505.14815},
  year={2025},
  url={https://arxiv.org/abs/2505.14815}
}

```
