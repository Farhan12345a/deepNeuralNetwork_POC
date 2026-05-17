# Deep neural network (text → price)

This folder contains **`deep_neural_networkF.py`**, **A stack of residual MLP blocks** that maps **bag-of-words–style text features** to a **single positive scalar** (here: a mock “price”).

## What the model does


## What the model does

1. **Text → sparse vector**  
   `HashingVectorizer` turns each `summary` string into a fixed-size feature vector (hashing trick: no vocabulary storage, fast for demos).

2. **Target scaling**  
   Prices are transformed with `log(price + 1)`, then **z-scored** using train-set mean and std. The network predicts that normalized log target.

3. **Network**  
   - Linear projection up to `hidden_size`, LayerNorm, ReLU, Dropout  
   - Repeated **residual blocks** (two linear layers + skip connection + ReLU)  
   - Final linear layer outputs one value  

4. **Training**  
   L1 loss on normalized targets, **AdamW**, **cosine LR schedule**, **gradient clipping**.

5. **Inference**  
   Predict normalized log → invert z-score → `exp` → subtract 1 → clamp at zero.

- Inspired by Ed Donner's LLM Engineering course on Udemy
  
## Layout

| Path | Role |
|------|------|
| `deep_neural_networkF.py` | Model + `DeepNeuralNetworkRunner` |
| `python/demo.py` | Tiny synthetic dataset, short training run, sample prediction |
| `README.md` | This overview |

## Dependencies

```text
numpy torch tqdm scikit-learn
```

## Run the demo

From this directory (`deep_neural_network/`):

```bash
python python/demo.py
```
