# Composition Attention Grammar (CAG)

This repository provides code for training and testing Composition Attention Grammars (CAGs).

> [Composition, Attention, or Both?](https://arxiv.org/abs/2210.12958) <br>
> Ryo Yoshida and Yohei Oseki <br>
> Findings of EMNLP 2022 <br>

## Requirement
- `python==3.8.10`

## Installation
```shell
git clone https://github.com/osekilab/CAG.git
cd CAG
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_and_patch_transformers.sh
```

## Data preparation
```shell
python src/preprocess.py \
    --unkmethod gpt2 \
    --keep_ptb_bracket \
    --trainfile data/train.txt \
    --valfile data/valid.txt \
    --testfile data/test.txt \
    --outputfile data/example
```

## Training
- Setup
    ```shell
    accelerate config
    ```
- Launch
    ```shell
    accelerate launch src/train.py \
        --train_file data/example-train.json \
        --val_file data/example-val.json \
        --tokenizer_model data/example-tokenizer \
        --fixed_stack \
        --strategy top_down \
        --w_dim 256 \
        --h_dim 256 \
        --num_layers 3 \
        --num_heads 4 \
        --max_stack_size 100 \
        --dropout 0.1 \
        --embd_dropout 0.1 \
        --batch_group similar_action_length \
        --optimizer adam \
        --save_path cag.pt \
        --num_epochs 15 \
        --lr 0.001 \
        --amp \
        --early_stop \
        --early_stop_patience 100 \
        --tensorboard_log_dir logs \
        --batch_size 32 \
        --seed 3435 \
        --print_every 500 \
        --batch_token_size 100000 \
        --batch_action_size 100000 \
        --max_group_length_diff 100000
    ```

## Inference
```shell
python src/beam_search.py \
    --test_file data/test_tokens.txt \
    --lm_output_file surprisals.txt \
    --model_file cag.pt \
    --beam_size 100 \
    --word_beam_size 10 \
    --shift_size 5 \
    --device cpu > trees.txt
```

## Estimate word-level perplexity
```shell
python src/estimate_word_ppl.py \
    --test_file data/example-test.json \
    --model_file cag.pt \
    --device cpu
```

## Credits
This repository is based on the [Pytorch implementation of RNNG](https://github.com/aistairc/rnng-pytorch) by [Noji and Oseki (2021)](https://aclanthology.org/2021.findings-acl.380/).

## License
MIT

## Note
If you want to download CAGs trained in our paper, please contact `yoshiryo0617 [at] g.ecc.u-tokyo.ac.jp`
