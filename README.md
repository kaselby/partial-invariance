# partial-invariance

This is the repository for our paper Learning Functions on Multiple Sets using Multi-Set Transformers. This repository provides implementations of our multi-set models, as well as a trainining pipeline to train on any of the tasks in our paper(KL Divergence, Mutual information, Counting, Alignment and Distinguishabilty).

To train the model, run the main.py script with the appropriate arguments. For example:
```
python3 main.py example_run \
  --model mst --task stat/MI --n 8 \
  --num_blocks 4 --num_heads 4 --latent_size 128 --hidden_size 256 \
  --batch_size 64 --lr 0.0001 --set_size 100 300 \
  --train_steps 100000 --save_every 1000 \
  --normalize whiten \
  --equi --vardim
```
