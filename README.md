# RHEPP-Transformer: Reconstructing Human Expressiveness in Piano Performances with a Transformer Network
This repo presents the code implementation for the paper [Reconstructing Human Expressiveness in Piano Performances with a Transformer Network](https://arxiv.org/abs/2306.06040)
## Training
The training was monitored by with [W&B](https://api.wandb.ai/links/tangjingjingbetsy/4j1gjpx5). The pre-trained model could be found and downloaded [here](https://drive.google.com/drive/folders/1YKU3V_UxbZILHyfPQ9nfMgj905pxh04l?usp=sharing). 

For re-training the model, please contact me for the data and run the following commands:
```
python main.py --cuda_devices YOUR_CUDA_DEVICES
```
## Generation
For generate expressive piano performance from transcribed score (in MIDI format), run:
```
python inference.py --ckpt_path PATH_TO_MODEL --input_file PATH_TO_INPUT_MIDI --output_file PATH_TO_OUTPUT_FILE
```
## Citation
```
@article{tang2023reconstructing,
  title={Reconstructing Human Expressiveness in Piano Performances with a Transformer Network},
  author={Tang, Jingjing and Wiggins, Geraint and Fazekas, George},
  journal={arXiv preprint arXiv:2306.06040},
  year={2023}
}
```
## Contact
Jingjing Tang: jingjing.tang@qmul.ac.uk


