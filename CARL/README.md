## CARL
A **C**ontext-**A**ware **R**einforcement **L**earning framework to improve the performance of LLMs on time-sensitive QA.

### Training
Train the base model with CARL

`./run_train.sh`

Export the model with trained LoRA weights

`python export_hf_checkpoint.py`

Run inference on test sets

`python run_inference.py`

For running evaluations, please refer to the scripts provided [here](https://drive.google.com/drive/folders/1-kcDw39xbatpurJF94_YMprt8I-G8Ami?usp=drive_link).