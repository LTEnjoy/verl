set -x

torchrun --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/sujin/PycharmProjects/verl/data/gsm8k/train.parquet \
    data.val_files=/sujin/PycharmProjects/verl/data/gsm8k/test.parquet \
    data.prompt_key=extra_info.question \
    data.response_key=extra_info.answer \
    data.micro_batch_size_per_gpu=4 \
    data.train_batch_size=64 \
    model.partial_pretrain=/sujin/Models/Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_hdfs_dir=hdfs://user/verl/experiments/gsm8k/deepseek-coder-6.7b-instruct/ \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-Qwen2.5-0.5B-Instruct \
    trainer.total_epochs=4 \
    trainer.logger=['console',"wandb"]