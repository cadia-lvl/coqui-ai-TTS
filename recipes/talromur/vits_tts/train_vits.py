import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config_train = BaseDatasetConfig(formatter="talromur", meta_file_train="index_train_1000.tsv", path=os.path.join(output_path, "../salka/"))
dataset_config_eval = BaseDatasetConfig(formatter="talromur", meta_file_train="index_eval.tsv", path=os.path.join(output_path, "../salka/"))

audio_config = VitsAudioConfig(
    sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_talromur",
    batch_size=8,
    eval_batch_size=8,
    batch_group_size=2,
    num_loader_workers=1,
    num_eval_loader_workers=1,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=True,
    phoneme_language="is",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config_train, dataset_config_eval],
    cudnn_benchmark=False,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, _ = load_tts_samples(
    dataset_config_train,
    eval_split=False,
)
eval_samples, _ = load_tts_samples(
    dataset_config_eval,
    eval_split=False,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
