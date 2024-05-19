import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import os
from pydub import AudioSegment
from inference import inference
import torch
import time
import model

# # Check if MPS is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("MPS device is available.")
# else:
#     device = torch.device("cpu")
#     print("MPS device is not available. Using CPU.")

device='cpu'

# try:
#     # Load the model on the CPU
#     model = torch.load(model_path, map_location=torch.device('cpu'))
#     print("Model loaded successfully on CPU.")
    
#     # Move the model to the MPS device if available
#     model.to(device)
#     print(f"Model moved to {device}.")
    
# except RuntimeError as e:
#     print(f"Error loading model: {e}")


HELP = '''
Welcome to the Speaking Rate Estimator Bot! 🤖

This chatbot estimates the speaking rate of your input audio or recording. Here are the details of what it can do:

1. Speaking Rate: Measures the rate of speech in syllables per second (syl/sec).
2. Syllable Count: Counts the total number of syllables in your input speech.
3. Inference Time: Provides the time taken to process your audio and deliver the results.

How to Use:
Simply send me your audio or voice recording, and I will analyze it to give you the following metrics:

Speaking Rate (syl/sec)
Syllable Count
Inference Time (sec)

Feel free to test it by sending an audio message now!
'''

START = '''Hello! Welcome to the Speaking Rate Estimator Bot! 🎉

I'm here to help you analyze the speaking rate of any audio or recording you send me. Whether you're curious about your own speech or analyzing someone else's, I'm here to provide the insights you need.

What I Can Do for You:

1. Estimate Speaking Rate: Get the rate of speech in syllables per second.
2. Count Syllables: Find out the total number of syllables in the audio.
3. Provide Inference Time: See how quickly the analysis is performed.

How to Get Started:
Simply send me any audio or voice recording, and I'll analyze it for you. It's that easy!

Let's get started! Send me an audio message now, and I'll do the rest. 📲🎙️'''



logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await context.bot.send_message(chat_id=update.effective_chat.id, text=START)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(HELP)

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    audio_file = update.message.audio or update.message.voice
    if not audio_file:
        await update.message.reply_text('Please send an audio file.')
        return

    file = await context.bot.get_file(audio_file.file_id)
    ogg_dir = './audios_ogg'
    os.makedirs(ogg_dir, exist_ok=True)
    ogg_path = os.path.join(ogg_dir, f'{audio_file.file_id}.ogg')
    await file.download_to_drive(ogg_path)

    wav_dir = './audios_wav'
    os.makedirs(wav_dir, exist_ok=True)
    wav_path = os.path.join(wav_dir, f'{audio_file.file_id}.wav')
    audio = AudioSegment.from_file(ogg_path, format="ogg")
    audio.export(wav_path, format="wav")


    my_model = model.MatchBoxNetreg(B=3, R=2, C=112)
    path = '/Users/dianasimonyan/Desktop/Thesis/torch_implementation/SREregression/models_2sec/rMatchBoxNet-3x2x112/checkpoints/best-epoch=198-val_loss=1.50-val_pcc=0.93.ckpt'
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    my_model.load_state_dict(state_dict['state_dict'])

    my_model.to(device)

    start_time = time.time()
    pred = inference(my_model, audio_path=wav_path)
    end_time = time.time()


    # Your speaking rate estimation logic here
    # speaking_rate = estimate_speaking_rate(local_path)
    await update.message.reply_text(#f"Speaking rate: {pred['speaking_rate']}",
                                    f"Syllable count: {pred['syl_count']}")
                                    # f"Inference time: {end_time-start_time}sec")


if __name__ == '__main__':
    TOKEN = '7176232923:AAG_73mXU5hcbqlIgADJOzkeABPbh0Ur57M'
    application = ApplicationBuilder().token(TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    
    application.run_polling()
    application.idle()
    application.run_polling()

    # my_model = model.MatchBoxNetreg(B=3, R=2, C=112)
    # path = '/Users/dianasimonyan/Desktop/Thesis/torch_implementation/SREregression/models_2sec/rMatchBoxNet-3x2x112/checkpoints/best-epoch=198-val_loss=1.50-val_pcc=0.93.ckpt'
    # state_dict = torch.load(path, map_location=torch.device('cpu'))
    # my_model.load_state_dict(state_dict['state_dict'])

    # my_model.to(device)

    # start_time = time.time()
    # pred = inference(my_model, audio_path='/Users/dianasimonyan/Desktop/Thesis/torch_implementation/SRE_demo/audios_wav/AwACAgIAAxkBAAM9Zko8AAH7PCZhSeiq_ZWMnuhqPlEaAAJuVQACWRNQSkK2XG3R338TNQQ.wav')
    # end_time = time.time()

    # print(f"Speaking rate: {pred['speaking_rate']}",
    #     f"Syllable count: {pred['syl_count']}",
    #     f"Inference time: {end_time-start_time}sec")