import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import os
from pydub import AudioSegment
from inference import inference
import torch
import time
import model


device='cpu'

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

*1. Estimate Speaking Rate:* Get the rate of speech in syllables per second.
2. Count Syllables: Find out the total number of syllables in the audio.
3. Provide Inference Time: See how quickly the analysis is performed.

How to Get Started:
Simply send me any audio or voice recording, and I'll analyze it for you. It's that easy!

Let's get started! Send me an audio message now, and I'll do the rest. 📲🎙️'''



logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def Eng_syl_computing(sentence):
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    # unicode_eng_vowels = [1377, 1381, 1383, 1384, 1387, 1400, 1413]
    sentence = sentence.lower()
    syl_count = 0
    for s in sentence:
        if s in vowels:
            syl_count += 1
    return syl_count


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    start_message = (
        "🎤 <strong>Hello! Welcome to the Speaking Rate Estimator Bot!</strong> 🎉\n\n"
        "I'm here to help you analyze the speaking rate of any audio or recording you send me. Whether you're curious about your own speech or analyzing someone else's, I'm here to provide the insights you need.\n\n"
        "🔍 <b>What I Can Do for You:</b>\n"
        "1. <b>Estimate Speaking Rate:</b> Get the rate of speech in syllables per second.\n"
        "2. <b>Count Syllables:</b> Find out the total number of syllables in the audio.\n"
        "3. <b>Provide Inference Time:</b> See how quickly the analysis is performed.\n\n"
        "📋 <b>How to Get Started:</b>\n"
        "Simply send me any audio or voice recording, and I'll analyze it for you. It's that easy\n\n"
        "Let's get started! Send me an audio message now, and I'll do the rest. 📲🎙️"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=start_message, parse_mode='HTML')
    # await context.bot.send_message(chat_id=update.effective_chat.id, text=START)


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

    print(type(pred['speaking_rate']), pred['speaking_rate'])
    print(type(pred['syl_count']), pred['syl_count'])
    print(type(end_time-start_time), end_time-start_time)
    await update.message.reply_text(f'''<b>Speaking rate:</b> {round(pred['speaking_rate'].item(), 4)}\n'''
                                    f'''<b>Syllable count:</b> {round(pred['syl_count'].item(), 4)}\n'''
                                    f'''<b>Inference time:</b> {round(end_time-start_time, 4)} seconds''', 
                                    parse_mode='HTML')
    
async def handle_text(update: Update, context):

    # Get the incoming message text
    text = update.message.text

    # Count the number of syllables in the text
    syllables = Eng_syl_computing(text)

    # Respond with the computed syllable count
    await update.message.reply_text(f"*Actual Syllable count:* {syllables}", parse_mode='MarkdownV2')


if __name__ == '__main__':
    TOKEN = '7176232923:AAG_73mXU5hcbqlIgADJOzkeABPbh0Ur57M'
    application = ApplicationBuilder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    application.run_polling()
    application.idle()
    application.run_polling()