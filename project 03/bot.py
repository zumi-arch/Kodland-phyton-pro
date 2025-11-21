import discord
from discord.ext import commands
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import io
import os 
from model2 import get_class 

# --- Configuration ---
TOKEN = "MTM1NzcxNDEyMjMzODMzNjg0OQ.Ge0tKA.2Ioa0qxHwgUwxQsSW5CZ7CyPuCwZaHtjQD62M0"

intents = discord.Intents.default()
intents.message_content = True 
bot = commands.Bot(command_prefix='!', intents=intents) 

np.set_printoptions(suppress=True)


@bot.event
async def on_ready():
    print(f'bot sudah siap. log in ke {bot.user}')

@bot.command()
async def classify(ctx):
    """
    Command untuk mengklasifikasikan gambar yang diunggah. 
    Sekarang sudah termasuk pemeriksaan file dan format output yang bersih.
    """
    
    if not ctx.message.attachments:
        await ctx.send("Mohon gunakan command ini dengan mengunggah file gambar. Contoh: `!classify` + image file.")
        return

    attachment = ctx.message.attachments[0]
    
    if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        await ctx.send("File yang diunggah tidak terlihat seperti format gambar umum (PNG/JPG/JPEG).")
        return

    await ctx.send("Processing image... this may take a moment.")
    
    image_path = f"./temp_image_{ctx.message.id}_{attachment.filename}" 
    
    try:
        await attachment.save(image_path)
        
        result_tuple = get_class(
            model_path="./keras_model.h5", 
            labels_path="labels.txt", 
            image_path=image_path
        )
        
        if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
            class_name_raw, confidence_score = result_tuple
            
            # Membersihkan class_name_raw (misalnya 'Ini adalah Gas\n' -> 'Gas')
            class_name = class_name_raw.strip().replace("Ini adalah ", "") 

            await ctx.send(
                f"**Classification Result:**\n"
                f"Class: **{class_name}**\n"
                f"Confidence: **{confidence_score:.2f}**" 
            )
        else:
            await ctx.send(f"Format hasil dari `get_class` tidak terduga: {result_tuple}")

    except discord.errors.HTTPException:
        await ctx.send("Tidak bisa mengunduh gambar. Silakan coba lagi.")
    except Exception as e:
        await ctx.send(f"Sebuah error terjadi saat memproses gambar: {e}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
        
if __name__ == '__main__':
    if TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE":
        print("\n*** ERROR: mohon ganti 'YOUR_DISCORD_BOT_TOKEN_HERE' dengan token asli mu. ***\n")
    else:
        try:
            bot.run(TOKEN)
        except discord.LoginFailure:
            print("\n*** ERROR: gagal untuk log in mohon cek TOKEN BOT kamu. ***\n")
        except Exception as e:
            print(f"\n sebuah error terjadi: {e}\n")