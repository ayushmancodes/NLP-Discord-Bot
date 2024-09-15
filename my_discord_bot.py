import discord
from discord.ext import commands
from discord import Member
from discord.ext.commands import has_permissions,MissingPermissions
import requests
from routing.mongo import get_client
import asyncio
import json
from dotenv import load_dotenv
import os
load_dotenv()

mongo_client=get_client()
db = mongo_client['discord_db']
collection = db['toxicity_levels']
token=os.getenv("DISCORD_TOKEN")
bot=commands.Bot(command_prefix="!",intents=discord.Intents.all())
threshold = 50
MUTE_DURATION_SECONDS = 86400

@bot.event
async def on_ready():
    print("Success: Bot is online")

@bot.event
async def on_message(msg):
    if msg.author!=bot.user:
        message_content = msg.content
        
        if(len(message_content.strip())==0):
            return
        
        # Checking for spam data
        data = {'input_value':message_content}
        url = 'http://127.0.0.1:5000/spam'
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            if result['result'][0] == 1:
                await msg.delete()
                user_mention = msg.author.mention
                await msg.channel.send(f"Hello {user_mention}!, Please don't spam")
                return 
        else:
            print(f"Failed to call API. Status code: {response.status_code}, Error: {response.text}")
        
        # Checking for toxicity
        data = {'input_value':message_content}
        url = 'http://127.0.0.1:5000/toxic'
        response = requests.post(url, json=data)
        result=0
        if response.status_code == 200:
            result = response.json()
            result=result["result"].replace("'", '"')
            result=json.loads(result)
            result=result["toxic"]

        else:
            print(f"Failed to call API. Status code: {response.status_code}, Error: {response.text}")
        
        if result <= 0.15 :
            return
        
        elif 0.15 < result < 0.2 :
            await msg.author.send("Hello! This is a warning for the recent message sen't by you.")
        
        elif 0.2 <= result < 0.5 :
            await msg.delete()            
            user_mention = msg.author.mention
            user_id = str(msg.author.id)
            username = msg.author.name
            user = collection.find_one({"user": user_id})
            toxicity_level=1

            if user:
                toxicity_level = user['toxicity_level'] + 1
                if(toxicity_level!=7):
                    collection.update_one(
                        {"user": user_id},
                        {"$set": {"toxicity_level": toxicity_level}}
                    )

            else:
                collection.insert_one({
                    "user": user_id,
                    "toxicity_level": toxicity_level
                })
            
            if(toxicity_level<=3):
                await msg.channel.send(f"Hello {user_mention}!, Don't send toxic messages, otherwise there will be actions\n You have violated {toxicity_level} times")

            if(3<toxicity_level<=6):
                role = discord.utils.get(msg.guild.roles, name='Muted')
                await msg.channel.send(f"{msg.author.mention}, you have been muted for 24 hours due to violation of {toxicity_level} times.")
                await msg.author.add_roles(role)
        
                # Schedule the unmute after 24 hours
                await asyncio.sleep(MUTE_DURATION_SECONDS)
                await msg.author.remove_roles(role)
                await msg.channel.send(f"{msg.author.mention}, you have been unmuted after 24 hours.")

        elif 0.5<=result :
            await msg.channel.send(f"{msg.author.mention}, you have been kicked for a high toxicity level of {result:.2f}.")
            try:
                await msg.author.kick(reason=f"Toxicity level of {result:.2f}")
            except discord.Forbidden:
                await msg.channel.send(f"I don't have permission to kick {msg.author.mention}.")
            except discord.HTTPException as e:
                await msg.channel.send(f"Failed to kick {msg.author.mention}: {e}")

            if result>=0.75:
                await msg.channel.send(f"{msg.author.mention}, you have been banned for continued toxic behavior.")
                try:
                    await msg.author.ban(reason=f"Repeated toxicity. Toxicity level: {result:.2f}")
                except discord.Forbidden:
                    await msg.channel.send(f"I don't have permission to ban {msg.author.mention}.")
                except discord.HTTPException as e:
                    await msg.channel.send(f"Failed to ban {msg.author.mention}: {e}")

    await bot.process_commands(msg)

@bot.command()
async def hello(ctx):
    username = ctx.message.author.mention
    await ctx.send("Hello "+ username)

bot.run(token)